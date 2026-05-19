"""
Orchestrates the full preprocessing sequence with hash-based cache invalidation.

Pipeline order (leakage-safe):
  1. Parse     — era5_parser / saws_parser → (T, N, d) array + coords
  2. Impute    — IDW missing value fill (ERA5 skip: no missing values)
  3. Split     — chronological 60/20/20 (MUST precede label/normalise)
  4. Label     — 95th-percentile thresholds computed on train only
  5. Normalise — z-score stats computed on train only
  6. Graph     — adjacency matrix from node lat/lon coordinates
  7. DataLoaders — sliding-window Dataset + DataLoader factory

Hash check: SHA-256 fingerprint of raw file mtimes + preprocessing config.
On a match, cached .npy files are loaded directly. On a miss (or --no-cache),
the pipeline runs, results are saved to preprocessing/cache/, and the hash
file is updated.
"""

import hashlib
import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader

from preprocessing.era5_parser  import parse_era5_files
from preprocessing.impute        import exclude_sparse_nodes, idw_impute
from preprocessing.split         import chronological_split
from preprocessing.labels        import label_all_splits
from preprocessing.normalise     import normalise_all_splits
from preprocessing.graph         import build_adjacency_matrix
from preprocessing.dataset       import build_dataloaders

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
HASH_FILE = CACHE_DIR / "preprocessing_hash.json"

ERA5_RAW_DIR = Path(__file__).parent.parent / "data" / "era5" / "raw"
SAWS_RAW_DIR = Path(__file__).parent.parent / "data" / "saws" / "raw"


# ---------------------------------------------------------------------------
# Hash utilities
# ---------------------------------------------------------------------------

def _file_fingerprints(directory: Path, pattern: str) -> list:
    files = sorted(directory.glob(pattern))
    return [(str(f), f.stat().st_mtime, f.stat().st_size) for f in files if f.exists()]


def compute_data_hash(dataset: str, config: Dict[str, Any]) -> str:
    """SHA-256 fingerprint of raw data file mtimes + preprocessing config keys."""
    h = hashlib.sha256()
    if dataset == "era5":
        fps = _file_fingerprints(ERA5_RAW_DIR, "era5_southafrica_*.nc")
    elif dataset == "saws":
        fps = _file_fingerprints(SAWS_RAW_DIR, "*.csv")
    else:
        fps = []
    h.update(json.dumps(fps, sort_keys=True).encode())
    preproc_keys = ["input_len", "horizon", "train_ratio", "val_ratio",
                    "lat_stride", "lon_stride", "distance_threshold_km",
                    "sigma_km"]
    h.update(json.dumps({k: config.get(k) for k in preproc_keys}, sort_keys=True).encode())
    h.update(dataset.encode())
    return h.hexdigest()


def _cache_paths(dataset: str) -> Dict[str, Path]:
    prefix = CACHE_DIR / dataset
    return {
        "X_train":    prefix / "X_train.npy",
        "X_val":      prefix / "X_val.npy",
        "X_test":     prefix / "X_test.npy",
        "y_train":    prefix / "y_train.npy",
        "y_val":      prefix / "y_val.npy",
        "y_test":     prefix / "y_test.npy",
        "adj":        prefix / "adj.npy",
        "coords":     prefix / "coords.npy",
        "mean":       prefix / "mean.npy",
        "std":        prefix / "std.npy",
        "thresholds": prefix / "thresholds.npy",
        "meta":       prefix / "meta.json",
    }


def _cache_exists(dataset: str) -> bool:
    return all(p.exists() for p in _cache_paths(dataset).values())


def load_cache(dataset: str) -> Dict[str, Any]:
    paths = _cache_paths(dataset)
    result = {k: np.load(str(v)) for k, v in paths.items() if v.suffix == ".npy"}
    with open(paths["meta"]) as f:
        result["meta"] = json.load(f)
    return result


def save_cache(dataset: str, arrays: Dict[str, Any]) -> None:
    prefix = CACHE_DIR / dataset
    prefix.mkdir(parents=True, exist_ok=True)
    for key, value in arrays.items():
        if key == "meta":
            with open(prefix / "meta.json", "w") as f:
                json.dump(value, f, indent=2)
        else:
            np.save(str(prefix / f"{key}.npy"), value)


def _check_hash(dataset: str, current_hash: str) -> bool:
    if not HASH_FILE.exists():
        return False
    with open(HASH_FILE) as f:
        stored = json.load(f)
    return stored.get(dataset) == current_hash


def _write_hash(dataset: str, current_hash: str) -> None:
    HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if HASH_FILE.exists():
        with open(HASH_FILE) as f:
            existing = json.load(f)
    existing[dataset] = current_hash
    with open(HASH_FILE, "w") as f:
        json.dump(existing, f, indent=2)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    config: Dict[str, Any],
    force_rerun: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, Dict[str, Any]]:
    """
    Entry point called by cli.py. Returns:
        train_loader, val_loader, test_loader,
        adjacency_matrix (N, N),
        preprocessing_metadata {var_names, n_nodes, n_vars, mean, std, thresholds}

    Args:
        config:      merged config dict from YAML + CLI overrides.
        force_rerun: skip hash check and always rerun (--no-cache flag).
    """
    dataset     = config.get("dataset", "era5")
    input_len   = config["data"]["input_len"]
    horizon     = config["data"]["horizon"]
    task        = config.get("task", "regression")
    batch_size  = config["training"]["batch_size"]
    num_workers = config.get("num_workers", 0)

    preproc = config.get("preprocessing", {})
    train_ratio           = preproc.get("train_ratio", 0.6)
    val_ratio             = preproc.get("val_ratio", 0.2)
    lat_stride            = preproc.get("lat_stride", 2)
    lon_stride            = preproc.get("lon_stride", 2)
    distance_threshold_km = preproc.get("distance_threshold_km", 300.0)
    sigma_km              = preproc.get("sigma_km", 100.0)
    exclusion_threshold   = preproc.get("exclusion_threshold", 0.05)
    # Clip normalized feature values to ±clip_sigma standard deviations.
    # ERA5 total precipitation (tp) is extremely right-skewed — raw z-scores
    # reach 57σ, causing NaN in model activations after a few gradient steps.
    # 10σ retains all physically meaningful variance while preventing instability.
    clip_sigma = float(preproc.get("clip_sigma", 10.0))

    flat_config = {**preproc, "input_len": input_len, "horizon": horizon}
    current_hash = compute_data_hash(dataset, flat_config)

    if not force_rerun and _check_hash(dataset, current_hash) and _cache_exists(dataset):
        log.info("Cache hit — loading preprocessed arrays from disk.")
        cached = load_cache(dataset)
        adj  = cached["adj"]
        meta = cached["meta"]
        X_train = np.clip(cached["X_train"], -clip_sigma, clip_sigma)
        X_val   = np.clip(cached["X_val"],   -clip_sigma, clip_sigma)
        X_test  = np.clip(cached["X_test"],  -clip_sigma, clip_sigma)
        train_loader, val_loader, test_loader = build_dataloaders(
            X_train, X_val, X_test,
            cached["y_train"], cached["y_val"],  cached["y_test"],
            input_len=input_len, horizon=horizon, task=task,
            batch_size=batch_size, num_workers=num_workers,
        )
        meta.update({"mean": cached["mean"], "std": cached["std"],
                     "thresholds": cached["thresholds"],
                     "coords": cached["coords"]})
        return train_loader, val_loader, test_loader, adj, meta

    # 1. Parse -----------------------------------------------------------
    log.info("Step 1/6: Parsing raw %s data…", dataset)
    if dataset == "era5":
        X, timestamps, coords, var_names = parse_era5_files(
            str(ERA5_RAW_DIR), lat_stride=lat_stride, lon_stride=lon_stride,
        )
    elif dataset == "saws":
        from preprocessing.saws_parser import parse_saws_files
        X, timestamps, coords, var_names = parse_saws_files(str(SAWS_RAW_DIR))
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    # 2. Impute ----------------------------------------------------------
    log.info("Step 2/6: Imputing missing values…")
    if np.isnan(X).any():
        X, coords = exclude_sparse_nodes(X, coords, exclusion_threshold)
        X = idw_impute(X, coords)
    else:
        log.info("  No missing values — imputation skipped.")

    # 3. Split -----------------------------------------------------------
    log.info("Step 3/6: Chronological split (%.0f/%.0f/%.0f)…",
             train_ratio * 100, val_ratio * 100, (1 - train_ratio - val_ratio) * 100)
    X_train, X_val, X_test = chronological_split(X, train_ratio, val_ratio)

    # 4. Label -----------------------------------------------------------
    log.info("Step 4/6: 95th-percentile extreme-event labels…")
    y_train, y_val, y_test, thresholds = label_all_splits(X_train, X_val, X_test)

    # 5. Normalise -------------------------------------------------------
    log.info("Step 5/6: Z-score normalisation (stats from train only)…")
    X_train, X_val, X_test, mean, std = normalise_all_splits(X_train, X_val, X_test)

    # 6. Graph -----------------------------------------------------------
    log.info("Step 6/6: Building adjacency matrix (threshold=%.0f km)…",
             distance_threshold_km)
    adj = build_adjacency_matrix(
        coords,
        distance_threshold_km=distance_threshold_km,
        sigma_km=sigma_km,
    )

    # Save ---------------------------------------------------------------
    meta = {
        "var_names": var_names,
        "n_nodes":   int(adj.shape[0]),
        "n_vars":    len(var_names),
        "dataset":   dataset,
        "task":      task,
        "input_len": input_len,
        "horizon":   horizon,
    }
    save_cache(dataset, {
        "X_train": X_train, "X_val": X_val,   "X_test": X_test,
        "y_train": y_train, "y_val": y_val,   "y_test": y_test,
        "adj":     adj,
        "coords":  coords[["lat", "lon"]].values.astype(np.float32),
        "mean":    mean,
        "std":     std,
        "thresholds": thresholds,
        "meta":    meta,
    })
    _write_hash(dataset, current_hash)
    log.info("Preprocessing complete — cache written to %s", CACHE_DIR / dataset)

    train_loader, val_loader, test_loader = build_dataloaders(
        np.clip(X_train, -clip_sigma, clip_sigma),
        np.clip(X_val,   -clip_sigma, clip_sigma),
        np.clip(X_test,  -clip_sigma, clip_sigma),
        y_train, y_val, y_test,
        input_len=input_len, horizon=horizon, task=task,
        batch_size=batch_size, num_workers=num_workers,
    )
    coords_arr = coords[["lat", "lon"]].values.astype(np.float32)
    meta.update({"mean": mean, "std": std, "thresholds": thresholds, "coords": coords_arr})
    return train_loader, val_loader, test_loader, adj, meta
