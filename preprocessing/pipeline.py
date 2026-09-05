"""
Orchestrates the full preprocessing sequence with manifest-based cache invalidation.

Pipeline order (leakage-safe):
  1. Parse     — era5_parser / saws_parser → (T, N, d) array + coords
  2. Impute    — IDW missing value fill (ERA5 skip: no missing values)
  3. Split     — chronological 60/20/20 (MUST precede label/normalise)
  4. Label     — 95th-percentile thresholds computed on train only
  5. Normalise — z-score stats computed on train only
  6. Graph     — adjacency matrix from node lat/lon coordinates
  7. DataLoaders — sliding-window Dataset + DataLoader factory

Hash check: SHA-256 of the set of month identifiers (filename stems) on disk plus
preprocessing config keys.  Crucially this is *NOT* mtime-based — re-touching or
re-downloading a file without changing the set of months does not invalidate the cache.
Only adding new monthly files (or changing a config key) triggers a rebuild.

Use verify_cache_coverage(dataset) at any time to check whether the cache covers
all months currently on disk without rebuilding.
"""

import hashlib
import json
import logging
import os
import resource
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _rss_gib() -> float:
    """Resident-set size in GiB. Linux ru_maxrss is in KiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

from torch.utils.data import DataLoader

from preprocessing.era5_parser  import parse_era5_files
from preprocessing.derive        import derive_era5_variables
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
# SAWS 32-station block-format workbooks for 2016-2025 (locked 2026-08-08).
# See preprocessing/saws_parser.py header for the network definition. The
# earlier Lekuba long-form workbooks (data1.xlsx, data2.xls) that drove the
# 25-station 2013-2020 campaign live at data/saws/extracted and are still
# parsed by preprocessing/saws_parser_25node_lekuba.py — not this pipeline.
SAWS_RAW_DIR = (
    Path(__file__).parent.parent
    / "data" / "saws" / "raw" / "SAWS-60-node-campaign" / "SAWS" / "extracted"
)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _manifest_fingerprint(directory: Path, pattern: str) -> List[str]:
    """
    Return sorted filename stems for all matching files.

    Stable across mtime and size changes — a resumed download or a filesystem
    re-touch does NOT change this value.  Only adding or removing files does.
    """
    return sorted(f.stem for f in directory.glob(pattern) if f.is_file())


def _era5_month_list(directory: Path) -> List[Tuple[int, int]]:
    """
    Return sorted [(year, month), ...] for ERA5 monthly .nc files.

    Used to populate meta.json so the cache's temporal coverage is always
    human-readable without re-parsing.
    """
    months = []
    for f in sorted(directory.glob("era5_southafrica_*.nc")):
        parts = f.stem.split("_")   # era5_southafrica_YYYY_MM
        try:
            months.append((int(parts[-2]), int(parts[-1])))
        except (ValueError, IndexError):
            pass
    return sorted(months)


# ---------------------------------------------------------------------------
# Hash utilities
# ---------------------------------------------------------------------------

def _preproc_defaults_for(dataset: str) -> Dict[str, Any]:
    """Dataset-aware defaults matching the run-time branching in run_pipeline.

    Used by compute_data_hash so equivalent-but-differently-expressed configs
    hash the same. Without this, a config that omits `distance_threshold_km`
    (relying on the pipeline's default of 500 for SAWS) hashes differently
    from a config that sets it explicitly to 500 — the previous bug that
    silently caused cache misses whenever cli.py loaded a model yaml without
    an explicit `preprocessing:` block. Bug found 2026-08-08 on the HPO smoke.
    """
    common = {
        "input_len":        None,   # required from config["data"]; no default
        "horizon":          None,   # required from config["data"]; no default
        "train_ratio":      0.6,
        "val_ratio":        0.2,
        "lat_stride":       2,
        "lon_stride":       2,
        "derive_variables": True,
    }
    if dataset == "saws":
        common.update({
            "distance_threshold_km": 500.0,
            "sigma_km":              200.0,
            "exclusion_threshold":   0.10,   # Davidson §3.2
        })
    else:  # era5 (or unknown → era5-shaped)
        common.update({
            "distance_threshold_km": 300.0,
            "sigma_km":              100.0,
            "exclusion_threshold":   0.05,
        })
    return common


def compute_data_hash(dataset: str, config: Dict[str, Any]) -> str:
    """
    SHA-256 fingerprint of raw data manifest (filename stems) + preprocessing
    config keys with dataset-aware defaults applied.

    Filename stems (e.g. 'era5_southafrica_2010_01') are stable identifiers —
    they change only when new months are added, not when existing files are
    re-touched or re-downloaded.

    Preprocessing keys are resolved against `_preproc_defaults_for(dataset)`
    so that:
      config = {"input_len": 12, "horizon": 48}
    and
      config = {"input_len": 12, "horizon": 48, "distance_threshold_km": 500.0, ...}
    hash identically for SAWS. This prevents silent cache invalidation when
    cli.py loads a model yaml without an explicit `preprocessing:` block.
    """
    h = hashlib.sha256()
    if dataset == "era5":
        manifest = _manifest_fingerprint(ERA5_RAW_DIR, "era5_southafrica_*.nc")
    elif dataset == "saws":
        manifest = (_manifest_fingerprint(SAWS_RAW_DIR, "*.xlsx")
                    + _manifest_fingerprint(SAWS_RAW_DIR, "*.xls"))
    else:
        manifest = []
    h.update(json.dumps(manifest, sort_keys=True).encode())

    defaults = _preproc_defaults_for(dataset)
    resolved = {k: (config.get(k) if config.get(k) is not None else defaults.get(k))
                for k in defaults}
    h.update(json.dumps(resolved, sort_keys=True).encode())
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
    # tp_wet_trigger is Phase-2; older caches don't have it. Load if present.
    prefix = CACHE_DIR / dataset
    tp_wet_path = prefix / "tp_wet_trigger.npy"
    if tp_wet_path.exists():
        result["tp_wet_trigger"] = np.load(str(tp_wet_path))
    # Calendar-hour scheme artefacts (Phase-2 v2). Present iff
    # scripts/build_ch_labels.py has been run. Optional — the caller
    # decides which scheme to use via config.data.threshold_scheme.
    for name in ("thresholds_ch", "y_train_ch", "y_val_ch", "y_test_ch",
                 "tp_wet_trigger_ch"):
        p = prefix / f"{name}.npy"
        if p.exists():
            result[name] = np.load(str(p))
    ch_meta_path = prefix / "labels_ch_meta.json"
    if ch_meta_path.exists():
        with open(ch_meta_path) as f:
            result["labels_ch_meta"] = json.load(f)
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
# Cache coverage verification
# ---------------------------------------------------------------------------

def verify_cache_coverage(dataset: str) -> Dict[str, Any]:
    """
    Compare what the cache covers against what is on disk.

    Returns a dict:
      status:             "ok" | "stale" | "no_cache"
      n_cached:           number of months in the cache manifest
      n_on_disk:          number of monthly files on disk right now
      missing_from_cache: [(year, month), ...] on disk but absent from cache
      extra_in_cache:     [(year, month), ...] in cache but not on disk
                          (should never happen; indicates a cache corruption)
    """
    if not _cache_exists(dataset):
        return {
            "status":             "no_cache",
            "n_cached":           0,
            "n_on_disk":          0,
            "missing_from_cache": [],
            "extra_in_cache":     [],
        }

    cached = load_cache(dataset)
    cached_months = set(map(tuple, cached["meta"].get("manifest", [])))

    if dataset == "era5":
        disk_months = set(_era5_month_list(ERA5_RAW_DIR))
    else:
        disk_months = set()  # SAWS manifest not yet implemented

    missing = sorted(disk_months - cached_months)
    extra   = sorted(cached_months - disk_months)

    return {
        "status":             "ok" if not missing else "stale",
        "n_cached":           len(cached_months),
        "n_on_disk":          len(disk_months),
        "missing_from_cache": missing,
        "extra_in_cache":     extra,
    }


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
        preprocessing_metadata {var_names, n_nodes, n_vars, mean, std, thresholds,
                                 manifest, date_range, n_months}

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

    # Phase 2 classification knobs. Regression ignores both. Defaults chosen
    # per Phase-2 plan §2.1 and §2.4.
    data_cfg        = config.get("data", {})
    label_mode      = data_cfg.get("label_mode", "per_variable")
    label_variables = data_cfg.get("label_variables", None)
    # Regression: univariate target per Gaibie (2024) — None means multivariate
    # output (all features predicted jointly, legacy behaviour).
    target_variable = data_cfg.get("target_variable", None)

    preproc = config.get("preprocessing", {})
    train_ratio           = preproc.get("train_ratio", 0.6)
    val_ratio             = preproc.get("val_ratio", 0.2)
    lat_stride            = preproc.get("lat_stride", 2)
    lon_stride            = preproc.get("lon_stride", 2)
    # Dataset-aware defaults: SAWS is a 25-node sparse station network with
    # ~500 km typical spacing, so ERA5's 300 km/100 km graph defaults leave
    # some Cape stations (e.g. ALEXANDERBAAI, 449 km from nearest neighbour)
    # with zero adjacency edges. Its missingness threshold also matches
    # Davidson (2022)'s <10% rather than ERA5's stricter 5%.
    if dataset == "saws":
        distance_threshold_km = preproc.get("distance_threshold_km", 500.0)
        sigma_km              = preproc.get("sigma_km", 200.0)
        exclusion_threshold   = preproc.get("exclusion_threshold", 0.10)
    else:
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
        print(f"[mem] {_rss_gib():.2f} GiB before cache load", flush=True)
        log.info("Cache hit — loading preprocessed arrays from disk.")
        cached = load_cache(dataset)
        print(f"[mem] {_rss_gib():.2f} GiB after load_cache", flush=True)
        adj  = cached["adj"]
        meta = cached["meta"]
        # np.clip returns a fresh array; release the raw cached copies
        # immediately (each is 1.5–4.4 GB at production scale) so we don't
        # sit on both the raw and clipped versions during dataset build.
        X_train = np.clip(cached.pop("X_train"), -clip_sigma, clip_sigma)
        X_val   = np.clip(cached.pop("X_val"),   -clip_sigma, clip_sigma)
        X_test  = np.clip(cached.pop("X_test"),  -clip_sigma, clip_sigma)
        # Threshold-scheme dispatch (Phase-2 §2 update, 2026-08-13). Static is
        # the legacy behaviour and remains the default; calendar_hour_relative
        # uses the ETCCDI-style windowed-percentile artefacts built by
        # scripts/build_ch_labels.py. The scheme lives in config.data because
        # it changes labels, not features — no cache hash change; static and
        # ch artefacts coexist so Phase-1 metrics remain reproducible.
        threshold_scheme = (config.get("data", {}) or {}).get(
            "threshold_scheme", "static"
        )
        if threshold_scheme == "calendar_hour_relative":
            required = ("y_train_ch", "y_val_ch", "y_test_ch", "thresholds_ch")
            missing = [k for k in required if k not in cached]
            if missing:
                raise FileNotFoundError(
                    f"threshold_scheme='calendar_hour_relative' but the ch "
                    f"artefacts are absent from the {dataset} cache "
                    f"(missing: {missing}). Run scripts/build_ch_labels.py "
                    f"once to populate them (~30s), then re-run."
                )
            y_train = cached.pop("y_train_ch"); _ = cached.pop("y_train", None)
            y_val   = cached.pop("y_val_ch");   _ = cached.pop("y_val",   None)
            y_test  = cached.pop("y_test_ch");  _ = cached.pop("y_test",  None)
            thresholds_out = cached["thresholds_ch"]
        elif threshold_scheme == "static":
            y_train = cached.pop("y_train")
            y_val   = cached.pop("y_val")
            y_test  = cached.pop("y_test")
            thresholds_out = cached["thresholds"]
        else:
            raise ValueError(
                f"data.threshold_scheme={threshold_scheme!r} not recognised. "
                f"Expected 'static' or 'calendar_hour_relative'."
            )
        print(f"[mem] {_rss_gib():.2f} GiB after clip + pop", flush=True)
        train_loader, val_loader, test_loader = build_dataloaders(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            input_len=input_len, horizon=horizon, task=task,
            batch_size=batch_size, num_workers=num_workers,
            label_mode=label_mode, var_names=meta.get("var_names"),
            label_variables=label_variables,
            target_variable=target_variable,
        )
        print(f"[mem] {_rss_gib():.2f} GiB after build_dataloaders", flush=True)
        meta.update({"mean": cached["mean"], "std": cached["std"],
                     "thresholds": thresholds_out,
                     # static_thresholds is always the 2-D (N, d) per-station
                     # 95th-percentile array, regardless of threshold_scheme.
                     # Regression stratification (mae_extreme / mae_nonextreme /
                     # peak_intensity_error) uses this directly so it is never
                     # accidentally wired to the 4-D calendar-hour-relative tensor.
                     "static_thresholds": cached["thresholds"],
                     "threshold_scheme": threshold_scheme,
                     "coords": cached["coords"],
                     "label_mode": label_mode,
                     "label_variables": label_variables or meta.get("var_names"),
                     "target_variable": target_variable,
                     "tp_wet_trigger": cached.get("tp_wet_trigger_ch")
                                         if threshold_scheme == "calendar_hour_relative"
                                         else cached.get("tp_wet_trigger")})
        if threshold_scheme == "calendar_hour_relative" and "labels_ch_meta" in cached:
            meta["labels_ch_meta"] = cached["labels_ch_meta"]
        return train_loader, val_loader, test_loader, adj, meta

    # 1. Parse -----------------------------------------------------------
    log.info("Step 1/6: Parsing raw %s data…", dataset)
    if dataset == "era5":
        X, timestamps, coords, var_names = parse_era5_files(
            str(ERA5_RAW_DIR), lat_stride=lat_stride, lon_stride=lon_stride,
        )
        manifest = _era5_month_list(ERA5_RAW_DIR)
    elif dataset == "saws":
        from preprocessing.saws_parser import parse_saws_files
        X, timestamps, coords, var_names = parse_saws_files(str(SAWS_RAW_DIR))
        manifest = []
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    # 1b. Derive ---------------------------------------------------------
    derive = preproc.get("derive_variables", True)
    if dataset == "era5" and derive:
        log.info("Step 1b/6: Deriving ws10 and rh from raw ERA5 variables…")
        X, var_names = derive_era5_variables(X, var_names)
        log.info("  var_names=%s  d=%d", var_names, X.shape[-1])

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
    y_train, y_val, y_test, thresholds, tp_wet_trigger = label_all_splits(
        X_train, X_val, X_test, var_names=var_names,
    )
    if tp_wet_trigger.any():
        log.info("  tp wet-hour rule fired at %d / %d stations",
                 int(tp_wet_trigger.sum()), tp_wet_trigger.shape[0])

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
    date_range = {}
    if manifest:
        date_range = {
            "start": f"{manifest[0][0]}-{manifest[0][1]:02d}",
            "end":   f"{manifest[-1][0]}-{manifest[-1][1]:02d}",
        }

    meta = {
        "var_names":  var_names,
        "n_nodes":    int(adj.shape[0]),
        "n_vars":     len(var_names),
        "dataset":    dataset,
        "task":       task,
        "input_len":  input_len,
        "horizon":    horizon,
        "manifest":   manifest,
        "n_months":   len(manifest),
        "date_range": date_range,
    }
    save_cache(dataset, {
        "X_train": X_train, "X_val": X_val,   "X_test": X_test,
        "y_train": y_train, "y_val": y_val,   "y_test": y_test,
        "adj":     adj,
        "coords":  coords[["lat", "lon"]].values.astype(np.float32),
        "mean":    mean,
        "std":     std,
        "thresholds":     thresholds,
        "tp_wet_trigger": tp_wet_trigger,
        "meta":    meta,
    })
    _write_hash(dataset, current_hash)
    log.info("Preprocessing complete — cache written to %s", CACHE_DIR / dataset)
    log.info("  Coverage: %s → %s  (%d months)",
             date_range.get("start", "?"), date_range.get("end", "?"), len(manifest))

    train_loader, val_loader, test_loader = build_dataloaders(
        np.clip(X_train, -clip_sigma, clip_sigma),
        np.clip(X_val,   -clip_sigma, clip_sigma),
        np.clip(X_test,  -clip_sigma, clip_sigma),
        y_train, y_val, y_test,
        input_len=input_len, horizon=horizon, task=task,
        batch_size=batch_size, num_workers=num_workers,
        label_mode=label_mode, var_names=var_names,
        label_variables=label_variables,
        target_variable=target_variable,
    )
    coords_arr = coords[["lat", "lon"]].values.astype(np.float32)
    meta.update({
        "mean": mean, "std": std, "thresholds": thresholds, "coords": coords_arr,
        "label_mode": label_mode,
        "label_variables": label_variables or var_names,
        "target_variable": target_variable,
        "tp_wet_trigger": tp_wet_trigger,
    })
    return train_loader, val_loader, test_loader, adj, meta
