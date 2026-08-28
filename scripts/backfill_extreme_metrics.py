"""
Backfill mae_extreme / mae_nonextreme / peak_intensity_error for regression runs.

Root cause: clcrn.yaml sets threshold_scheme=calendar_hour_relative, causing
_build_extreme_mask to receive a 4-D tensor it skipped → all three stratified
metrics became NaN.  The code fix (pipeline.py / trainer.py / cli.py) prevents
this going forward.  This script is a one-off backfill for already-completed runs.

Algorithm per run:
  1. Load checkpoint.pt (best-epoch weights).
  2. Run inference on test set — same forward path as training.
  3. GATE: assert plain MAE / RMSE / sMAPE / NRMSE-curve reproduce stored values
     to within REL_TOL.  Abort the run (not the whole script) on failure.
  4. Build extreme mask using cached["thresholds"] (static 2-D, always correct).
  5. Compute mae_extreme, mae_nonextreme, peak_intensity_error.
  6. Back up and patch summary.json; widen per_station.npz; save predictions +
     targets; write provenance block.

Usage:
    python scripts/backfill_extreme_metrics.py [--device cpu|cuda] [--dry-run]
        [--models clcrn] [--force]

    --models   comma-separated model names, or 'all'  (default: clcrn)
    --force    re-patch runs already showing non-NaN extreme metrics
    --dry-run  run gate only, write nothing

Requirements: same environment as training.
"""

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.regression import (
    nrmse_per_horizon,
    peak_intensity_error,
    per_station_var_mae,
    per_station_var_rmse,
    per_station_var_smape,
    stratified_var_mae,
)
from preprocessing.dataset import STGNNDataset
from preprocessing.pipeline import load_cache
from torch.utils.data import DataLoader

REL_TOL = 5e-3   # relative tolerance for gate (CPU/GPU float precision: ~1-5e-4 typical)
REG_VARS = {"t2m", "sp", "rh", "ws"}


# ── adapter registry ─────────────────────────────────────────────────────────

def _get_adapter(model_name: str):
    if model_name == "clcrn":
        from models.baselines.clcrn_adapter import CLCRNAdapter
        return CLCRNAdapter
    if model_name == "agcrn":
        from models.baselines.agcrn_adapter import AGCRNAdapter
        return AGCRNAdapter
    if model_name == "mtgode":
        from models.continuous_time.mtgode_adapter import MTGODEAdapter
        return MTGODEAdapter
    if model_name == "stgode":
        from models.continuous_time.stgode_adapter import STGODEAdapter
        return STGODEAdapter
    raise KeyError(f"unknown model: {model_name!r}")


def _get_config(model_name: str) -> dict:
    import yaml
    cfg_path = ROOT / "experiments" / "configs" / f"{model_name}.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Force static thresholds so the pipeline reads cached["thresholds"] (2-D).
    cfg.setdefault("data", {})["threshold_scheme"] = "static"
    return cfg


# ── helpers ──────────────────────────────────────────────────────────────────

def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-9)


def _sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_summary(run_dir: Path) -> dict:
    with open(run_dir / "summary.json") as f:
        return json.load(f)


def _save_summary(run_dir: Path, s: dict) -> None:
    with open(run_dir / "summary.json", "w") as f:
        json.dump(s, f, indent=2)


# ── per-run backfill ──────────────────────────────────────────────────────────

def backfill_run(
    run_dir: Path,
    cached: dict,
    var_names: list,
    device: torch.device,
    dry_run: bool,
    force: bool,
) -> str:
    """Returns: 'patched' | 'patched(dry)' | 'gate_fail' | 'skip_already_done' | 'no_checkpoint'"""
    summary = _load_summary(run_dir)
    tm = summary["test_metrics"]

    model_name = summary.get("model") or run_dir.name.split("_")[0]
    target_var = summary["target"]
    seed       = summary["seed"]

    # Skip if already fully patched (extreme metrics valid AND per-horizon present) and not forcing
    existing_ext = tm.get("mae_extreme", {})
    has_valid_extreme = existing_ext and not all(np.isnan(v) for v in existing_ext.values())
    has_per_horizon   = target_var in tm.get("mae_per_horizon", {})
    if not force and has_valid_extreme and has_per_horizon:
        return "skip_already_done"

    checkpoint = run_dir / "checkpoint.pt"
    if not checkpoint.exists():
        return "no_checkpoint"

    # ── validate thresholds ──────────────────────────────────────────────────
    thresholds_2d = cached["thresholds"]
    assert thresholds_2d.ndim == 2, (
        f"expected static (N,d) thresholds, got shape {thresholds_2d.shape}; "
        "cached['thresholds'] may be the CH tensor")
    N_cache, d_cache = thresholds_2d.shape
    assert d_cache == len(var_names), \
        f"threshold cols {d_cache} != len(var_names) {len(var_names)}"

    mean_np = cached["mean"]   # (1, N, d)
    std_np  = cached["std"]    # (1, N, d)

    # ── build test DataLoader ────────────────────────────────────────────────
    config     = _get_config(model_name)
    input_len  = config["data"]["input_len"]
    horizon    = config["data"]["horizon"]
    batch_size = config["training"]["batch_size"]

    tv_idx_int = var_names.index(target_var)
    ds_test = STGNNDataset(
        cached["X_test"], labels=None,
        input_len=input_len, horizon=horizon, task="regression",
        target_variable_index=tv_idx_int,
    )
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)

    # ── load model weights ───────────────────────────────────────────────────
    torch.manual_seed(seed)
    adapter_cls = _get_adapter(model_name)
    adapter     = adapter_cls()
    adj_np      = cached["adj"]
    adj_t       = torch.tensor(adj_np, dtype=torch.float32).to(device)

    # Reproduce the same meta dict that run_pipeline returns and cli.py puts into config
    run_meta = dict(cached["meta"])
    run_meta.update({
        "coords":           cached["coords"],
        "mean":             cached["mean"],
        "std":              cached["std"],
        "thresholds":       cached["thresholds"],
        "static_thresholds": cached["thresholds"],
    })
    config["meta"] = run_meta
    adapter.build_model(config, adj_np)
    n_params = sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
    print(f"  model={model_name}  target={target_var}  seed={seed}  n_params={n_params:,}")
    adapter.model.to(device)

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        saved_epoch = state.get("epoch")
        best_epoch  = summary.get("best_epoch") or summary.get("stopped_epoch")
        if saved_epoch is not None and saved_epoch != best_epoch:
            print(f"  WARNING: checkpoint epoch {saved_epoch} != summary best_epoch {best_epoch}")
        state = state["model_state_dict"]
    adapter.model.load_state_dict(state)
    adapter.model.eval()

    # ── inference ────────────────────────────────────────────────────────────
    all_preds, all_targets = [], []
    t0 = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x    = x.to(device)
            pred = adapter.predict(x, adj_t)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())
    elapsed_s = time.time() - t0
    preds   = np.concatenate(all_preds,   axis=0)   # (S, H, N, d_model)
    targets = np.concatenate(all_targets, axis=0)   # (S, H, N, 1)

    # ── slice to target variable ─────────────────────────────────────────────
    if preds.shape[-1] > targets.shape[-1]:
        preds = preds[..., tv_idx_int:tv_idx_int + 1]
    eval_mean = mean_np[..., tv_idx_int:tv_idx_int + 1]
    eval_std  = std_np[...,  tv_idx_int:tv_idx_int + 1]
    thr_2d    = thresholds_2d[:, tv_idx_int:tv_idx_int + 1]   # (N, 1)

    # ── denormalise ──────────────────────────────────────────────────────────
    m = eval_mean.reshape(1, 1, preds.shape[2], 1)
    s = eval_std.reshape( 1, 1, preds.shape[2], 1)
    p_phys = preds   * s + m
    t_phys = targets * s + m

    # ── compute metrics ──────────────────────────────────────────────────────
    mae_nd   = per_station_var_mae(p_phys, t_phys)    # (N, 1)
    rmse_nd  = per_station_var_rmse(p_phys, t_phys)
    smape_nd = per_station_var_smape(p_phys, t_phys)
    # nrmse uses z-scored arrays; normalised by training σ → matches stored value
    nrmse_c  = nrmse_per_horizon(preds, targets)       # (H, 1) z-space

    # per-horizon physical metrics — free: same forward pass, keep the H axis
    _err_h  = p_phys[..., 0] - t_phys[..., 0]            # (S, H, N)
    mae_h   = np.nanmean(np.abs(_err_h), axis=(0, 2))     # (H,) physical
    rmse_h  = np.sqrt(np.nanmean(_err_h ** 2, axis=(0, 2)))  # (H,) physical

    recomputed_mae   = float(np.nanmean(mae_nd[:, 0]))
    recomputed_rmse  = float(np.nanmean(rmse_nd[:, 0]))
    recomputed_smape = float(np.nanmean(smape_nd[:, 0]))

    stored_mae   = tm["mae"].get(target_var, float("nan"))
    stored_rmse  = tm["rmse"].get(target_var, float("nan"))
    stored_smape = tm["smape"].get(target_var, float("nan"))

    gate_results = {
        "mae":   (_rel_err(recomputed_mae,   stored_mae),   stored_mae,   recomputed_mae),
        "rmse":  (_rel_err(recomputed_rmse,  stored_rmse),  stored_rmse,  recomputed_rmse),
        "smape": (_rel_err(recomputed_smape, stored_smape), stored_smape, recomputed_smape),
    }

    # NRMSE curve gate — 48-point match is much stronger than a scalar coincidence
    if "nrmse_per_horizon" in tm and target_var in tm["nrmse_per_horizon"]:
        stored_nrmse  = np.asarray(tm["nrmse_per_horizon"][target_var], dtype=float)
        recomp_nrmse  = np.asarray(nrmse_c[:, 0], dtype=float)
        if recomp_nrmse.shape == stored_nrmse.shape:
            nrmse_rel = np.abs(recomp_nrmse - stored_nrmse) / np.maximum(np.abs(stored_nrmse), 1e-9)
            gate_results["nrmse_curve"] = (
                float(nrmse_rel.max()), float(stored_nrmse[-1]), float(recomp_nrmse[-1])
            )
        else:
            print(f"  WARNING: nrmse shape mismatch stored={stored_nrmse.shape} recomputed={recomp_nrmse.shape}, skipping curve gate")

    gate_pass  = all(v[0] <= REL_TOL for v in gate_results.values())
    gate_lines = [
        f"    {mn}: stored={v[1]:.6f}  recomputed={v[2]:.6f}  rel_err={v[0]:.2e}  {'OK' if v[0]<=REL_TOL else 'FAIL'}"
        for mn, v in gate_results.items()
    ]
    print(f"  gate ({elapsed_s:.1f}s inference):")
    for line in gate_lines:
        print(line)

    if not gate_pass:
        print("  → GATE FAIL — skipping patch")
        return "gate_fail"

    if dry_run:
        print("  → dry-run: would patch")
        return "patched(dry)"

    # ── compute extreme metrics ───────────────────────────────────────────────
    mask4d       = t_phys > thr_2d[None, None, :, :]           # (S, H, N, 1)
    mae_ext_nd, mae_non_nd = stratified_var_mae(p_phys, t_phys, mask4d)
    pie_nd       = peak_intensity_error(p_phys, t_phys, mask4d)

    new_ext = float(np.nanmean(mae_ext_nd[:, 0]))
    new_non = float(np.nanmean(mae_non_nd[:, 0]))
    new_pie = float(np.nanmean(pie_nd[:, 0]))
    print(f"  → mae_extreme={new_ext:.4f}  mae_nonextreme={new_non:.4f}  pie={new_pie:.4f}")

    # consistency check: mean(mae_per_horizon) should equal pooled mae to float tol
    implied_mae  = float(np.mean(mae_h))
    implied_rmse = float(np.mean(rmse_h))
    relerr_mae  = abs(implied_mae  - recomputed_mae)  / max(abs(recomputed_mae),  1e-9)
    relerr_rmse = abs(implied_rmse - recomputed_rmse) / max(abs(recomputed_rmse), 1e-9)
    print(f"  per-horizon consistency: mae_rel={relerr_mae:.2e}  rmse_rel={relerr_rmse:.2e}")

    # ── backup + patch summary.json ───────────────────────────────────────────
    shutil.copy2(run_dir / "summary.json", run_dir / "summary.json.pre_backfill")

    tm["mae_extreme"][target_var]          = new_ext
    tm["mae_nonextreme"][target_var]       = new_non
    tm["peak_intensity_error"][target_var] = new_pie
    tm.setdefault("mae_per_horizon",  {})[target_var] = mae_h.tolist()
    tm.setdefault("rmse_per_horizon", {})[target_var] = rmse_h.tolist()
    summary["test_metrics"] = tm
    summary["n_params"] = n_params

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    summary["backfill"] = {
        "script":            Path(__file__).name,
        "git_commit":        git_commit,
        "timestamp_utc":     datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint_sha256": _sha256(checkpoint),
        "fields_written":    ["mae_extreme", "mae_nonextreme", "peak_intensity_error",
                              "mae_per_horizon", "rmse_per_horizon",
                              "n_params", "per_station.npz", "test_preds.npy", "test_targets.npy"],
        "gate_max_rel_err":  max(v[0] for v in gate_results.values()),
        "gate_tolerance":    REL_TOL,
        "device":            str(device),
    }
    _save_summary(run_dir, summary)

    # ── widen per_station.npz ────────────────────────────────────────────────
    npz_path = run_dir / "per_station.npz"
    existing_npz = dict(np.load(str(npz_path), allow_pickle=False)) if npz_path.exists() else {}
    existing_npz.update({
        "mae":                  mae_nd,
        "rmse":                 rmse_nd,
        "smape":                smape_nd,
        "mae_extreme":          mae_ext_nd,
        "mae_nonextreme":       mae_non_nd,
        "peak_intensity_error": pie_nd,
        "nrmse_per_horizon":    nrmse_c,
        "mae_per_horizon":      mae_h[:, np.newaxis],    # (H, 1) station-pooled
        "rmse_per_horizon":     rmse_h[:, np.newaxis],   # (H, 1) station-pooled
    })
    np.savez(str(npz_path), **existing_npz)

    # ── save raw predictions + targets ───────────────────────────────────────
    np.save(str(run_dir / "test_preds.npy"),   preds.astype(np.float32))
    np.save(str(run_dir / "test_targets.npy"), targets.astype(np.float32))

    return "patched"


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backfill extreme metrics + per-station arrays + n_params"
    )
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run gate only, write nothing")
    parser.add_argument("--force",   action="store_true",
                        help="Re-patch runs that already have non-NaN extreme metrics")
    parser.add_argument("--models",  default="clcrn",
                        help="Comma-separated model names, or 'all'  (default: clcrn)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}  dry-run: {args.dry_run}  force: {args.force}  models: {args.models}")

    # Determinism — CLCRN gather/scatter ops are non-deterministic otherwise
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # ── load shared cache once ────────────────────────────────────────────────
    cached = load_cache("saws")
    cached["X_test"] = np.clip(cached["X_test"], -10.0, 10.0)

    var_names = list(
        cached.get("var_names")
        or (cached.get("meta") or {}).get("var_names")
        or []
    )
    if not var_names:
        raise RuntimeError(
            "cache has no var_names (checked cached['var_names'] and cached['meta']['var_names']); "
            "re-run the preprocessing pipeline to regenerate the cache"
        )
    assert len(var_names) == 6, f"expected 6 SAWS variables, got {len(var_names)}: {var_names}"
    print(f"var_names: {var_names}")

    thresholds_2d = cached["thresholds"]
    assert thresholds_2d.ndim == 2, \
        f"cached['thresholds'] has shape {thresholds_2d.shape}, expected (N, d)"

    # ── determine target model set ────────────────────────────────────────────
    ALL_MODELS = {"agcrn", "clcrn", "mtgode", "stgode"}
    target_models = ALL_MODELS if args.models == "all" else {
        m.strip() for m in args.models.split(",")
    }
    unknown = target_models - ALL_MODELS
    if unknown:
        raise ValueError(f"unknown models: {unknown}")

    # ── enumerate regression run directories ──────────────────────────────────
    runs_dir = ROOT / "runs"
    candidate_runs = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        parts = d.name.split("_")
        # expected: {model}_{var}_s{seed}
        if len(parts) != 3:
            continue
        model_name = parts[0]
        var        = parts[1]
        if model_name not in target_models or var not in REG_VARS:
            continue
        if not (d / "summary.json").exists():
            continue
        candidate_runs.append((d, model_name))

    print(f"\nFound {len(candidate_runs)} regression run directories to process:\n")
    for r, _ in candidate_runs:
        print(f"  {r.name}")
    print()

    if not candidate_runs:
        print("Nothing to backfill.")
        return

    # ── process each run ──────────────────────────────────────────────────────
    results: dict[str, str] = {}
    for run_dir, model_name in candidate_runs:
        print(f"\n{'='*60}")
        print(f"  {run_dir.name}")
        status = backfill_run(
            run_dir=run_dir,
            cached=cached,
            var_names=var_names,
            device=device,
            dry_run=args.dry_run,
            force=args.force,
        )
        results[run_dir.name] = status
        print(f"  → {status}")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BACKFILL SUMMARY")
    print(f"{'='*60}")
    counts = Counter(results.values())
    for status, n in sorted(counts.items()):
        print(f"  {status:25s}: {n}")

    gate_fails = [k for k, v in results.items() if v == "gate_fail"]
    if gate_fails:
        print("\nGATE FAILURES (NOT patched — investigate before re-running):")
        for k in gate_fails:
            print(f"  {k}")
        print("\nTip: if rel_err is in the 1e-3 to 1e-5 range on CUDA, retry with --device cpu")
        sys.exit(1)


if __name__ == "__main__":
    main()
