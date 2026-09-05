"""
Compute persistence and climatology baselines from the preprocessed cache.

Baselines are seed-independent (no training), so compute once per dataset.
Metrics reported per variable and per horizon step (matches the schema used
by the trainer's compute_all output for models, but computed chunk-by-chunk
so we never allocate the full (samples, H, N, d) tensor in memory at once).

Persistence (regression):
    ŷ(t+h) = y(t)  — repeat the last input step H times.

Climatology (regression):
    ŷ(t+h) = mean of training set per (station, variable). Unconditional.

Persistence (classification):
    label̂(t+h) = label(t)  — repeat last observed extreme label as probability
    (0 or 1). Threshold-free metrics (AUC-ROC, PR-AUC) still defined.

Climatology (classification):
    Constant per-(station, variable) training positive rate as probability.

Output: results/baselines/{regression,classification}/{persistence,climatology}.json

Memory strategy: chunk over samples axis. Peak RSS should be <20 GB for
chunk=2000, well within any HPC allocation.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Dataset-specific defaults (2026-08-06 SAWS 25-node campaign added).
# ERA5: 12h horizon, 999-node, ERA5 var lineage.
# SAWS: 48h horizon (Moodley 24h+ demand), 25-node, SAWS var lineage.
DATASET_DEFAULTS = {
    "era5": {
        "cache_subdir": "era5",
        "horizon": 12,
        "label_vars": ["t2m", "d2m", "tp", "ws10", "rh", "sp"],
    },
    "saws": {
        "cache_subdir": "saws",
        "horizon": 48,
        "label_vars": ["t2m", "tp", "rh", "wd", "ws", "sp"],
    },
}
INPUT_LEN = 12
VPT_EPSILON = 0.3
CHUNK = 2000   # samples per chunk

# Set from CLI at main() entry; module globals kept for backward compatibility
# with the internal helpers that reference them by name.
CACHE: Path = Path()
OUT: Path = Path()
HORIZON: int = 0
LABEL_VARS: list = []


def load_arrays():
    print("Loading cache...", flush=True)
    X_train = np.load(CACHE / "X_train.npy", mmap_mode="r")
    X_test  = np.load(CACHE / "X_test.npy",  mmap_mode="r")
    y_train = np.load(CACHE / "y_train.npy", mmap_mode="r")
    y_test  = np.load(CACHE / "y_test.npy",  mmap_mode="r")
    mean    = np.load(CACHE / "mean.npy")
    std     = np.load(CACHE / "std.npy")
    thresholds = np.load(CACHE / "thresholds.npy")
    with open(CACHE / "meta.json") as f:
        meta = json.load(f)
    var_names = meta["var_names"]
    print(f"  X_test  {X_test.shape}  {X_test.dtype}")
    print(f"  y_test  {y_test.shape}  {y_test.dtype}")
    print(f"  var_names: {var_names}")
    return X_train, X_test, y_train, y_test, mean, std, thresholds, var_names


def denorm_broadcasts(mean, std, shape_4d):
    """Return mean, std broadcast to (1, 1, N, d) shape for arithmetic."""
    m = np.asarray(mean).reshape(1, 1, shape_4d[2], shape_4d[3]).astype(np.float32)
    s = np.asarray(std).reshape(1, 1, shape_4d[2], shape_4d[3]).astype(np.float32)
    return m, s


# ---------------------------------------------------------------------------
# Regression — chunked over samples
# ---------------------------------------------------------------------------

def _persistence_pred(X_test, chunk_start, chunk_end):
    """(chunk, H, N, d) NORMALISED persistence prediction for samples [chunk_start:chunk_end]."""
    # For sample i, the input window is X_test[i : i+INPUT_LEN] and target is X_test[i+INPUT_LEN : i+INPUT_LEN+HORIZON].
    # Persistence pred at each horizon step h = X_test[i + INPUT_LEN - 1] (the last observed input).
    last_obs = X_test[INPUT_LEN - 1 + chunk_start : INPUT_LEN - 1 + chunk_end]   # (chunk, N, d)
    last_obs = np.asarray(last_obs, dtype=np.float32)
    # Tile across horizon axis
    return np.broadcast_to(last_obs[:, None, :, :], (last_obs.shape[0], HORIZON, *last_obs.shape[1:])).copy()


def _target(X_test, chunk_start, chunk_end):
    """(chunk, H, N, d) NORMALISED target for samples [chunk_start:chunk_end]."""
    starts = np.arange(chunk_start, chunk_end)
    # Vectorised slice: X_test[i+INPUT_LEN : i+INPUT_LEN+HORIZON]
    # Build via stack (small in a chunk of 2000)
    return np.stack([np.asarray(X_test[i + INPUT_LEN : i + INPUT_LEN + HORIZON], dtype=np.float32) for i in starts], axis=0)


def _climatology_pred(train_mean_norm, chunk_size):
    """Constant climatology prediction, (chunk, H, N, d)."""
    return np.broadcast_to(
        train_mean_norm[None, None, :, :],
        (chunk_size, HORIZON, *train_mean_norm.shape),
    ).copy()


def regression_baseline_metrics(pred_fn, name, X_test, mean, std, thresholds, var_names):
    print(f"\n=== Regression {name} ===", flush=True)
    T, N, d = X_test.shape
    n_samples = T - INPUT_LEN - HORIZON + 1
    print(f"  {n_samples} sliding windows, chunk={CHUNK}", flush=True)

    # Accumulators — all in physical units
    # per (H, N, d): sum of |err|, sum of err^2, count
    abs_err_sum = np.zeros((HORIZON, N, d), dtype=np.float64)
    sq_err_sum  = np.zeros((HORIZON, N, d), dtype=np.float64)
    # per-horizon per-var NRMSE numerator = sum of (err_norm)^2 → used for horizon curve
    sq_err_norm_sum_hd = np.zeros((HORIZON, d), dtype=np.float64)   # summed over samples and N
    count_hd = 0
    # Extreme-stratified per (N, d): sum |err| where mask==True, count(mask==True), and same for non-extreme
    ext_err_sum   = np.zeros((N, d), dtype=np.float64)
    ext_count     = np.zeros((N, d), dtype=np.int64)
    nonext_err_sum= np.zeros((N, d), dtype=np.float64)
    nonext_count  = np.zeros((N, d), dtype=np.int64)
    # Peak-intensity error per (N, d): |max_pred_when_ext - max_target_when_ext| — we approximate with mean over extreme samples
    thresh_bcast = thresholds[None, None, :, :]

    m_1n_d = mean.reshape(1, 1, N, d).astype(np.float32)
    s_1n_d = std.reshape(1, 1, N, d).astype(np.float32)

    for cstart in range(0, n_samples, CHUNK):
        cend = min(cstart + CHUNK, n_samples)
        # Predictions and targets in NORMALISED space
        preds_norm  = pred_fn(cstart, cend)                                       # (c, H, N, d)
        target_norm = _target(X_test, cstart, cend)                                # (c, H, N, d)
        # Denormalise to PHYSICAL units
        preds_phys  = preds_norm  * s_1n_d + m_1n_d
        target_phys = target_norm * s_1n_d + m_1n_d
        # Errors
        err = preds_phys - target_phys                                             # (c, H, N, d)
        abs_err_sum += np.sum(np.abs(err), axis=0)                                 # (H, N, d)
        sq_err_sum  += np.sum(err * err, axis=0)
        # Normalised error for NRMSE-per-horizon (over samples and stations)
        err_norm = preds_norm - target_norm
        sq_err_norm_sum_hd += np.sum(err_norm * err_norm, axis=(0, 2))             # (H, d)
        count_hd += (cend - cstart) * N
        # Extreme stratification
        mask = target_phys > thresh_bcast                                          # (c, H, N, d) bool
        abs_err = np.abs(err)
        # sum |err| where mask, over samples and horizon
        ext_contrib     = (abs_err * mask).sum(axis=(0, 1))                        # (N, d)
        ext_count_delta = mask.sum(axis=(0, 1))
        ext_err_sum += ext_contrib
        ext_count   += ext_count_delta
        nonext_contrib     = (abs_err * (~mask)).sum(axis=(0, 1))
        nonext_count_delta = (~mask).sum(axis=(0, 1))
        nonext_err_sum += nonext_contrib
        nonext_count   += nonext_count_delta
        if cstart % (CHUNK * 4) == 0 or cend == n_samples:
            pct = 100.0 * cend / n_samples
            print(f"  chunk {cstart:>6d}–{cend:>6d}  ({pct:5.1f}%)", flush=True)

    total_count_per_hnd = n_samples   # per (H, N, d) each element has n_samples measurements
    # Aggregate to per-variable scalars (physical units, mean over samples, horizon, stations)
    def per_var(sum_hnd):
        # sum_hnd: (H, N, d) — total sum
        return sum_hnd.mean(axis=(0, 1))   # (d,) — mean per var (equivalent to /(H*N))
        # Actually, we already summed over samples; total elements per (H, N, d) index = n_samples.
        # per_var mean over samples, H, N = sum / (n_samples * H * N)

    # Correct denominator for per-variable means:
    n_elements_per_var = n_samples * HORIZON * N   # total elements averaged over
    mae_per_var  = {var_names[v]: float(abs_err_sum[..., v].sum() / n_elements_per_var) for v in range(d)}
    mse_per_var  = {var_names[v]: float(sq_err_sum[..., v].sum()  / n_elements_per_var) for v in range(d)}
    rmse_per_var = {v: float(np.sqrt(mse_per_var[v])) for v in mse_per_var}

    # NRMSE per horizon (normalised units): sqrt(mean_over_samples_and_N of sq_err_norm) per (H, var)
    nrmse_curve = np.sqrt(sq_err_norm_sum_hd / count_hd)   # (H, d)
    nrmse_per_horizon = {var_names[v]: nrmse_curve[:, v].tolist() for v in range(d)}

    # MAE and RMSE per horizon (physical units): mean over samples and stations at each step
    n_per_step = n_samples * N
    mae_per_horizon  = {var_names[v]: (abs_err_sum[:, :, v].sum(axis=1) / n_per_step).tolist() for v in range(d)}
    rmse_per_horizon = {var_names[v]: np.sqrt(sq_err_sum[:, :, v].sum(axis=1) / n_per_step).tolist() for v in range(d)}

    # VPT
    def vpt_from_curve(curve):
        above = np.where(np.asarray(curve) >= VPT_EPSILON)[0]
        return int(above[0]) if above.size > 0 else len(curve)
    vpt = {var_names[v]: vpt_from_curve(nrmse_curve[:, v]) for v in range(d)}

    # Stratified per-variable (station-averaged) MAE
    mae_extreme_per_var = {}
    mae_nonextreme_per_var = {}
    for v in range(d):
        # For each variable v, per-station mae_extreme = ext_err_sum[:,v] / ext_count[:,v]; NaN where count==0.
        with np.errstate(divide="ignore", invalid="ignore"):
            mae_ext_by_station = np.where(ext_count[:, v] > 0, ext_err_sum[:, v] / ext_count[:, v], np.nan)
            mae_non_by_station = np.where(nonext_count[:, v] > 0, nonext_err_sum[:, v] / nonext_count[:, v], np.nan)
        mae_extreme_per_var[var_names[v]]    = float(np.nanmean(mae_ext_by_station))
        mae_nonextreme_per_var[var_names[v]] = float(np.nanmean(mae_non_by_station))

    return {
        "units": "physical",
        "vpt_epsilon": VPT_EPSILON,
        "mae": mae_per_var,
        "rmse": rmse_per_var,
        "mse": mse_per_var,
        "mae_extreme": mae_extreme_per_var,
        "mae_nonextreme": mae_nonextreme_per_var,
        "mae_per_horizon": mae_per_horizon,
        "rmse_per_horizon": rmse_per_horizon,
        "nrmse_per_horizon": nrmse_per_horizon,
        "vpt": vpt,
        "n_samples": int(n_samples),
    }


# ---------------------------------------------------------------------------
# Classification — chunked over samples
# ---------------------------------------------------------------------------

def _cls_target(y_test_lab, cstart, cend):
    starts = np.arange(cstart, cend)
    return np.stack([np.asarray(y_test_lab[i + INPUT_LEN : i + INPUT_LEN + HORIZON], dtype=np.int8) for i in starts], axis=0)


def _persistence_cls_pred(y_test_lab, cstart, cend):
    last_obs = y_test_lab[INPUT_LEN - 1 + cstart : INPUT_LEN - 1 + cend]
    last_obs = np.asarray(last_obs, dtype=np.float32)
    return np.broadcast_to(last_obs[:, None, :, :], (last_obs.shape[0], HORIZON, *last_obs.shape[1:])).copy()


def _climatology_cls_pred(train_rate, chunk_size):
    return np.broadcast_to(
        train_rate[None, None, :, :],
        (chunk_size, HORIZON, *train_rate.shape),
    ).copy()


def classification_baseline_metrics(pred_fn, name, y_test, var_names, extra=None):
    print(f"\n=== Classification {name} ===", flush=True)
    label_idx = [var_names.index(v) for v in LABEL_VARS]
    y_test_lab_all = np.asarray(y_test)[..., label_idx].astype(np.int8)
    T, N, d_lab = y_test_lab_all.shape
    n_samples = T - INPUT_LEN - HORIZON + 1

    # Accumulators — global (pooled over samples, H, N) per variable
    # For F1 / precision / recall / CSI we need TP, FP, FN, TN
    # Use tau=0.5 → predict positive iff prob >= 0.5
    tau = 0.5
    tp = np.zeros(d_lab, dtype=np.int64)
    fp = np.zeros(d_lab, dtype=np.int64)
    fn = np.zeros(d_lab, dtype=np.int64)
    tn = np.zeros(d_lab, dtype=np.int64)
    # Per-station arrays
    tp_ns = np.zeros((N, d_lab), dtype=np.int64)
    fp_ns = np.zeros((N, d_lab), dtype=np.int64)
    fn_ns = np.zeros((N, d_lab), dtype=np.int64)
    tn_ns = np.zeros((N, d_lab), dtype=np.int64)
    # For AUC and PR-AUC — need all preds and targets ranked together per var (memory-heavy).
    # We approximate with a coarse-histogram approach: bin prob into 1000 bins, count positives/negatives per bin per var.
    N_BINS = 1000
    pos_hist = np.zeros((d_lab, N_BINS + 2), dtype=np.int64)   # +2 for < 0 and > 1
    neg_hist = np.zeros((d_lab, N_BINS + 2), dtype=np.int64)

    for cstart in range(0, n_samples, CHUNK):
        cend = min(cstart + CHUNK, n_samples)
        preds = pred_fn(cstart, cend)                                              # (c, H, N, d_lab) probabilities
        target = _cls_target(y_test_lab_all, cstart, cend)                          # (c, H, N, d_lab) int8

        pred_bin = (preds >= tau).astype(np.int8)
        # Vectorised confusion
        for v in range(d_lab):
            p = pred_bin[..., v]
            t = target[..., v]
            tp[v] += int(((p == 1) & (t == 1)).sum())
            fp[v] += int(((p == 1) & (t == 0)).sum())
            fn[v] += int(((p == 0) & (t == 1)).sum())
            tn[v] += int(((p == 0) & (t == 0)).sum())
            # Per-station: reduce over sample and horizon axes
            tp_ns[:, v] += ((p == 1) & (t == 1)).sum(axis=(0, 1))
            fp_ns[:, v] += ((p == 1) & (t == 0)).sum(axis=(0, 1))
            fn_ns[:, v] += ((p == 0) & (t == 1)).sum(axis=(0, 1))
            tn_ns[:, v] += ((p == 0) & (t == 0)).sum(axis=(0, 1))
            # Score histogram — clip prob into [0, 1], bin, accumulate
            probs = preds[..., v].ravel()
            targets_flat = t.ravel().astype(np.int64)
            bins = np.clip((probs * N_BINS).astype(np.int64), 0, N_BINS - 1) + 1
            for pos, count in zip(*np.unique(bins[targets_flat == 1], return_counts=True)):
                pos_hist[v, pos] += int(count)
            for pos, count in zip(*np.unique(bins[targets_flat == 0], return_counts=True)):
                neg_hist[v, pos] += int(count)

        if cstart % (CHUNK * 4) == 0 or cend == n_samples:
            pct = 100.0 * cend / n_samples
            print(f"  chunk {cstart:>6d}–{cend:>6d}  ({pct:5.1f}%)", flush=True)

    per_variable = {}
    per_station = {m: np.zeros((N, d_lab), dtype=np.float32) for m in ("f1", "precision", "recall", "csi", "accuracy")}
    for v_idx, v in enumerate(LABEL_VARS):
        TP, FP, FN, TN = tp[v_idx], fp[v_idx], fn[v_idx], tn[v_idx]
        precision = TP / (TP + FP) if TP + FP > 0 else 0.0
        recall    = TP / (TP + FN) if TP + FN > 0 else 0.0
        f1        = 2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN > 0 else 0.0
        csi       = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0.0
        acc       = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN > 0 else 0.0

        # Compute AUC-ROC and PR-AUC from the histogram
        # ROC: sweep threshold from bin_max downward, accumulate TPR and FPR
        pos_cum = np.cumsum(pos_hist[v_idx][::-1])[::-1]  # cumulative positives with prob >= bin
        neg_cum = np.cumsum(neg_hist[v_idx][::-1])[::-1]
        total_pos = int(pos_hist[v_idx].sum())
        total_neg = int(neg_hist[v_idx].sum())
        auc_roc = 0.0
        pr_auc  = 0.0
        if total_pos > 0 and total_neg > 0:
            # Trapezoidal AUC-ROC over TPR vs FPR at each bin threshold
            tpr = pos_cum / total_pos
            fpr = neg_cum / total_neg
            # Sort by FPR ascending (which is reverse of thresholds since higher bin = lower recall)
            order = np.argsort(fpr)
            fpr_sorted = fpr[order]
            tpr_sorted = tpr[order]
            auc_roc = float(np.trapz(tpr_sorted, fpr_sorted))
            # PR-AUC: precision vs recall
            with np.errstate(divide="ignore", invalid="ignore"):
                prec_at_bin = np.where(pos_cum + neg_cum > 0, pos_cum / (pos_cum + neg_cum), 1.0)
            recall_at_bin = tpr
            # Sort by recall ascending
            order2 = np.argsort(recall_at_bin)
            pr_auc = float(np.trapz(prec_at_bin[order2], recall_at_bin[order2]))

        per_variable[v] = {
            "tau": tau,
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "csi": float(csi),
            "auc_roc": float(auc_roc),
            "pr_auc": float(pr_auc),
            "tp": int(TP), "fp": int(FP), "fn": int(FN), "tn": int(TN),
        }
        for n in range(N):
            TPn, FPn, FNn, TNn = tp_ns[n, v_idx], fp_ns[n, v_idx], fn_ns[n, v_idx], tn_ns[n, v_idx]
            per_station["precision"][n, v_idx] = TPn / (TPn + FPn) if TPn + FPn > 0 else np.nan
            per_station["recall"][n, v_idx]    = TPn / (TPn + FNn) if TPn + FNn > 0 else np.nan
            per_station["f1"][n, v_idx]        = 2 * TPn / (2 * TPn + FPn + FNn) if 2 * TPn + FPn + FNn > 0 else np.nan
            per_station["csi"][n, v_idx]       = TPn / (TPn + FPn + FNn) if TPn + FPn + FNn > 0 else np.nan
            per_station["accuracy"][n, v_idx]  = (TPn + TNn) / (TPn + FPn + FNn + TNn) if TPn + FPn + FNn + TNn > 0 else np.nan

    # Macro means over variables
    macro = {m: float(np.mean([per_variable[v][m] for v in LABEL_VARS]))
             for m in ("precision", "recall", "f1", "csi", "auc_roc", "pr_auc", "accuracy")}

    return {
        "per_variable": per_variable,
        "macro": macro,
        "taus": [tau] * len(LABEL_VARS),
        "label_variables": LABEL_VARS,
        "n_samples": int(n_samples),
    }


def save_json(path, obj):
    def convert(x):
        if isinstance(x, dict):  return {k: convert(v) for k, v in x.items()}
        if isinstance(x, list):  return [convert(v) for v in x]
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (np.floating, np.integer)): return x.item()
        return x
    with open(path, "w") as f:
        json.dump(convert(obj), f, indent=2)
    print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_DEFAULTS.keys()), default="era5",
                    help="which cached dataset to compute baselines for")
    ap.add_argument("--out", default=None,
                    help="output directory (default: results/baselines/)")
    ap.add_argument("--skip-classification", action="store_true",
                    help="regression baselines only (skip classification)")
    args = ap.parse_args()

    ds = DATASET_DEFAULTS[args.dataset]
    CACHE = Path(__file__).parent.parent / "preprocessing" / "cache" / ds["cache_subdir"]
    OUT = Path(args.out) if args.out else (Path(__file__).parent.parent / "results" / "baselines")
    OUT.mkdir(parents=True, exist_ok=True)
    HORIZON = ds["horizon"]
    LABEL_VARS = ds["label_vars"]
    print(f"Dataset: {args.dataset}  cache={CACHE}  out={OUT}  horizon={HORIZON}  label_vars={LABEL_VARS}")

    X_train, X_test, y_train, y_test, mean, std, thresholds, var_names = load_arrays()

    # Precompute climatology inputs
    print("\nComputing training-set climatology stats (streaming)...", flush=True)
    train_mean_sum = np.zeros(X_train.shape[1:], dtype=np.float64)
    n_rows = 0
    for start in range(0, X_train.shape[0], 20000):
        block = np.asarray(X_train[start : start + 20000], dtype=np.float64)
        train_mean_sum += block.sum(axis=0)
        n_rows += block.shape[0]
    train_mean_norm = (train_mean_sum / n_rows).astype(np.float32)
    print(f"  train_mean_norm |mean|={np.abs(train_mean_norm).mean():.4f} (~0 by z-score construction)")

    reg_persist  = regression_baseline_metrics(
        lambda cs, ce: _persistence_pred(X_test, cs, ce), "persistence",
        X_test, mean, std, thresholds, var_names,
    )
    save_json(OUT / "regression-persistence.json", reg_persist)

    reg_climate  = regression_baseline_metrics(
        lambda cs, ce: _climatology_pred(train_mean_norm, ce - cs), "climatology",
        X_test, mean, std, thresholds, var_names,
    )
    save_json(OUT / "regression-climatology.json", reg_climate)

    if args.skip_classification:
        print("\n--skip-classification set — regression baselines only.")
    else:
        label_idx = [var_names.index(v) for v in LABEL_VARS]
        print("Computing training-set positive rate per (station, variable)...", flush=True)
        train_rate_sum = np.zeros((X_train.shape[1], len(LABEL_VARS)), dtype=np.float64)
        n_rows_y = 0
        for start in range(0, y_train.shape[0], 20000):
            block = np.asarray(y_train[start : start + 20000][..., label_idx], dtype=np.float64)
            train_rate_sum += block.sum(axis=0)
            n_rows_y += block.shape[0]
        train_rate = (train_rate_sum / n_rows_y).astype(np.float32)
        print(f"  mean positive rate per var: {train_rate.mean(axis=0)}")

        y_test_lab = np.asarray(y_test)[..., label_idx].astype(np.int8)
        cls_persist  = classification_baseline_metrics(
            lambda cs, ce: _persistence_cls_pred(y_test_lab, cs, ce), "persistence",
            y_test, var_names,
        )
        save_json(OUT / "classification-persistence.json", cls_persist)

        cls_climate  = classification_baseline_metrics(
            lambda cs, ce: _climatology_cls_pred(train_rate, ce - cs), "climatology",
            y_test, var_names,
        )
        save_json(OUT / "classification-climatology.json", cls_climate)

    print("\nAll baselines computed.")
