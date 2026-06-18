"""
Regression evaluation metrics for Phase 1.

Design (post-2026-06-18 audit):
  * Error metrics (MAE / RMSE / MSE / sMAPE, stratified, peak) are computed
    PER VARIABLE and, when z-score stats are supplied, in PHYSICAL UNITS.
    Pooling across variables is meaningless once denormalised, so every error
    metric is keyed by variable name.
  * NRMSE / VPT are computed on the NORMALISED arrays, where each variable has
    ~unit variance, so the epsilon threshold reads as a fraction of a standard
    deviation and is comparable across variables.
  * Per-station (N, d) arrays are retained under 'per_station' for Phase 3.
    NOTE: these are ndarrays — strip or convert before JSON/MLflow serialisation.

Array conventions:
  preds / targets : (samples, H, N, d)   NORMALISED, as the model emits.
  mean / std      : (1, N, d)            train-only z-score stats.
  extreme_mask    : (samples, H, N, d) or (samples*H, N, d), bool/0-1, or None.
"""

import warnings

import numpy as np
from typing import Dict, List, Optional, Tuple

ERA5_VAR_NAMES = ["t2m", "d2m", "tp", "u10", "v10", "sp", "ws10", "rh"]
REPORT_VARS = ["t2m", "rh", "ws10", "sp"]   # the four Gaibie (2024) targets


def _denormalise(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Invert z-score. arr: (samples, H, N, d); mean/std broadcast from (1, N, d)."""
    m = np.asarray(mean).reshape(1, 1, arr.shape[2], arr.shape[3])
    s = np.asarray(std).reshape(1, 1, arr.shape[2], arr.shape[3])
    return arr * s + m


def _as_4d_mask(mask: np.ndarray, shape: Tuple[int, int, int, int]) -> np.ndarray:
    """Accept (samples, H, N, d) or (samples*H, N, d); return (samples, H, N, d) bool."""
    samples, H, N, d = shape
    mask = np.asarray(mask)
    if mask.ndim == 4:
        return mask.astype(bool)
    return mask.reshape(samples, H, N, d).astype(bool)


# --- per-variable point-error metrics (operate on physical-unit arrays) ----- #

def per_station_var_mae(preds, targets):    # (samples,H,N,d) -> (N,d)
    return np.abs(preds - targets).mean(axis=(0, 1))

def per_station_var_rmse(preds, targets):   # -> (N,d)
    return np.sqrt(((preds - targets) ** 2).mean(axis=(0, 1)))

def per_station_var_mse(preds, targets):    # -> (N,d)
    return ((preds - targets) ** 2).mean(axis=(0, 1))

def per_station_var_smape(preds, targets, eps=1e-3):   # -> (N,d), range [0,2]
    """Symmetric MAPE; bounded even when targets ~ 0 (unlike MAPE)."""
    denom = (np.abs(preds) + np.abs(targets)) / 2.0 + eps
    return (np.abs(preds - targets) / denom).mean(axis=(0, 1))


def stratified_var_mae(preds, targets, mask4d):
    """Per-station, per-variable MAE split extreme/non-extreme. (N,d) each, NaN if empty."""
    err = np.abs(preds - targets)
    ext_err = np.where(mask4d, err, np.nan)
    non_err = np.where(mask4d, np.nan, err)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        return np.nanmean(ext_err, axis=(0, 1)), np.nanmean(non_err, axis=(0, 1))


def peak_intensity_error(preds, targets, mask4d):
    """
    Per-station, per-variable peak error WITHIN each forecast window.
    For each window containing >=1 extreme timestep for (n,v), compare
    max(pred over H) with max(target over H). Averaged over windows. (N,d), NaN if none.

    Deliberately does NOT scan the flattened sample axis — those 'runs' are
    sliding-window artefacts. The H axis of one sample is a genuine contiguous horizon.
    """
    win_has_ext = mask4d.any(axis=1)            # (samples,N,d)
    pred_peak = preds.max(axis=1)
    tgt_peak = targets.max(axis=1)
    peak_err = np.where(win_has_ext, np.abs(pred_peak - tgt_peak), np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        return np.nanmean(peak_err, axis=0)     # (N,d)


# --- temporal robustness (operate on NORMALISED arrays) --------------------- #

def nrmse_per_horizon(preds_norm, targets_norm):   # (samples,H,N,d) -> (H,d)
    se = (preds_norm - targets_norm) ** 2
    rmse_hv = np.sqrt(se.mean(axis=(0, 2)))
    std_hv = targets_norm.std(axis=(0, 2)) + 1e-8
    return rmse_hv / std_hv


def valid_prediction_time(nrmse_curve, epsilon):
    """
    First-crossing VPT for one variable: number of leading horizon steps (0..H)
    that stay below epsilon. Once the curve crosses epsilon the forecast is
    invalid thereafter, even if it dips back (which 'last step below' wrongly rewards).
    nrmse_curve: (H,).
    """
    above = np.where(nrmse_curve >= epsilon)[0]
    return int(above[0]) if above.size > 0 else len(nrmse_curve)


def compute_all(
    preds: np.ndarray,
    targets: np.ndarray,
    extreme_mask: Optional[np.ndarray] = None,
    vpt_epsilon: float = 0.3,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    var_names: Optional[List[str]] = None,
    report_vars: Optional[List[str]] = None,
) -> Dict:
    """
    preds, targets : (samples, H, N, d), NORMALISED.
    extreme_mask   : (samples,H,N,d) or (samples*H,N,d) or None.
    vpt_epsilon    : NRMSE threshold (fraction of a std). Default 0.3.
    mean, std      : (1,N,d) train stats. If given, error metrics are PHYSICAL.
    """
    samples, H, N, d = preds.shape
    if var_names is None:
        var_names = ERA5_VAR_NAMES if d == len(ERA5_VAR_NAMES) else [f"var{i}" for i in range(d)]
    if report_vars is None:
        report_vars = [v for v in REPORT_VARS if v in var_names]

    nrmse_curve = nrmse_per_horizon(preds, targets)            # (H,d), normalised
    vpt = {var_names[v]: valid_prediction_time(nrmse_curve[:, v], vpt_epsilon)
           for v in range(d)}

    if mean is not None and std is not None:
        p, t, units = _denormalise(preds, mean, std), _denormalise(targets, mean, std), "physical"
    else:
        p, t, units = preds, targets, "normalised"

    mae_nd = per_station_var_mae(p, t)
    rmse_nd = per_station_var_rmse(p, t)
    mse_nd = per_station_var_mse(p, t)
    smape_nd = per_station_var_smape(p, t)

    if extreme_mask is not None:
        mask4d = _as_4d_mask(extreme_mask, (samples, H, N, d))
        mae_ext_nd, mae_non_nd = stratified_var_mae(p, t, mask4d)
        pie_nd = peak_intensity_error(p, t, mask4d)
    else:
        mae_ext_nd = mae_non_nd = pie_nd = np.full((N, d), np.nan)

    def by_var(a):
        with np.errstate(invalid="ignore"):
            return {var_names[v]: float(np.nanmean(a[:, v])) for v in range(d)}

    return {
        "units": units,
        "vpt_epsilon": vpt_epsilon,
        "report_vars": report_vars,
        "mae": by_var(mae_nd),
        "rmse": by_var(rmse_nd),
        "mse": by_var(mse_nd),
        "smape": by_var(smape_nd),
        "mae_extreme": by_var(mae_ext_nd),
        "mae_nonextreme": by_var(mae_non_nd),
        "peak_intensity_error": by_var(pie_nd),
        "nrmse_per_horizon": {var_names[v]: nrmse_curve[:, v].tolist() for v in range(d)},
        "vpt": vpt,
        "mae_pooled_norm": float(np.abs(preds - targets).mean()),   # z-space monitor only
        "per_station": {
            "mae": mae_nd, "rmse": rmse_nd,
            "mae_extreme": mae_ext_nd, "mae_nonextreme": mae_non_nd,
            "peak_intensity_error": pie_nd,
        },
    }
