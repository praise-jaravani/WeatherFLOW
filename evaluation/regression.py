"""
Regression evaluation metrics for Phase 1.

All metrics are computed per station then aggregated (mean ± std) across stations.
Per-station results are also returned to support geographic stratification in Phase 3.

Input convention for most functions: preds/targets of shape (T, N, d), where
T = samples * H flattened, N = nodes, d = variables. nrmse_per_horizon is the
exception — it requires the (samples, H, N, d) shape to compute per-step NRMSE.

Metrics:
  MAE              — mean absolute error (primary metric)
  RMSE             — root mean squared error
  MSE              — mean squared error (training loss reference)
  MAPE             — mean absolute percentage error (scale-independent)
  Stratified MAE   — MAE split by extreme vs. non-extreme timesteps
  Peak Intensity Error — MAE on the predicted maximum within each extreme window
  NRMSE(t_f)       — RMSE at each forecast step t_f normalised by ground-truth std
  VPT              — max { t_f : NRMSE(t_f) < ε }; measures temporal robustness
"""

import numpy as np
from typing import Dict, Optional, Tuple


def mae(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """MAE per station averaged over time and variables. Shape: (N,)."""
    return np.abs(preds - targets).mean(axis=(0, 2))


def rmse(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """RMSE per station averaged over time and variables. Shape: (N,)."""
    return np.sqrt(((preds - targets) ** 2).mean(axis=(0, 2)))


def mse(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """MSE per station averaged over time and variables. Shape: (N,)."""
    return ((preds - targets) ** 2).mean(axis=(0, 2))


def mape(preds: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """MAPE per station. Shape: (N,). eps guards against division by zero."""
    return (np.abs(preds - targets) / (np.abs(targets) + eps)).mean(axis=(0, 2))


def stratified_mae(
    preds: np.ndarray, targets: np.ndarray, extreme_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MAE computed separately for extreme and non-extreme timesteps per station.

    Args:
        preds, targets: shape (T, N, d).
        extreme_mask: binary array shape (T, N, d), 1 where timestep is extreme.

    Returns:
        (mae_extreme, mae_nonextreme), each shape (N,).
    """
    T, N, d = preds.shape
    err = np.abs(preds - targets)

    mask = extreme_mask.astype(bool)
    inv_mask = ~mask

    mae_ext = np.zeros(N)
    mae_non = np.zeros(N)

    for n in range(N):
        ext_vals = err[:, n, :][mask[:, n, :]]
        non_vals = err[:, n, :][inv_mask[:, n, :]]
        mae_ext[n] = ext_vals.mean() if ext_vals.size > 0 else 0.0
        mae_non[n] = non_vals.mean() if non_vals.size > 0 else 0.0

    return mae_ext, mae_non


def peak_intensity_error(
    preds: np.ndarray, targets: np.ndarray, extreme_mask: np.ndarray
) -> np.ndarray:
    """
    MAE on the predicted maximum within each contiguous extreme event window.

    For each station and variable, finds contiguous runs of extreme timesteps,
    then computes |max(pred in window) - max(target in window)|.

    Shape: (N,).
    """
    T, N, d = preds.shape
    result = np.zeros(N)

    for n in range(N):
        errors = []
        for v in range(d):
            mask_nv = extreme_mask[:, n, v].astype(bool)
            if not mask_nv.any():
                continue

            # Find contiguous windows via diff
            padded = np.concatenate([[False], mask_nv, [False]])
            starts = np.where(~padded[:-1] & padded[1:])[0]
            ends = np.where(padded[:-1] & ~padded[1:])[0]

            for ws, we in zip(starts, ends):
                pred_peak = preds[ws:we, n, v].max()
                tgt_peak = targets[ws:we, n, v].max()
                errors.append(abs(pred_peak - tgt_peak))

        result[n] = float(np.mean(errors)) if errors else 0.0

    return result


def nrmse_per_horizon(
    preds: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """
    NRMSE at each forecast step t_f, normalised by std of ground truth.

    Args:
        preds, targets: shape (samples, H, N, d).

    Returns:
        NRMSE curve of shape (H,).
    """
    _, H, _, _ = preds.shape
    nrmse_curve = np.zeros(H)

    for h in range(H):
        pred_h = preds[:, h, :, :]    # (samples, N, d)
        tgt_h = targets[:, h, :, :]
        rmse_h = float(np.sqrt(((pred_h - tgt_h) ** 2).mean()))
        std_h = float(tgt_h.std()) + 1e-8
        nrmse_curve[h] = rmse_h / std_h

    return nrmse_curve


def valid_prediction_time(nrmse_curve: np.ndarray, epsilon: float) -> int:
    """
    VPT = max { t_f : NRMSE(t_f) < epsilon }.

    Returns the last (0-indexed) forecast step where NRMSE is still below
    epsilon, or 0 if NRMSE exceeds epsilon at step 0.
    """
    below = np.where(nrmse_curve < epsilon)[0]
    return int(below[-1]) if below.size > 0 else 0


def compute_all(
    preds: np.ndarray,
    targets: np.ndarray,
    extreme_mask: Optional[np.ndarray],
    vpt_epsilon: float,
) -> Dict[str, float]:
    """
    Compute all regression metrics and return as a flat dict (mean across stations).

    Args:
        preds, targets: shape (samples, H, N, d).
        extreme_mask:   shape (T, N, d) where T = samples * H, or None.
        vpt_epsilon:    NRMSE threshold for VPT.
    """
    samples, H, N, d = preds.shape
    T = samples * H

    preds_flat = preds.reshape(T, N, d)
    targets_flat = targets.reshape(T, N, d)

    mae_vals = mae(preds_flat, targets_flat)
    rmse_vals = rmse(preds_flat, targets_flat)
    mse_vals = mse(preds_flat, targets_flat)
    mape_vals = mape(preds_flat, targets_flat)

    nrmse_curve = nrmse_per_horizon(preds, targets)
    vpt = valid_prediction_time(nrmse_curve, vpt_epsilon)

    if extreme_mask is not None:
        mae_ext, mae_non = stratified_mae(preds_flat, targets_flat, extreme_mask)
        pie = peak_intensity_error(preds_flat, targets_flat, extreme_mask)
    else:
        mae_ext = mae_non = pie = np.zeros(N)

    return {
        "mae":                  float(mae_vals.mean()),
        "mae_std":              float(mae_vals.std()),
        "rmse":                 float(rmse_vals.mean()),
        "rmse_std":             float(rmse_vals.std()),
        "mse":                  float(mse_vals.mean()),
        "mape":                 float(mape_vals.mean()),
        "mae_extreme":          float(mae_ext.mean()),
        "mae_nonextreme":       float(mae_non.mean()),
        "peak_intensity_error": float(pie.mean()),
        "nrmse_per_horizon":    nrmse_curve.tolist(),
        "vpt":                  int(vpt),
    }
