"""
Regression evaluation metrics for Phase 1.

All metrics are computed per station then aggregated (mean ± std) across stations.
Per-station results are also returned to support geographic stratification in Phase 3.

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
from typing import Dict, Tuple


def mae(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """MAE per station. Shape: (N,)."""
    raise NotImplementedError


def rmse(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """RMSE per station. Shape: (N,)."""
    raise NotImplementedError


def mse(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """MSE per station. Shape: (N,)."""
    raise NotImplementedError


def mape(preds: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """MAPE per station. Shape: (N,). eps guards against division by zero."""
    raise NotImplementedError


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
    raise NotImplementedError


def peak_intensity_error(
    preds: np.ndarray, targets: np.ndarray, extreme_mask: np.ndarray
) -> np.ndarray:
    """
    MAE on the predicted maximum within each contiguous extreme event window.

    Shape: (N,).
    """
    raise NotImplementedError


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
    raise NotImplementedError


def valid_prediction_time(nrmse_curve: np.ndarray, epsilon: float) -> int:
    """
    VPT = max { t_f : NRMSE(t_f) < epsilon }.

    Returns the last forecast step index before NRMSE exceeds epsilon,
    or 0 if NRMSE exceeds epsilon immediately.
    """
    raise NotImplementedError


def compute_all(
    preds: np.ndarray,
    targets: np.ndarray,
    extreme_mask: np.ndarray,
    vpt_epsilon: float,
) -> Dict[str, float]:
    """Compute all regression metrics and return as a flat dict (mean across stations)."""
    raise NotImplementedError
