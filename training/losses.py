"""
Loss functions for regression and classification tasks.

MSE  — Phase 1 regression. Standard mean squared error.

WBCE — Phase 2 classification (per-variable multi-label formulation).

       For per-variable labels of shape (B, H, N, d_label), one weight per
       label channel: ``pos_weight ∈ ℝ^{d_label}``, with
       ``pos_weight[v] = N_neg,v / N_pos,v`` computed on the TRAIN split only.
       At the 95th-percentile threshold this sits near ~19 for most variables
       (tp differs under the wet-hour rule). Weights are computed once at
       experiment start, logged to MLflow, and passed to the loss for all 10
       seed runs. BCEWithLogitsLoss broadcasts ``pos_weight`` across the
       leading axes and matches on the trailing (variable) axis.

       For the legacy ANY-collapsed labels of shape (B, H, N), pos_weight
       degenerates to a scalar (still passed as a length-1 tensor to
       BCEWithLogitsLoss). This path is retained for backwards compatibility
       only; Phase 2 uses per-variable.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def mse_loss() -> nn.Module:
    """Return the MSE loss module used for Phase 1 regression."""
    return nn.MSELoss()


def compute_class_weights(
    train_labels: np.ndarray,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Compute negative/positive class weights from the training label distribution.

    Args:
        train_labels: binary label array.
            - Per-variable formulation: (..., d_label). The positive rate is
              computed per channel (mean over all leading axes), returning
              two arrays of shape (d_label,).
            - ANY formulation: any shape without a variable axis, or a
              1-D array. Returns two scalars.

    Returns:
        (w_0, w_1) where ``w_1 = (1 - pos_rate) / pos_rate`` — the ratio
        BCEWithLogitsLoss consumes as ``pos_weight`` (with w_0 held at 1).

        Both are either Python floats (ANY / scalar case) or numpy arrays
        of shape (d_label,) (per-variable case).
    """
    labels = np.asarray(train_labels)
    # np.mean promotes int input to float64 internally; skipping an explicit
    # .astype(float) here avoids a full-array copy (~9 GB at N=999).

    if labels.ndim <= 1:
        # ANY / already-flat case: single scalar rate.
        pos_rate = float(labels.mean())
        pos_rate = min(max(pos_rate, 1e-6), 1 - 1e-6)
        return 1.0, (1.0 - pos_rate) / pos_rate

    # Per-variable: last axis is the label variable axis.
    d_label = labels.shape[-1]
    flat = labels.reshape(-1, d_label)                       # view, no copy
    pos_rate = flat.mean(axis=0, dtype=np.float64)           # (d_label,)
    pos_rate = np.clip(pos_rate, 1e-6, 1 - 1e-6)
    w_1 = (1.0 - pos_rate) / pos_rate
    w_0 = np.ones_like(w_1)
    return w_0, w_1


def wbce_loss(
    w_0: Union[float, np.ndarray],
    w_1: Union[float, np.ndarray],
    device: torch.device,
) -> nn.Module:
    """
    Return a Weighted Binary Cross-Entropy loss module.

    ``BCEWithLogitsLoss`` uses ``pos_weight = w_1 / w_0`` to up-weight positive
    samples. Accepts either scalars (ANY collapse) or arrays of shape
    (d_label,) (per-variable). The tensor shape is ``(d_label,)`` in the
    latter case, which broadcasts across the leading (B, H, N) axes of the
    prediction tensor.
    """
    w_1_arr = np.asarray(w_1, dtype=np.float32)
    w_0_arr = np.asarray(w_0, dtype=np.float32)
    ratio = (w_1_arr / w_0_arr).reshape(-1)
    pos_weight = torch.from_numpy(ratio.astype(np.float32)).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
