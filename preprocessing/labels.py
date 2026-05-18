"""
Binary extreme event labelling.

Computes the 95th-percentile threshold for each variable at each station using
the TRAINING partition only, then applies it across all three splits to produce
binary labels (1 = extreme, 0 = non-extreme).

Must be called after split.py so training-only statistics can be computed
without leaking future distributional information into the threshold.

Thresholds are saved to the .npz cache so they can be inspected and reused
without re-running the pipeline.
"""

import numpy as np
from typing import Tuple, Dict


def compute_thresholds(
    train_data: np.ndarray, percentile: float = 95.0
) -> np.ndarray:
    """
    Compute per-variable, per-station extreme event threshold from training data.

    Args:
        train_data: array of shape (T_train, N, d).
        percentile: percentile defining 'extreme' (default 95).

    Returns:
        threshold array of shape (N, d) — one threshold per station per variable.
    """
    raise NotImplementedError


def apply_labels(
    data: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """
    Produce binary extreme event labels by comparing data against thresholds.

    Args:
        data: array of shape (T, N, d).
        thresholds: array of shape (N, d) computed from training data only.

    Returns:
        Binary label array of shape (T, N, d), dtype int8.
    """
    raise NotImplementedError


def label_all_splits(
    train: np.ndarray, val: np.ndarray, test: np.ndarray, percentile: float = 95.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: compute thresholds on train, apply to all three splits.

    Returns:
        (train_labels, val_labels, test_labels, thresholds)
    """
    raise NotImplementedError
