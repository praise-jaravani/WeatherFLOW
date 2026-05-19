"""
Binary extreme event labelling for Phase 2 classification.

95th-percentile thresholds are computed from the TRAINING partition only and
applied across all three splits. Must be called after split.py.

Thresholds are saved alongside data tensors in the .npz cache so they can be
inspected and reused without re-running the pipeline.
"""

import numpy as np
from typing import Tuple


def compute_thresholds(
    train_data: np.ndarray,
    percentile: float = 95.0,
) -> np.ndarray:
    """
    Compute per-variable, per-node extreme-event threshold from training data.

    Args:
        train_data: (T_train, N, d) array.
        percentile: percentile defining "extreme" (default 95).

    Returns:
        thresholds: (N, d) array — one value per node per variable.
    """
    return np.percentile(train_data, percentile, axis=0).astype(np.float32)


def apply_labels(
    data: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Produce binary extreme-event labels: 1 where data > threshold, else 0.

    Args:
        data:       (T, N, d) array.
        thresholds: (N, d) array computed from training data.

    Returns:
        labels: (T, N, d) int8 array.
    """
    return (data > thresholds[None, :, :]).astype(np.int8)


def label_all_splits(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    percentile: float = 95.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute thresholds on train, apply to all three splits.

    Returns:
        (train_labels, val_labels, test_labels, thresholds)
    """
    thresholds = compute_thresholds(train, percentile)
    return (
        apply_labels(train, thresholds),
        apply_labels(val,   thresholds),
        apply_labels(test,  thresholds),
        thresholds,
    )
