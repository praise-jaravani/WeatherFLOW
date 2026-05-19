"""
Z-score normalisation per variable per node.

Mean and std are computed from the TRAINING partition only and applied to
validation and test sets. Must be called after split.py.

Stats are saved to .npz cache alongside data tensors so that
denormalise() can convert model outputs back to original units at eval time.
"""

import numpy as np
from typing import Tuple


def compute_stats(train_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-variable, per-node mean and std from training data.

    Args:
        train_data: (T_train, N, d) array.

    Returns:
        mean: (1, N, d) — broadcast-ready.
        std:  (1, N, d) — zero-std entries are replaced with 1.0 to avoid division by zero.
    """
    mean = train_data.mean(axis=0, keepdims=True)
    std  = train_data.std(axis=0, keepdims=True)
    std  = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalise(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply z-score normalisation: (data - mean) / std."""
    return ((data - mean) / std).astype(np.float32)


def denormalise(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Invert z-score normalisation: data * std + mean."""
    return (data * std + mean).astype(np.float32)


def normalise_all_splits(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute stats on train, normalise all three splits.

    Returns:
        (train_norm, val_norm, test_norm, mean, std)
    """
    mean, std = compute_stats(train)
    return normalise(train, mean, std), normalise(val, mean, std), normalise(test, mean, std), mean, std
