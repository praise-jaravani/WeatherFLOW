"""
Z-score normalisation per variable per station.

Mean and standard deviation are computed from the TRAINING partition only and
applied to validation and test sets. This prevents future distributional
information from leaking into the normalisation statistics.

Normalisation statistics (mean, std) are saved to the .npz cache alongside the
data tensors so that inverse-normalisation can be applied to model outputs at
evaluation time (converting predictions back to original units for reporting).

Must be called after split.py.
"""

import numpy as np
from typing import Tuple


def compute_stats(train_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-variable, per-station mean and std from training data.

    Args:
        train_data: array of shape (T_train, N, d).

    Returns:
        mean of shape (N, d), std of shape (N, d).
    """
    raise NotImplementedError


def normalise(
    data: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply z-score normalisation: (data - mean) / std."""
    raise NotImplementedError


def denormalise(
    data: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Invert z-score normalisation: data * std + mean."""
    raise NotImplementedError


def normalise_all_splits(
    train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: compute stats on train, normalise all three splits.

    Returns:
        (train_norm, val_norm, test_norm, mean, std)
    """
    raise NotImplementedError
