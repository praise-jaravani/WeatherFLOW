"""
Chronological train/validation/test split.

Splits the time axis of the preprocessed tensor strictly by time — no shuffling.
Default ratio: 60% train / 20% validation / 20% test.

The split is performed BEFORE labelling and normalisation so that all downstream
statistics (95th-percentile thresholds, z-score mean/std) are computed from the
training partition only, preventing data leakage.

Returns the three index boundaries (train_end, val_end) that partition the
time dimension, plus the three sub-arrays for each split.
"""

import numpy as np
from typing import Tuple


def chronological_split(
    data: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data along the time axis (axis=0) into train, val, test.

    Args:
        data: array of shape (T, N, d) — time × stations × features.
        train_ratio: fraction of timesteps for training.
        val_ratio: fraction of timesteps for validation.

    Returns:
        Tuple of (train, val, test) arrays.
    """
    raise NotImplementedError
