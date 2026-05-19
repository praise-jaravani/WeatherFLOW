"""
Chronological train/validation/test split.

Splits the time axis strictly by time — no shuffling. Default ratio:
60% train / 20% validation / 20% test.

MUST be called before labels.py and normalise.py so that threshold and
z-score statistics are computed from the training partition only.
"""

import numpy as np
from typing import Tuple


def chronological_split(
    data: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data along axis 0 (time) into train, val, test.

    Args:
        data:        array of shape (T, ...) — time is axis 0.
        train_ratio: fraction of timesteps for training.
        val_ratio:   fraction of timesteps for validation.

    Returns:
        (train, val, test) arrays with the same dtype as input.
    """
    T = data.shape[0]
    train_end = int(T * train_ratio)
    val_end   = train_end + int(T * val_ratio)
    return data[:train_end], data[train_end:val_end], data[val_end:]


def split_indices(
    T: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[int, int]:
    """
    Return (train_end, val_end) index boundaries for a sequence of length T.
    Useful when the same boundaries must be applied to several arrays
    (e.g. X and timestamps) without duplicating the ratio arithmetic.
    """
    train_end = int(T * train_ratio)
    val_end   = train_end + int(T * val_ratio)
    return train_end, val_end
