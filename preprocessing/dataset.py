"""
PyTorch Dataset and DataLoader factory for sliding-window spatio-temporal data.

Wraps the preprocessed split tensors in a Dataset subclass that yields
(x, y) pairs via a sliding window:
  - x: historical window of T timesteps, shape (T, N, d)
  - y: target of the next H timesteps — shape (H, N, d) for regression
       or (H, N) for classification (binary extreme event labels)

DataLoaders are constructed with shuffle=True for train and shuffle=False for
val/test, with a configurable batch size and number of workers.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class STGNNDataset(Dataset):
    """Sliding-window dataset for spatio-temporal graph neural networks."""

    def __init__(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray],
        input_len: int,
        horizon: int,
        task: str,
    ):
        """
        Args:
            features: normalised feature array, shape (T, N, d).
            labels: binary label array, shape (T, N, d) or None for regression.
            input_len: number of historical timesteps T fed as input.
            horizon: number of future timesteps H to predict.
            task: 'regression' or 'classification'.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def build_dataloaders(
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: Optional[np.ndarray],
    val_labels: Optional[np.ndarray],
    test_labels: Optional[np.ndarray],
    input_len: int,
    horizon: int,
    task: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    raise NotImplementedError
