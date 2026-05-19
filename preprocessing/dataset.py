"""
PyTorch Dataset and DataLoader factory for sliding-window spatio-temporal data.

Yields (x, y) pairs via a sliding window:
  x: historical window of shape (input_len, N, d)
  y: regression target  shape (horizon, N, d)
     OR classification target shape (horizon, N) — majority vote over vars
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple


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
            features:  (T, N, d) float32 normalised array.
            labels:    (T, N, d) int8 binary labels, or None for regression.
            input_len: historical context length.
            horizon:   prediction horizon.
            task:      'regression' or 'classification'.
        """
        assert task in ("regression", "classification")
        self.features  = torch.from_numpy(features.astype(np.float32))
        self.labels    = (torch.from_numpy(labels.astype(np.float32))
                          if labels is not None else None)
        self.input_len = input_len
        self.horizon   = horizon
        self.task      = task
        self.n_samples = len(features) - input_len - horizon + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_start = idx
        x_end   = idx + self.input_len
        y_start = x_end
        y_end   = y_start + self.horizon

        x = self.features[x_start:x_end]  # (input_len, N, d)

        if self.task == "regression":
            y = self.features[y_start:y_end]  # (horizon, N, d)
        else:
            # For classification use the label array; collapse vars by max (any extreme)
            y = self.labels[y_start:y_end].max(dim=-1).values  # (horizon, N)

        return x, y


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
    def _make(feats, labs, shuffle):
        ds = STGNNDataset(feats, labs, input_len, horizon, task)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=shuffle,  # drop incomplete last batch only during training
        )

    return (
        _make(train_features, train_labels, shuffle=True),
        _make(val_features,   val_labels,   shuffle=False),
        _make(test_features,  test_labels,  shuffle=False),
    )
