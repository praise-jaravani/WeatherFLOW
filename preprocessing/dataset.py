"""
PyTorch Dataset and DataLoader factory for sliding-window spatio-temporal data.

Yields (x, y) pairs via a sliding window:

  x: historical window of shape (input_len, N, d)
  y: regression target      — shape (horizon, N, d)
     OR classification target under `label_mode`:
        - "per_variable" (default): (horizon, N, d_label) — one binary label
          per selected label variable per station per horizon step. This is
          the Phase 2 formulation (see WeatherFLOW_Phase2_Plan.md §2.1–2.2).
        - "any": (horizon, N) — collapses label variables via ``max``
          ("any variable extreme"). Preserved for backwards compatibility
          only; not used for training in Phase 2. The "any" story is
          reconstructed at analysis time from per-variable predictions.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

VALID_LABEL_MODES = ("per_variable", "any")


class STGNNDataset(Dataset):
    """Sliding-window dataset for spatio-temporal graph neural networks."""

    def __init__(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray],
        input_len: int,
        horizon: int,
        task: str,
        label_mode: str = "per_variable",
        label_indices: Optional[Sequence[int]] = None,
        target_variable_index: Optional[int] = None,
    ):
        """
        Args:
            features:  (T, N, d) float32 normalised array.
            labels:    (T, N, d) int8 binary labels, or None for regression.
            input_len: historical context length.
            horizon:   prediction horizon.
            task:      'regression' or 'classification'.
            label_mode: only consulted when task == 'classification'.
                        'per_variable' (default) keeps the variable axis and
                        subsets by `label_indices`. 'any' collapses the
                        subset via ``max`` to (horizon, N).
            label_indices: indices into the variable axis to use as label
                        channels. Required when task == 'classification'.
            target_variable_index: if not None AND task == 'regression', y is
                        sliced to only that channel — shape (horizon, N, 1).
                        Matches Gaibie et al. (2024) univariate output regime
                        (one experiment per target variable). Downstream loss
                        computation must therefore compare pred[..., idx:idx+1]
                        to y — the model still outputs D channels, but only
                        the target channel contributes to the loss (functionally
                        equivalent to a single-output head).
        """
        assert task in ("regression", "classification"), f"unknown task {task}"
        if task == "classification":
            if labels is None:
                raise ValueError("classification requires precomputed labels")
            if label_mode not in VALID_LABEL_MODES:
                raise ValueError(
                    f"label_mode must be one of {VALID_LABEL_MODES}, got {label_mode!r}"
                )
            if label_indices is None:
                raise ValueError(
                    "label_indices must be provided when task == 'classification'"
                )

        # Avoid the 4× blow-up of (T, N, d) int8 labels → float32 by keeping
        # them in their on-disk dtype (typically int8) and casting the small
        # (H, N, d_label) slice at __getitem__ time. On the ERA5 999-station
        # cache this drops peak preprocessing RSS by ~5.5 GB.
        if features.dtype != np.float32:
            features = features.astype(np.float32)
        self.features   = torch.from_numpy(features)
        if labels is not None:
            self.labels = torch.from_numpy(np.ascontiguousarray(labels))
        else:
            self.labels = None
        self.input_len  = input_len
        self.horizon    = horizon
        self.task       = task
        self.label_mode = label_mode
        # Store as a LongTensor so PyTorch indexing works cleanly.
        self.label_indices = (torch.as_tensor(list(label_indices), dtype=torch.long)
                              if label_indices is not None else None)
        # Univariate regression target index (Gaibie regime). None → multivariate.
        self.target_variable_index = target_variable_index
        if (target_variable_index is not None
                and not 0 <= target_variable_index < features.shape[-1]):
            raise ValueError(
                f"target_variable_index {target_variable_index} out of range "
                f"[0, {features.shape[-1]})"
            )
        self.n_samples  = len(features) - input_len - horizon + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_start = idx
        x_end   = idx + self.input_len
        y_start = x_end
        y_end   = y_start + self.horizon

        x = self.features[x_start:x_end]                            # (input_len, N, d)

        if self.task == "regression":
            y = self.features[y_start:y_end]                        # (horizon, N, d)
            if self.target_variable_index is not None:
                # Univariate regime per Gaibie (2024). Slice to a single
                # variable channel — model still outputs D channels but
                # trainer's _compute_loss must only compare this one.
                y = y[..., self.target_variable_index:self.target_variable_index + 1]
            return x, y

        # Classification: index the pre-computed binary label tensor. Labels
        # are stored as int8; cast the small window slice to float32 here so
        # BCEWithLogitsLoss sees the dtype it expects.
        y_full = self.labels[y_start:y_end]                         # (horizon, N, d) int8
        y_sub  = y_full.index_select(-1, self.label_indices)        # (horizon, N, d_label) int8

        if self.label_mode == "any":
            return x, y_sub.max(dim=-1).values.to(torch.float32)    # (horizon, N)
        return x, y_sub.to(torch.float32)                           # (horizon, N, d_label)


def _resolve_label_indices(
    var_names: Sequence[str],
    label_variables: Optional[Sequence[str]],
) -> List[int]:
    """Look up the indices of `label_variables` in `var_names`."""
    if label_variables is None:
        return list(range(len(var_names)))
    missing = [v for v in label_variables if v not in var_names]
    if missing:
        raise ValueError(
            f"label_variables not in var_names: {missing}. "
            f"Available: {list(var_names)}"
        )
    return [var_names.index(v) for v in label_variables]


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
    label_mode: str = "per_variable",
    var_names: Optional[Sequence[str]] = None,
    label_variables: Optional[Sequence[str]] = None,
    target_variable: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader).

    When `task == 'classification'`, the label variable set is resolved once
    from `var_names` + `label_variables` and passed as an index tensor to each
    Dataset. If `label_variables` is None, all variables in `var_names` are
    used (matching the regression variable set exactly).

    When `task == 'regression'` and `target_variable` is set, the dataset
    slices y to that variable's channel (Gaibie 2024 univariate regime).
    """
    if task == "classification":
        label_indices = _resolve_label_indices(var_names or [], label_variables)
    else:
        label_indices = None

    target_variable_index = None
    if task == "regression" and target_variable is not None:
        if var_names is None:
            raise ValueError("target_variable requires var_names")
        if target_variable not in var_names:
            raise ValueError(
                f"target_variable={target_variable!r} not in var_names {list(var_names)}"
            )
        target_variable_index = list(var_names).index(target_variable)

    def _make(feats, labs, shuffle):
        ds = STGNNDataset(
            feats, labs, input_len, horizon, task,
            label_mode=label_mode, label_indices=label_indices,
            target_variable_index=target_variable_index,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=shuffle,   # drop incomplete last batch only during training
        )

    return (
        _make(train_features, train_labels, shuffle=True),
        _make(val_features,   val_labels,   shuffle=False),
        _make(test_features,  test_labels,  shuffle=False),
    )
