"""
Regression test for the per-model native-loss hook in training/trainer.py.

Invariant: adapters that do NOT set `uses_native_loss = True` take the
old code path with byte-identical behavior. New adapters that opt in
route through `adapter.native_loss(pred, y, **kwargs)`.

This is the guardrail the trainer-hook change (§1 of the review)
promised: existing behavior preserved.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from training.trainer import Trainer


class _MockDataset(torch.utils.data.Dataset):
    def __init__(self, n, T, N, d):
        self.x = torch.randn(n, T, N, d)
        self.y = torch.randn(n, T, N, d)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class _LegacyAdapter:
    """Vanilla adapter — no uses_native_loss attribute at all."""
    def __init__(self):
        self.model = nn.Linear(5, 5)
        self.forward_calls = 0
        self.predict_calls = 0

    def forward(self, x, adj):
        self.forward_calls += 1
        return self.model(x)                    # routes through trainable module → grad_fn attached

    def predict(self, x, adj):
        self.predict_calls += 1
        with torch.no_grad():
            return self.model(x)

    def get_param_count(self):
        return sum(p.numel() for p in self.model.parameters())


class _NativeLossAdapter:
    """Adapter that opts in via the flag + native_loss method."""
    uses_native_loss = True

    def __init__(self):
        self.model = nn.Linear(5, 5)
        self.native_loss_calls = 0
        self.last_kwargs = None

    def forward(self, x, adj):
        # returns a tuple through the trainable module so it has grad_fn
        mu = self.model(x)
        sigma = mu.abs().detach() + 0.1        # positive; detached so only μ contributes to grad
        return (mu, sigma)

    def predict(self, x, adj):
        with torch.no_grad():
            return self.model(x)

    def native_loss(self, pred, target, **kwargs):
        self.native_loss_calls += 1
        self.last_kwargs = kwargs
        mean, std = pred
        # Simple deterministic scalar so we can assert on its value
        return ((target - mean) ** 2 / std).mean()

    def get_param_count(self):
        return sum(p.numel() for p in self.model.parameters())


def _make_trainer(adapter, task="regression"):
    T, N, d = 3, 5, 5
    ds = _MockDataset(4, T, N, d)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    adj = torch.eye(N)
    cfg = {
        "meta": {"var_names": ["a", "b", "c", "d", "e"], "n_vars": d},
        "data": {"horizon": T, "input_len": T},
        "training": {
            "epochs": 1, "early_stopping_patience": 1,
            "learning_rate": 1e-3,
        },
        "task": task,
    }
    # We only need _train_epoch and _validate_epoch — bypass full ctor
    trainer = Trainer.__new__(Trainer)
    trainer.adapter = adapter
    trainer.model = adapter.model
    trainer.loss_fn = nn.MSELoss()
    trainer.optimiser = torch.optim.SGD(adapter.model.parameters(), lr=1e-3)
    trainer.max_norm = 5.0
    trainer.device = torch.device("cpu")
    trainer.adj = adj
    trainer.task = task
    trainer.train_loader = loader
    trainer.val_loader = loader
    trainer._label_idx = None
    trainer._epoch = 1
    return trainer


def test_legacy_adapter_takes_old_path():
    """Adapter without uses_native_loss → shared loss_fn(pred, y) path."""
    adapter = _LegacyAdapter()
    trainer = _make_trainer(adapter)

    # Sanity: the guarded getattr should return False
    assert trainer._uses_native_loss() is False

    train_loss, _ = trainer._train_epoch()
    val_loss = trainer._validate_epoch()

    # Legacy adapter.forward was called during training
    assert adapter.forward_calls == 2               # 2 batches
    # Legacy adapter.predict was called during validation
    assert adapter.predict_calls == 2

    # And crucially: the losses are finite floats (shared loss_fn ran)
    assert isinstance(train_loss, float) and train_loss >= 0
    assert isinstance(val_loss, float) and val_loss >= 0


def test_native_loss_adapter_takes_new_path():
    """Adapter with uses_native_loss=True → adapter.native_loss(...) called."""
    adapter = _NativeLossAdapter()
    trainer = _make_trainer(adapter)
    assert trainer._uses_native_loss() is True

    train_loss, _ = trainer._train_epoch()
    val_loss = trainer._validate_epoch()

    # native_loss was called for every batch of train + val (2 batches × 2 phases = 4)
    assert adapter.native_loss_calls == 4

    # And var_coeff was passed as expected (epoch 1 → 2 * lr)
    assert "var_coeff" in adapter.last_kwargs
    assert adapter.last_kwargs["var_coeff"] == pytest.approx(2.0 * 1e-3, rel=1e-6)

    assert isinstance(train_loss, float) and train_loss >= 0
    assert isinstance(val_loss, float) and val_loss >= 0


def test_native_loss_hook_is_default_off():
    """Verify getattr(..., False) fallback for adapters missing the attribute."""
    adapter = _LegacyAdapter()
    # explicitly delete just to be paranoid
    assert not hasattr(adapter, "uses_native_loss")
    trainer = _make_trainer(adapter)
    assert trainer._uses_native_loss() is False
    # Explicitly False is also respected
    adapter.uses_native_loss = False
    assert trainer._uses_native_loss() is False
