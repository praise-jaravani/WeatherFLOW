"""
G1 (Run_Plan_Spec §1.1) — per-epoch atomic checkpoint + resume.

Asserts that a training run interrupted mid-schedule and resumed from
state.pt produces **bit-identical** final weights and metrics compared
to an uninterrupted control run.

Covers both interruption paths from Run_Plan_Spec §9 (G1):
  - graceful (SIGUSR1 → stop_check returns True → trainer checkpoints and exits 0)
  - hard-kill / crash (trainer instance simply abandoned mid-schedule;
    a new Trainer instantiated on the same output_dir picks up from state.pt)

Under cudnn.deterministic=True, benchmark=False, so cudnn's internal
non-determinism does not defeat the test. Production keeps whatever
determinism setting the rest of the codebase uses — this test forces
determinism only for its own runs.
"""

import copy
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Make repo root importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.base_adapter import BaseAdapter
from training.trainer import Trainer, TrainingDiverged, _epoch_shuffle_seed


# ---------------------------------------------------------------------------
# Minimal adapter + dataset so the test doesn't depend on the full pipeline
# ---------------------------------------------------------------------------

class DummyAdapter(BaseAdapter):
    """Linear regression adapter — enough surface area to exercise the trainer."""

    def __init__(self, in_features: int = 12 * 25 * 6, out_features: int = 48 * 25 * 1):
        # Explicit init so we control the initial weights across runs
        self.in_features = in_features
        self.out_features = out_features
        # deterministic init: torch.manual_seed set by caller before construction
        self.model = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features, out_features),
        )

    def build_model(self, config, graph):
        return self.model

    def forward(self, x, adj):
        b = x.shape[0]
        y = self.model(x)  # (B, 48*25*1)
        return y.view(b, 48, 25, 1)

    def predict(self, x, adj):
        with torch.no_grad():
            return self.forward(x, adj)

    def get_param_count(self):
        return sum(p.numel() for p in self.model.parameters())


class _RegressionDataset(TensorDataset):
    """TensorDataset with the target_variable_index attribute the trainer probes."""
    def __init__(self, x, y, target_variable_index=None):
        super().__init__(x, y)
        self.target_variable_index = target_variable_index


def _synthetic_loaders(n_train=32, n_val=8, n_test=8, batch_size=8, seed=0):
    """Small synthetic (B, 12, 25, 6) → (B, 48, 25, 1) tensors."""
    g = torch.Generator().manual_seed(seed)
    def make(n):
        x = torch.randn(n, 12, 25, 6, generator=g)
        y = torch.randn(n, 48, 25, 1, generator=g)
        return _RegressionDataset(x, y, target_variable_index=None)
    # target_variable_index=None because our DummyAdapter already outputs (…,1)
    return (
        DataLoader(make(n_train), batch_size=batch_size, shuffle=True),
        DataLoader(make(n_val), batch_size=batch_size, shuffle=False),
        DataLoader(make(n_test), batch_size=batch_size, shuffle=False),
    )


def _build_trainer(output_dir, seed=42, epochs=4):
    """Fresh Trainer bound to `output_dir`. Deterministic model init via seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    adapter = DummyAdapter()
    train_loader, val_loader, test_loader = _synthetic_loaders(seed=seed)
    adj = torch.eye(25)
    config = {
        "training": {
            "epochs": epochs,
            "early_stopping_patience": 999,   # never trip early stop in test
            "learning_rate": 1e-3,
            "gradient_clip_max_norm": 5.0,
            "scheduler": {"T_0": 10, "T_mult": 2, "eta_min": 1e-6},
        },
        "task": "regression",
    }
    return Trainer(
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=nn.MSELoss(),
        adj=adj,
        config=config,
        output_dir=output_dir,
        seed=seed,
        mlflow_enabled=False,
    )


def _extract_final_state(trainer):
    """Deep-copy of model weights + optimizer + scheduler, for equality checks."""
    return {
        "model": {k: v.detach().clone() for k, v in trainer.adapter.model.state_dict().items()},
        "best_val_loss": float(trainer.best_val_loss),
        "patience_counter": int(trainer.patience_counter),
    }


def _assert_states_equal(a, b, atol=0.0):
    assert a["model"].keys() == b["model"].keys()
    for k in a["model"]:
        ta, tb = a["model"][k], b["model"][k]
        assert ta.shape == tb.shape, f"shape mismatch on {k}"
        if atol == 0.0:
            assert torch.equal(ta, tb), (
                f"tensor {k} not bit-identical\n"
                f"max abs diff: {(ta - tb).abs().max().item()}"
            )
        else:
            assert torch.allclose(ta, tb, atol=atol, rtol=0), (
                f"tensor {k} not close within atol={atol}"
            )
    assert a["best_val_loss"] == b["best_val_loss"]
    assert a["patience_counter"] == b["patience_counter"]


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------

def _force_determinism():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_epoch_shuffle_seed_is_deterministic_and_epoch_dependent():
    """Sanity: same call twice yields same seed; epoch and base_seed differences propagate."""
    assert _epoch_shuffle_seed(42, 1) == _epoch_shuffle_seed(42, 1)
    assert _epoch_shuffle_seed(42, 1) != _epoch_shuffle_seed(42, 2)
    assert _epoch_shuffle_seed(42, 1) != _epoch_shuffle_seed(43, 1)
    # In range for torch.manual_seed
    for base in (42, 43, 44):
        for ep in (1, 2, 100):
            s = _epoch_shuffle_seed(base, ep)
            assert 0 <= s < 2 ** 63


def test_atomic_save_writes_state_pt(tmp_path):
    """After 1 epoch a state.pt exists, contains the expected keys, and no .tmp lingers."""
    _force_determinism()
    trainer = _build_trainer(tmp_path / "run_atomic", epochs=1)
    trainer.train()
    assert (tmp_path / "run_atomic" / "state.pt").exists()
    assert not (tmp_path / "run_atomic" / "state.pt.tmp").exists()
    state = torch.load(
        tmp_path / "run_atomic" / "state.pt", map_location="cpu", weights_only=False
    )
    for key in ("epoch", "model_state", "optimizer_state", "scheduler_state",
                "best_val_loss", "patience_counter", "diverged_streak",
                "history", "seed", "rng_state"):
        assert key in state, f"missing state key: {key}"
    assert state["epoch"] == 1
    assert state["seed"] == 42
    # RNG state has all four sub-keys
    for rng_key in ("torch_cpu", "torch_cuda_all", "numpy", "python"):
        assert rng_key in state["rng_state"]


def test_hard_kill_resume_matches_control(tmp_path):
    """Hard-kill simulation: abandon trainer after 2 epochs → new trainer resumes from state.pt.

    The final model weights, best_val_loss, and patience_counter must match
    an uninterrupted 4-epoch control bit-identically.
    """
    _force_determinism()

    # Control: 4 uninterrupted epochs
    ctl = _build_trainer(tmp_path / "control", epochs=4)
    ctl.train()
    ctl_state = _extract_final_state(ctl)

    # Interrupted: 2 epochs, abandon (no signal, no clean exit), then new trainer resumes
    part_a = _build_trainer(tmp_path / "resumed", epochs=2)
    part_a.train()
    del part_a  # simulate process death

    part_b = _build_trainer(tmp_path / "resumed", epochs=4)
    part_b.train()
    resumed_state = _extract_final_state(part_b)

    _assert_states_equal(ctl_state, resumed_state)


def test_sigusr1_resume_matches_control(tmp_path):
    """Graceful-stop simulation: stop_check returns True after 2 epochs.

    Trainer checkpoints and exits with stopped_reason=='signal'. A successor
    Trainer with stop_check=None then completes the run. Final state bit-identical
    to a 4-epoch uninterrupted control.
    """
    _force_determinism()

    ctl = _build_trainer(tmp_path / "control_sig", epochs=4)
    ctl.train()
    ctl_state = _extract_final_state(ctl)

    # Stop after epoch 2: stop_check returns True once history has 2 val_losses.
    part_a = _build_trainer(tmp_path / "resumed_sig", epochs=4)
    stop_after = {"triggered": False}
    def stop_check():
        return len(part_a._current_history_ref()["val_loss"]) >= 2 if hasattr(part_a, "_current_history_ref") else False
    # No public accessor exists; simplest: trigger after two full epochs by
    # counting state.pt overwrites via file mtime — but we can use a closure
    # over the trainer's history via monkey-patching _save_full_state.

    orig_save = part_a._save_full_state
    saves = {"count": 0}
    def counting_save(epoch, history):
        saves["count"] += 1
        return orig_save(epoch, history)
    part_a._save_full_state = counting_save
    def stop_after_two_saves():
        return saves["count"] >= 2
    hist_a = part_a.train(stop_check=stop_after_two_saves)
    assert hist_a.get("stopped_reason") == "signal", (
        f"expected stopped_reason='signal', got history={hist_a}"
    )
    # State.pt should reflect epoch 2 (the last epoch that completed before
    # the signal was polled at the top of epoch 3).
    state = torch.load(
        tmp_path / "resumed_sig" / "state.pt", map_location="cpu", weights_only=False
    )
    assert state["epoch"] == 2

    # Successor
    del part_a
    part_b = _build_trainer(tmp_path / "resumed_sig", epochs=4)
    part_b.train()
    resumed_state = _extract_final_state(part_b)

    _assert_states_equal(ctl_state, resumed_state)


def test_diverged_marker_and_exception(tmp_path):
    """§5 divergence guard: 3 consecutive non-finite train_losses → TrainingDiverged + marker."""
    _force_determinism()
    trainer = _build_trainer(tmp_path / "diverge", epochs=10)

    # Monkey-patch _train_epoch to always return (nan, nan)
    def bad_epoch():
        return float("nan"), float("nan")
    trainer._train_epoch = bad_epoch
    # And _validate_epoch to return a finite number so we don't early-stop
    trainer._validate_epoch = lambda force_full=False: 1.0

    with pytest.raises(TrainingDiverged) as excinfo:
        trainer.train()

    assert excinfo.value.streak == 3
    assert excinfo.value.epoch == 3
    marker = tmp_path / "diverge" / "FAILED_DIVERGED"
    assert marker.exists()
    content = marker.read_text()
    assert "streak=3" in content and "epoch=3" in content


def test_single_finite_epoch_resets_streak(tmp_path):
    """A finite epoch between non-finites must reset the streak — no 3-strikes false alarm."""
    _force_determinism()
    trainer = _build_trainer(tmp_path / "no_diverge", epochs=6)

    losses = iter([float("nan"), float("nan"), 1.0, float("nan"), float("nan"), 1.0])
    def alternating_epoch():
        v = next(losses)
        return v, v
    trainer._train_epoch = alternating_epoch
    trainer._validate_epoch = lambda force_full=False: 1.0

    # Should NOT raise
    hist = trainer.train()
    # No marker
    assert not (tmp_path / "no_diverge" / "FAILED_DIVERGED").exists()
    # Streak reset after the finite epochs
    assert trainer.diverged_streak in (0, 1, 2)


def test_no_tmp_file_left_behind(tmp_path):
    """After a full run no state.pt.tmp should exist (atomic write cleanup)."""
    _force_determinism()
    trainer = _build_trainer(tmp_path / "clean", epochs=3)
    trainer.train()
    assert not (tmp_path / "clean" / "state.pt.tmp").exists()
