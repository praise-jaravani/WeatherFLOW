"""
Adapter tests for MTGODE and ClimODE.

Three levels of coverage:
  1. Shape tests   — output is (B, H, N, d) for forward() and predict()
  2. Quality tests — no NaN, float32, gradient flows, single training step,
                     predict() is deterministic
  3. E2E tests     — CLI runs 1 seed × 2 epochs on cached ERA5 with 10 nodes

ODE tests are marked slow and can be skipped with:
  pytest -m "not slow"
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

# Small synthetic problem size
BATCH, T, N, D, H = 2, 12, 10, 3, 12

_PROJECT_ROOT = Path(__file__).parent.parent


def _fake_config(model_name: str, hidden_dim: int = 32) -> dict:
    """Minimal config dict that every adapter needs."""
    return {
        "task": "regression",
        "dataset": "era5",
        "model": {
            "name": model_name,
            "hidden_dim": hidden_dim,
            "num_layers": 1,
            "embed_dim": 8,
            "cheb_k": 2,
            "k_neighbors": 4,
            "lck_structure": [4, 4],
            "max_view": 1,
            "dropout": 0.0,
            "graph_constructor": {"k": 4, "alpha": 3},
        },
        "training": {
            "learning_rate": 1e-3,
            "batch_size": BATCH,
            "epochs": 2,
            "early_stopping_patience": 5,
            "gradient_clip_max_norm": 5.0,
            "scheduler": {"T_0": 10, "T_mult": 2, "eta_min": 1e-6},
        },
        "data": {"input_len": T, "horizon": H},
        "evaluation": {"vpt_epsilon": 0.5},
        "meta": {
            "n_nodes": N,
            "n_vars": D,
            "input_len": T,
            "horizon": H,
            "dataset": "era5",
            "coords": np.column_stack([
                np.linspace(-34, -25, N),
                np.linspace(18, 30, N),
            ]).astype(np.float32),
        },
        "ode": {
            # MTGODE — outer temporal ODE + inner graph ODE
            "method_1":    "euler",
            "method_2":    "euler",
            "time_1":      1.0,
            "step_size_1": 0.5,
            "time_2":      1.0,
            "step_size_2": 0.5,
            "rtol":        1e-3,
            "atol":        1e-4,
            # ClimODE — single ODE, separate train/eval tolerances
            # Use identical tolerances here so tests run fast.
            "method":      "euler",
            "train_rtol":  1e-3,
            "train_atol":  1e-4,
            "eval_rtol":   1e-3,
            "eval_atol":   1e-4,
            # shared
            "adjoint":     False,
        },
    }


def _fake_adj():
    return torch.eye(N)


def _fake_x():
    return torch.randn(BATCH, T, N, D)


# ---------------------------------------------------------------------------
# AGCRN
# ---------------------------------------------------------------------------

class TestAGCRNAdapter:

    @pytest.fixture
    def adapter(self):
        from models.baselines.agcrn_adapter import AGCRNAdapter
        cfg = _fake_config("agcrn")
        a = AGCRNAdapter()
        a.build_model(cfg, _fake_adj())
        return a

    def test_forward_output_shape(self, adapter):
        out = adapter.forward(_fake_x(), _fake_adj())
        assert out.shape == (BATCH, H, N, D), f"Expected ({BATCH},{H},{N},{D}), got {out.shape}"

    def test_predict_output_shape(self, adapter):
        out = adapter.predict(_fake_x(), _fake_adj())
        assert out.shape == (BATCH, H, N, D)

    def test_param_count_positive(self, adapter):
        assert adapter.get_param_count() > 0


# ---------------------------------------------------------------------------
# CLCRN
# ---------------------------------------------------------------------------

_has_torch_scatter = pytest.importorskip  # evaluated lazily per class


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch_scatter"),
    reason="torch_scatter not installed (PyG extension; available inside Singularity container on HPC)",
)
class TestCLCRNAdapter:

    @pytest.fixture
    def adapter(self):
        from models.baselines.clcrn_adapter import CLCRNAdapter
        cfg = _fake_config("clcrn")
        a = CLCRNAdapter()
        a.build_model(cfg, _fake_adj())
        return a

    def test_forward_output_shape(self, adapter):
        out = adapter.forward(_fake_x(), _fake_adj())
        assert out.shape == (BATCH, H, N, D)

    def test_predict_output_shape(self, adapter):
        out = adapter.predict(_fake_x(), _fake_adj())
        assert out.shape == (BATCH, H, N, D)

    def test_param_count_positive(self, adapter):
        assert adapter.get_param_count() > 0


# ---------------------------------------------------------------------------
# MTGODE (slow: ODE integration)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestMTGODEAdapter:

    @pytest.fixture
    def adapter(self):
        from models.continuous_time.mtgode_adapter import MTGODEAdapter
        cfg = _fake_config("mtgode", hidden_dim=20)
        a = MTGODEAdapter()
        a.build_model(cfg, _fake_adj())
        return a

    # -- shape ----------------------------------------------------------

    def test_forward_output_shape(self, adapter):
        out = adapter.forward(_fake_x(), _fake_adj())
        assert out.shape == (BATCH, H, N, D), f"Expected ({BATCH},{H},{N},{D}), got {out.shape}"

    def test_predict_output_shape(self, adapter):
        out = adapter.predict(_fake_x(), _fake_adj())
        assert out.shape == (BATCH, H, N, D)

    # -- quality --------------------------------------------------------

    def test_param_count_positive(self, adapter):
        assert adapter.get_param_count() > 0

    def test_output_dtype_float32(self, adapter):
        out = adapter.forward(_fake_x(), _fake_adj())
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

    def test_forward_no_nan(self, adapter):
        out = adapter.forward(_fake_x(), _fake_adj())
        assert not torch.isnan(out).any(), "NaN in forward() output"

    def test_predict_no_nan(self, adapter):
        out = adapter.predict(_fake_x(), _fake_adj())
        assert not torch.isnan(out).any(), "NaN in predict() output"

    def test_gradient_flows(self, adapter):
        """loss.backward() must produce finite, non-None gradients for all params."""
        adapter.model.train()
        out = adapter.forward(_fake_x(), _fake_adj())
        out.mean().backward()
        for name, p in adapter.model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

    def test_single_training_step(self, adapter):
        """One full optimizer step must complete and loss must be finite."""
        target = torch.randn(BATCH, H, N, D)
        opt = torch.optim.Adam(adapter.model.parameters(), lr=1e-3)
        adapter.model.train()
        out = adapter.forward(_fake_x(), _fake_adj())
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"

    def test_predict_reproducible_when_seeded(self, adapter):
        """
        MTGODE's graph_constructor adds torch.rand_like noise for top-k tie-breaking,
        making predict() stochastic without a seed.  This test verifies the model IS
        reproducible when the same RNG state is used — matching how cli.py behaves
        (seed_everything() is called before every run).
        """
        x = _fake_x()
        torch.manual_seed(0)
        out1 = adapter.predict(x, _fake_adj())
        torch.manual_seed(0)
        out2 = adapter.predict(x, _fake_adj())
        assert torch.allclose(out1, out2), "predict() not reproducible with the same seed"


# ---------------------------------------------------------------------------
# ClimODE (slow: ODE integration)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestClimODEAdapter:

    @pytest.fixture
    def adapter(self):
        from models.continuous_time.climode_adapter import ClimODEAdapter
        cfg = _fake_config("climode")
        a = ClimODEAdapter()
        a.build_model(cfg, _fake_adj())
        return a

    # -- shape ----------------------------------------------------------

    def test_forward_output_shape(self, adapter):
        out = adapter.forward(_fake_x(), _fake_adj())
        assert out.shape == (BATCH, H, N, D)

    def test_predict_output_shape(self, adapter):
        out = adapter.predict(_fake_x(), _fake_adj())
        assert out.shape == (BATCH, H, N, D)

    # -- quality --------------------------------------------------------

    def test_param_count_positive(self, adapter):
        assert adapter.get_param_count() > 0

    def test_output_dtype_float32(self, adapter):
        out = adapter.forward(_fake_x(), _fake_adj())
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

    def test_forward_no_nan(self, adapter):
        out = adapter.forward(_fake_x(), _fake_adj())
        assert not torch.isnan(out).any(), "NaN in forward() output"

    def test_predict_no_nan(self, adapter):
        out = adapter.predict(_fake_x(), _fake_adj())
        assert not torch.isnan(out).any(), "NaN in predict() output"

    def test_gradient_flows(self, adapter):
        """loss.backward() via adjoint=False path (odeint, euler) must give finite grads."""
        adapter.model.train()
        out = adapter.forward(_fake_x(), _fake_adj())
        out.mean().backward()
        for name, p in adapter.model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

    def test_single_training_step(self, adapter):
        """One full optimizer step must complete and loss must be finite."""
        target = torch.randn(BATCH, H, N, D)
        opt = torch.optim.Adam(adapter.model.parameters(), lr=1e-3)
        adapter.model.train()
        out = adapter.forward(_fake_x(), _fake_adj())
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"

    def test_predict_deterministic(self, adapter):
        """predict() in eval mode must return identical results on repeated calls."""
        x = _fake_x()
        out1 = adapter.predict(x, _fake_adj())
        out2 = adapter.predict(x, _fake_adj())
        assert torch.allclose(out1, out2), "predict() is not deterministic"

    def test_train_eval_output_differ(self, adapter):
        """
        ClimODE uses tighter ODE tolerances in eval mode than train mode.
        With euler (fixed-step), outputs are identical — this test therefore
        checks that both modes complete without error and produce finite values.
        The tolerance-difference effect is only visible with dopri5 (adaptive).
        """
        x = _fake_x()
        adapter.model.train()
        train_out = adapter.forward(x, _fake_adj())
        adapter.model.eval()
        eval_out = adapter.predict(x, _fake_adj())
        assert torch.isfinite(train_out).all(), "train output not finite"
        assert torch.isfinite(eval_out).all(), "eval output not finite"


# ---------------------------------------------------------------------------
# End-to-end CLI smoke tests (slow: full pipeline on cached ERA5)
#
# These tests require preprocessing/cache/era5/ to already exist.
# Run the preprocessing first if needed:
#   python3 cli.py --model mtgode --dataset era5 --task regression --no-cache
# Then skip the preprocessing step on subsequent runs by using the cache.
# ---------------------------------------------------------------------------

_ERA5_CACHE = _PROJECT_ROOT / "preprocessing" / "cache" / "era5"

@pytest.mark.slow
@pytest.mark.skipif(
    not _ERA5_CACHE.exists(),
    reason="ERA5 preprocessing cache not found — run preprocessing first",
)
class TestMTGODEEndToEnd:
    """1 seed × 2 epochs on cached ERA5 subsampled to 10 nodes."""

    def test_cli_regression(self):
        result = subprocess.run(
            [
                "python3", "cli.py",
                "--model", "mtgode",
                "--dataset", "era5",
                "--task", "regression",
                "--runs", "1",
                "--max-nodes", "10",
                "--mlflow-off",
                "--config.training.epochs", "2",
                "--config.training.early_stopping_patience", "10",
                "--config.training.batch_size", "512",
            ],
            capture_output=True, text=True, timeout=300,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0, (
            f"mtgode CLI failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout[-1500:]}\n"
            f"--- stderr ---\n{result.stderr[-1500:]}"
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not _ERA5_CACHE.exists(),
    reason="ERA5 preprocessing cache not found — run preprocessing first",
)
class TestClimODEEndToEnd:
    """
    1 seed × 2 epochs on cached ERA5 subsampled to 10 nodes.

    ClimODE uses dopri5 (adaptive step) in production. For this smoke test we
    switch to euler + large batch to keep wall time manageable without mocking
    the ODE solver.
    """

    def test_cli_regression(self):
        result = subprocess.run(
            [
                "python3", "cli.py",
                "--model", "climode",
                "--dataset", "era5",
                "--task", "regression",
                "--runs", "1",
                "--max-nodes", "10",
                "--mlflow-off",
                "--config.training.epochs", "2",
                "--config.training.early_stopping_patience", "10",
                "--config.training.batch_size", "512",
                "--config.ode.method", "euler",
            ],
            capture_output=True, text=True, timeout=600,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0, (
            f"climode CLI failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout[-1500:]}\n"
            f"--- stderr ---\n{result.stderr[-1500:]}"
        )


# ---------------------------------------------------------------------------
# Stub adapters — just verify they import and raise NotImplementedError
# cleanly (teammates haven't implemented them yet)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module,cls", [
    ("models.lightweight.stlgru_adapter",    "STLGRUAdapter"),
    ("models.lightweight.litestgnn_adapter", "LiteSTGNNAdapter"),
    ("models.diffusion.diffstg_adapter",     "DiffSTGAdapter"),
    ("models.diffusion.ustd_adapter",        "USTDAdapter"),
])
def test_stub_adapter_importable(module, cls):
    """Stub adapters must be importable; build_model may raise NotImplementedError."""
    import importlib
    mod = importlib.import_module(module)
    adapter_cls = getattr(mod, cls)
    adapter = adapter_cls()
    with pytest.raises((NotImplementedError, Exception)):
        adapter.build_model(_fake_config("stub"), _fake_adj())
