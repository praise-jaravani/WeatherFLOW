"""
Smoke tests for model adapters.

Verifies that each adapter's build_model(), forward(), predict(), and
get_param_count() are callable and return the correct output shapes
without requiring a full training run or GPU.

Tests use small N (10 nodes) to keep CPU runtime low. Models that are
heavyweight (MTGODE, ClimODE) are marked slow and can be skipped with
  pytest -m "not slow"
"""

import numpy as np
import pytest
import torch

# Small synthetic problem size
BATCH, T, N, D, H = 2, 12, 10, 3, 12


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
            "method_1": "euler",
            "method_2": "euler",
            "adjoint": False,
            "time_1": 1.0,
            "step_size_1": 0.5,
            "time_2": 1.0,
            "step_size_2": 0.5,
            "rtol": 1e-3,
            "atol": 1e-4,
        },
        "climode": {
            "hidden_dim": hidden_dim,
            "method": "euler",
            "adjoint": False,
            "rtol_train": 1e-3,
            "atol_train": 1e-4,
            "rtol_eval": 1e-5,
            "atol_eval": 1e-6,
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

    def test_forward_output_shape(self, adapter):
        out = adapter.forward(_fake_x(), _fake_adj())
        assert out.shape == (BATCH, H, N, D)

    def test_predict_output_shape(self, adapter):
        out = adapter.predict(_fake_x(), _fake_adj())
        assert out.shape == (BATCH, H, N, D)

    def test_param_count_positive(self, adapter):
        assert adapter.get_param_count() > 0


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

    def test_forward_output_shape(self, adapter):
        out = adapter.forward(_fake_x(), _fake_adj())
        assert out.shape == (BATCH, H, N, D)

    def test_predict_output_shape(self, adapter):
        out = adapter.predict(_fake_x(), _fake_adj())
        assert out.shape == (BATCH, H, N, D)


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
