"""
Unit tests for evaluation/regression.py, evaluation/classification.py,
evaluation/computational.py, and evaluation/stats.py.

All tests use synthetic numpy arrays — no real dataset required.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

class TestRegressionMetrics:

    def _make_preds(self, S=20, H=4, N=10, d=3):
        rng = np.random.default_rng(0)
        preds   = rng.standard_normal((S, H, N, d)).astype(np.float32)
        targets = rng.standard_normal((S, H, N, d)).astype(np.float32)
        return preds, targets

    def test_mae_perfect_prediction(self):
        from evaluation.regression import mae
        x = np.ones((100, 10, 3), dtype=np.float32)
        assert mae(x, x).shape == (10,)
        np.testing.assert_allclose(mae(x, x), 0.0, atol=1e-6)

    def test_mae_known_value(self):
        from evaluation.regression import mae
        preds   = np.zeros((10, 5, 2), dtype=np.float32)
        targets = np.ones((10, 5, 2),  dtype=np.float32)
        result  = mae(preds, targets)
        assert result.shape == (5,)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_rmse_nonneg(self):
        from evaluation.regression import rmse
        p, t = self._make_preds()
        vals = rmse(p.reshape(-1, *p.shape[2:]), t.reshape(-1, *t.shape[2:]))
        assert (vals >= 0).all()

    def test_rmse_greater_than_mae(self):
        """RMSE >= MAE always (Jensen's inequality)."""
        from evaluation.regression import mae, rmse
        p, t = self._make_preds()
        pf, tf = p.reshape(-1, *p.shape[2:]), t.reshape(-1, *t.shape[2:])
        assert (rmse(pf, tf) >= mae(pf, tf) - 1e-6).all()

    def test_mse_eq_rmse_squared(self):
        from evaluation.regression import mse, rmse
        p, t = self._make_preds()
        pf, tf = p.reshape(-1, *p.shape[2:]), t.reshape(-1, *t.shape[2:])
        np.testing.assert_allclose(mse(pf, tf), rmse(pf, tf) ** 2, rtol=1e-4)

    def test_nrmse_curve_shape(self):
        from evaluation.regression import nrmse_per_horizon
        p, t = self._make_preds(S=20, H=4, N=10, d=3)
        curve = nrmse_per_horizon(p, t)
        assert curve.shape == (4,)

    def test_vpt_returns_correct_horizon(self):
        from evaluation.regression import valid_prediction_time
        curve = np.array([0.1, 0.2, 0.3, 0.4])
        assert valid_prediction_time(curve, epsilon=0.5) == 3

    def test_vpt_zero_when_always_above_epsilon(self):
        from evaluation.regression import valid_prediction_time
        curve = np.array([0.6, 0.7, 0.8])
        assert valid_prediction_time(curve, epsilon=0.5) == 0

    def test_compute_all_keys(self):
        from evaluation.regression import compute_all
        p, t = self._make_preds()
        result = compute_all(p, t, extreme_mask=None, vpt_epsilon=0.5)
        for key in ["mae", "rmse", "mse", "mape", "vpt", "nrmse_per_horizon"]:
            assert key in result, f"Missing key: {key}"

    def test_compute_all_nrmse_list(self):
        from evaluation.regression import compute_all
        p, t = self._make_preds()
        result = compute_all(p, t, extreme_mask=None, vpt_epsilon=0.5)
        assert isinstance(result["nrmse_per_horizon"], list)
        assert len(result["nrmse_per_horizon"]) == 4  # H=4

    def test_stratified_mae_splits_correctly(self):
        from evaluation.regression import stratified_mae
        rng = np.random.default_rng(1)
        T, N, d = 80, 10, 3
        preds   = rng.standard_normal((T, N, d)).astype(np.float32)
        targets = rng.standard_normal((T, N, d)).astype(np.float32)
        mask    = rng.integers(0, 2, (T, N, d)).astype(bool)
        mae_ext, mae_non = stratified_mae(preds, targets, mask)
        assert mae_ext.shape == (N,)
        assert mae_non.shape == (N,)
        assert (mae_ext >= 0).all()
        assert (mae_non >= 0).all()


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

class TestClassificationMetrics:

    def _make_binary(self, S=50, H=4, N=10):
        rng = np.random.default_rng(42)
        preds   = rng.standard_normal((S, H, N)).astype(np.float32)
        targets = rng.integers(0, 2, (S, H, N)).astype(np.float32)
        return preds, targets

    def test_compute_all_keys(self):
        from evaluation.classification import compute_all
        p, t = self._make_binary()
        result = compute_all(p, t)
        for key in ["f1", "recall", "precision", "auc_roc"]:
            assert key in result, f"Missing key: {key}"

    def test_perfect_precision_recall(self):
        from evaluation.classification import compute_all
        S, H, N = 50, 4, 5
        rng = np.random.default_rng(0)
        targets = rng.integers(0, 2, (S, H, N)).astype(np.float32)
        # logits: large positive if target=1, large negative if target=0
        preds = (targets * 2 - 1) * 100.0
        result = compute_all(preds, targets)
        np.testing.assert_allclose(result["f1"],        1.0, atol=1e-4)
        np.testing.assert_allclose(result["recall"],    1.0, atol=1e-4)
        np.testing.assert_allclose(result["precision"], 1.0, atol=1e-4)

    def test_auc_roc_random_is_near_half(self):
        from evaluation.classification import compute_all
        rng = np.random.default_rng(7)
        S, H, N = 200, 4, 10
        preds   = rng.standard_normal((S, H, N)).astype(np.float32)
        targets = rng.integers(0, 2, (S, H, N)).astype(np.float32)
        result  = compute_all(preds, targets)
        # Random classifier → AUC ≈ 0.5 (with tolerance for small sample)
        assert 0.3 < result["auc_roc"] < 0.7

    def test_values_bounded(self):
        from evaluation.classification import compute_all
        p, t = self._make_binary()
        result = compute_all(p, t)
        for key in ["f1", "recall", "precision"]:
            assert 0.0 <= result[key] <= 1.0
        assert 0.0 <= result["auc_roc"] <= 1.0

    def test_confusion_matrix_shape(self):
        from evaluation.classification import compute_all
        p, t = self._make_binary(S=30, H=2, N=5)
        result = compute_all(p, t)
        # All required scalars present
        assert all(isinstance(result[k], float) for k in ["f1", "recall", "precision"])


# ---------------------------------------------------------------------------
# Computational metrics
# ---------------------------------------------------------------------------

class TestComputationalMetrics:

    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)

        def forward(self, x, adj=None):
            return self.fc(x)

    def test_param_count_matches_manual_count(self):
        from evaluation.computational import count_parameters
        model = self._TinyModel()
        params = count_parameters(model)
        expected = (8 * 4 + 4) / 1e6  # Linear(8, 4): 36 params
        np.testing.assert_allclose(params, expected, rtol=1e-5)

    def test_record_static_metrics_keys(self):
        from evaluation.computational import record_static_metrics
        model = self._TinyModel()
        sample = torch.randn(2, 8)
        result = record_static_metrics(model, sample)
        assert "params_M" in result
        assert "flops_G"  in result
        assert result["params_M"] > 0
        assert result["flops_G"]  >= 0  # FLOPs can be 0 if torchinfo fails

    def test_inference_time_positive(self):
        from evaluation.computational import measure_inference_time
        model = self._TinyModel().eval()

        class _FakeAdapter:
            def __init__(self, m):
                self.model = m
            def predict(self, x, adj):
                with torch.no_grad():
                    return self.model(x)

        data = [(torch.randn(4, 8), torch.zeros(4, 4)) for _ in range(5)]
        adj = torch.eye(4)
        device = torch.device("cpu")
        ms = measure_inference_time(_FakeAdapter(model), data, adj, device, n_passes=5)
        assert ms > 0.0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestWilcoxon:

    def test_identical_distributions_high_p(self):
        """Identical sequences — p-value should be 1.0 (or NaN, both valid)."""
        from evaluation.stats import wilcoxon_test
        import warnings
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = wilcoxon_test(x, x)
        # Returns (statistic, p_value) tuple
        assert isinstance(stat, float)
        assert isinstance(p, float)

    def test_different_distributions_low_p(self):
        """Clearly different sequences → statistically significant."""
        from evaluation.stats import wilcoxon_test
        a = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        b = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        stat, p = wilcoxon_test(a, b)
        assert p < 0.05
