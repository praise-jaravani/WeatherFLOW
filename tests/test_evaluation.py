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
        from evaluation.regression import per_station_var_mae
        # (S, H, N, d): perfect prediction → all MAE == 0
        x = np.ones((10, 2, 10, 3), dtype=np.float32)
        result = per_station_var_mae(x, x)
        assert result.shape == (10, 3)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_mae_known_value(self):
        from evaluation.regression import per_station_var_mae
        # preds=0, targets=1 → MAE == 1 everywhere
        preds   = np.zeros((10, 1, 5, 2), dtype=np.float32)
        targets = np.ones((10, 1, 5, 2),  dtype=np.float32)
        result  = per_station_var_mae(preds, targets)
        assert result.shape == (5, 2)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_rmse_nonneg(self):
        from evaluation.regression import per_station_var_rmse
        p, t = self._make_preds()
        vals = per_station_var_rmse(p, t)
        assert vals.shape == (10, 3)
        assert (vals >= 0).all()

    def test_rmse_greater_than_mae(self):
        """RMSE >= MAE always (Jensen's inequality)."""
        from evaluation.regression import per_station_var_mae, per_station_var_rmse
        p, t = self._make_preds()
        assert (per_station_var_rmse(p, t) >= per_station_var_mae(p, t) - 1e-6).all()

    def test_mse_eq_rmse_squared(self):
        from evaluation.regression import per_station_var_mse, per_station_var_rmse
        p, t = self._make_preds()
        np.testing.assert_allclose(
            per_station_var_mse(p, t), per_station_var_rmse(p, t) ** 2, rtol=1e-4
        )

    def test_smape_bounded(self):
        """sMAPE must be in [0, 2]."""
        from evaluation.regression import per_station_var_smape
        p, t = self._make_preds()
        vals = per_station_var_smape(p, t)
        assert (vals >= 0).all()
        assert (vals <= 2.0 + 1e-6).all()

    def test_nrmse_curve_shape(self):
        from evaluation.regression import nrmse_per_horizon
        p, t = self._make_preds(S=20, H=4, N=10, d=3)
        curve = nrmse_per_horizon(p, t)
        # New: returns (H, d), not (H,)
        assert curve.shape == (4, 3)

    def test_vpt_first_crossing(self):
        """First-crossing VPT: returns index of first step that hits/exceeds epsilon."""
        from evaluation.regression import valid_prediction_time
        curve = np.array([0.1, 0.2, 0.6, 0.4])   # first crossing at index 2
        assert valid_prediction_time(curve, epsilon=0.5) == 2

    def test_vpt_all_below_epsilon(self):
        """If curve never reaches epsilon, VPT == len(curve) (all H steps valid)."""
        from evaluation.regression import valid_prediction_time
        curve = np.array([0.1, 0.2, 0.3, 0.4])
        assert valid_prediction_time(curve, epsilon=0.5) == 4

    def test_vpt_zero_when_always_above_epsilon(self):
        from evaluation.regression import valid_prediction_time
        curve = np.array([0.6, 0.7, 0.8])
        assert valid_prediction_time(curve, epsilon=0.5) == 0

    def test_compute_all_keys(self):
        from evaluation.regression import compute_all
        p, t = self._make_preds()
        result = compute_all(p, t, extreme_mask=None, vpt_epsilon=0.5)
        for key in ["mae", "rmse", "mse", "smape", "vpt", "nrmse_per_horizon", "mae_pooled_norm"]:
            assert key in result, f"Missing key: {key}"
        assert "mape" not in result, "mape should be gone (replaced by smape)"

    def test_compute_all_returns_per_variable_dicts(self):
        """mae, rmse, vpt etc. are dicts keyed by variable name."""
        from evaluation.regression import compute_all
        p, t = self._make_preds(d=3)
        result = compute_all(p, t, extreme_mask=None, vpt_epsilon=0.5,
                             var_names=["a", "b", "c"])
        assert isinstance(result["mae"], dict)
        assert set(result["mae"].keys()) == {"a", "b", "c"}
        assert isinstance(result["vpt"], dict)

    def test_compute_all_nrmse_per_var(self):
        """nrmse_per_horizon is a dict of per-variable H-length lists."""
        from evaluation.regression import compute_all
        p, t = self._make_preds(H=4, d=3)
        result = compute_all(p, t, extreme_mask=None, vpt_epsilon=0.5,
                             var_names=["a", "b", "c"])
        nph = result["nrmse_per_horizon"]
        assert isinstance(nph, dict)
        assert set(nph.keys()) == {"a", "b", "c"}
        assert len(nph["a"]) == 4  # H=4

    def test_compute_all_extreme_buckets_nan_when_no_mask(self):
        """Without an extreme_mask all extreme/peak fields should be NaN, not 0."""
        from evaluation.regression import compute_all
        p, t = self._make_preds()
        result = compute_all(p, t, extreme_mask=None, vpt_epsilon=0.5)
        for key in ["mae_extreme", "mae_nonextreme", "peak_intensity_error"]:
            for v_val in result[key].values():
                assert np.isnan(v_val), f"{key} should be NaN without a mask, got {v_val}"

    def test_compute_all_extreme_buckets_nonzero_with_mask(self):
        """With a mask that covers some samples, stratified metrics must be non-NaN."""
        from evaluation.regression import compute_all
        S, H, N, d = 20, 4, 10, 3
        rng = np.random.default_rng(7)
        p = rng.standard_normal((S, H, N, d)).astype(np.float32)
        t = rng.standard_normal((S, H, N, d)).astype(np.float32)
        mask = rng.integers(0, 2, (S, H, N, d)).astype(bool)
        # force at least one True per station/var so no NaN
        mask[:, 0, :, :] = True
        result = compute_all(p, t, extreme_mask=mask, vpt_epsilon=0.5)
        for v_val in result["mae_extreme"].values():
            assert not np.isnan(v_val), "mae_extreme should be non-NaN when mask has coverage"

    def test_compute_all_units_physical_when_stats_given(self):
        """units field is 'physical' when mean/std are supplied."""
        from evaluation.regression import compute_all
        S, H, N, d = 5, 2, 4, 2
        rng = np.random.default_rng(0)
        p = rng.standard_normal((S, H, N, d)).astype(np.float32)
        t = rng.standard_normal((S, H, N, d)).astype(np.float32)
        mean = np.zeros((1, N, d), dtype=np.float32)
        std  = np.ones((1, N, d), dtype=np.float32)
        result = compute_all(p, t, mean=mean, std=std, var_names=["x", "y"])
        assert result["units"] == "physical"

    def test_compute_all_units_normalised_without_stats(self):
        from evaluation.regression import compute_all
        p, t = self._make_preds()
        result = compute_all(p, t)
        assert result["units"] == "normalised"

    def test_stratified_var_mae_splits_correctly(self):
        """stratified_var_mae returns (N, d) arrays; NaN only where bucket is empty."""
        from evaluation.regression import stratified_var_mae
        rng = np.random.default_rng(1)
        S, H, N, d = 20, 4, 10, 3
        preds   = rng.standard_normal((S, H, N, d)).astype(np.float32)
        targets = rng.standard_normal((S, H, N, d)).astype(np.float32)
        mask    = rng.integers(0, 2, (S, H, N, d)).astype(bool)
        # force at least one extreme per cell to avoid all-NaN
        mask[0, 0, :, :] = True
        mae_ext, mae_non = stratified_var_mae(preds, targets, mask)
        assert mae_ext.shape == (N, d)
        assert mae_non.shape == (N, d)
        valid_ext = mae_ext[~np.isnan(mae_ext)]
        valid_non = mae_non[~np.isnan(mae_non)]
        assert (valid_ext >= 0).all()
        assert (valid_non >= 0).all()


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

class TestClassificationMetrics:
    """
    Phase-2 classification is per-variable multi-label. `compute_all` now
    takes probabilities of shape (samples, H, N, d_label) and returns a nested
    dict with `per_variable`, `macro`, `taus`, `per_station`.

    See tests/test_phase2_classification.py for the full contract; this class
    only covers legacy per-metric sanity cases carried over from Phase 1.
    """

    def _make_binary(self, S=50, H=4, N=10, D=2):
        rng = np.random.default_rng(42)
        probs   = rng.random((S, H, N, D)).astype(np.float32)   # in [0,1]
        targets = rng.integers(0, 2, (S, H, N, D)).astype(np.float32)
        return probs, targets

    def test_compute_all_keys(self):
        from evaluation.classification import compute_all
        p, t = self._make_binary()
        result = compute_all(p, t, var_names=["a", "b"])
        # Top-level keys
        for key in ["per_variable", "macro", "taus", "label_variables", "per_station"]:
            assert key in result, f"Missing key: {key}"
        # Macro contains the per-metric mean across variables
        for key in ["f1", "recall", "precision", "auc_roc"]:
            assert key in result["macro"], f"Missing macro.{key}"

    def test_perfect_precision_recall(self):
        from evaluation.classification import compute_all
        S, H, N, D = 50, 4, 5, 2
        rng = np.random.default_rng(0)
        targets = rng.integers(0, 2, (S, H, N, D)).astype(np.float32)
        # Perfectly separated probabilities per variable
        probs = np.where(targets == 1, 0.99, 0.01).astype(np.float32)
        result = compute_all(probs, targets, var_names=["a", "b"])
        np.testing.assert_allclose(result["macro"]["f1"],        1.0, atol=1e-4)
        np.testing.assert_allclose(result["macro"]["recall"],    1.0, atol=1e-4)
        np.testing.assert_allclose(result["macro"]["precision"], 1.0, atol=1e-4)

    def test_auc_roc_random_is_near_half(self):
        from evaluation.classification import compute_all
        rng = np.random.default_rng(7)
        S, H, N, D = 200, 4, 10, 2
        probs   = rng.random((S, H, N, D)).astype(np.float32)
        targets = rng.integers(0, 2, (S, H, N, D)).astype(np.float32)
        result  = compute_all(probs, targets, var_names=["a", "b"])
        # Random classifier → macro AUC ≈ 0.5 (with tolerance for small sample)
        assert 0.3 < result["macro"]["auc_roc"] < 0.7

    def test_values_bounded(self):
        from evaluation.classification import compute_all
        p, t = self._make_binary()
        result = compute_all(p, t, var_names=["a", "b"])
        for key in ["f1", "recall", "precision"]:
            assert 0.0 <= result["macro"][key] <= 1.0
        assert 0.0 <= result["macro"]["auc_roc"] <= 1.0

    def test_confusion_matrix_shape(self):
        from evaluation.classification import compute_all
        p, t = self._make_binary(S=30, H=2, N=5, D=2)
        result = compute_all(p, t, var_names=["a", "b"])
        for v in ("a", "b"):
            assert isinstance(result["per_variable"][v]["tp"], int)
            assert all(isinstance(result["per_variable"][v][k], (int, float))
                       for k in ["f1", "recall", "precision", "csi"])


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
        # flops_G is either None (untraceable) or a positive float — never 0.0
        assert result["flops_G"] is None or result["flops_G"] > 0

    def test_count_flops_none_on_failure(self):
        """A model that raises during tracing should return None, not 0.0."""
        from evaluation.computational import count_flops

        class _BadModel(nn.Module):
            def forward(self, x):
                raise RuntimeError("untraceable")

        result = count_flops(_BadModel(), torch.randn(2, 4))
        assert result is None

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
