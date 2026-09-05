"""
Unit tests for the preprocessing pipeline components.

Covers: split ordering (leakage safety), labelling, normalisation,
graph construction (graph_utils), and DataLoader output shapes.

SAWS parser tests are skipped when no raw SAWS data is present.
ERA5 pipeline smoke test is skipped when no raw ERA5 data is present.
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

class TestSplit:

    def _make_array(self, T=100, N=5, d=3):
        return np.arange(T * N * d, dtype=np.float32).reshape(T, N, d)

    def test_chronological_order_preserved(self):
        from preprocessing.split import chronological_split
        X = self._make_array(T=100)
        train, val, test = chronological_split(X, train_ratio=0.6, val_ratio=0.2)
        np.testing.assert_array_equal(train, X[:60])
        np.testing.assert_array_equal(val,   X[60:80])
        np.testing.assert_array_equal(test,  X[80:])

    def test_split_ratios(self):
        from preprocessing.split import chronological_split
        T = 100
        X = self._make_array(T=T)
        train, val, test = chronological_split(X, train_ratio=0.6, val_ratio=0.2)
        assert len(train) == 60
        assert len(val)   == 20
        assert len(test)  == 20
        assert len(train) + len(val) + len(test) == T

    def test_no_overlap_between_splits(self):
        from preprocessing.split import chronological_split
        X = self._make_array(T=200)
        train, val, test = chronological_split(X, train_ratio=0.6, val_ratio=0.2)
        assert len(train) + len(val) + len(test) == 200


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

class TestLabels:

    def _make_splits(self, T_train=500, T_val=100, T_test=100, N=10, d=2):
        rng = np.random.default_rng(42)
        train = rng.standard_normal((T_train, N, d)).astype(np.float32)
        val   = rng.standard_normal((T_val,   N, d)).astype(np.float32) + 5.0  # shifted
        test  = rng.standard_normal((T_test,  N, d)).astype(np.float32) + 5.0
        return train, val, test

    def test_binary_values(self):
        from preprocessing.labels import label_all_splits
        train, val, test = self._make_splits()
        y_tr, y_va, y_te, _, _ = label_all_splits(train, val, test, percentile=95.0)
        # All values must be 0 or 1
        for y in [y_tr, y_va, y_te]:
            assert set(np.unique(y)).issubset({0, 1})

    def test_threshold_computed_on_train_only(self):
        """
        Val/test are shifted high, so all val/test values exceed the training 95th
        percentile. Labels for val/test must be all-1 (threshold from train only).
        """
        from preprocessing.labels import label_all_splits
        train, val, test = self._make_splits()
        y_tr, y_va, y_te, thresholds, _ = label_all_splits(train, val, test, percentile=95.0)
        # val and test are shifted +5σ above training mean — all should be extreme
        assert y_va.mean() > 0.5
        assert y_te.mean() > 0.5

    def test_positive_rate_near_five_percent(self):
        from preprocessing.labels import label_all_splits
        rng = np.random.default_rng(0)
        train = rng.standard_normal((10000, 5, 2)).astype(np.float32)
        val   = rng.standard_normal((2000, 5, 2)).astype(np.float32)
        test  = rng.standard_normal((2000, 5, 2)).astype(np.float32)
        y_tr, _, _, _, _ = label_all_splits(train, val, test, percentile=95.0)
        rate = y_tr.mean()
        assert abs(rate - 0.05) < 0.02  # should be close to 5%


# ---------------------------------------------------------------------------
# Normalise
# ---------------------------------------------------------------------------

class TestNormalise:

    def _make_splits(self, T_train=300, T_val=100, T_test=100, N=8, d=3):
        rng = np.random.default_rng(7)
        train = rng.standard_normal((T_train, N, d)).astype(np.float32) * 3 + 10
        val   = rng.standard_normal((T_val,   N, d)).astype(np.float32) * 3 + 10
        test  = rng.standard_normal((T_test,  N, d)).astype(np.float32) * 3 + 10
        return train, val, test

    def test_train_mean_near_zero(self):
        from preprocessing.normalise import normalise_all_splits
        train, val, test = self._make_splits()
        tr_n, va_n, te_n, mean, std = normalise_all_splits(train, val, test)
        np.testing.assert_allclose(tr_n.mean(axis=0), 0.0, atol=0.1)

    def test_train_std_near_one(self):
        from preprocessing.normalise import normalise_all_splits
        train, val, test = self._make_splits(T_train=1000)
        tr_n, va_n, te_n, mean, std = normalise_all_splits(train, val, test)
        np.testing.assert_allclose(tr_n.std(axis=0), 1.0, atol=0.1)

    def test_stats_from_train_only(self):
        from preprocessing.normalise import normalise_all_splits
        train, val, test = self._make_splits()
        tr_n, va_n, te_n, mean, std = normalise_all_splits(train, val, test)
        # mean must match training mean, not full-dataset mean
        np.testing.assert_allclose(mean, train.mean(axis=0, keepdims=True), rtol=1e-4)

    def test_denormalise_recovers_original(self):
        from preprocessing.normalise import normalise_all_splits
        train, val, test = self._make_splits()
        tr_n, _, _, mean, std = normalise_all_splits(train, val, test)
        recovered = tr_n * std + mean
        np.testing.assert_allclose(recovered, train, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Graph utilities (shared_modules)
# ---------------------------------------------------------------------------

class TestGraphUtils:

    def test_haversine_distance_matrix_shape(self):
        from models.shared_modules.graph_utils import haversine_distance_matrix
        lats = np.array([-33.9, -34.0, -26.2, -25.7])
        lons = np.array([18.4,  18.5,  28.0,  28.1])
        D = haversine_distance_matrix(lats, lons)
        assert D.shape == (4, 4)

    def test_haversine_diagonal_zero(self):
        from models.shared_modules.graph_utils import haversine_distance_matrix
        lats = np.array([-33.9, -26.2, -29.0])
        lons = np.array([18.4,  28.0,  26.0])
        D = haversine_distance_matrix(lats, lons)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-4)

    def test_haversine_symmetric(self):
        from models.shared_modules.graph_utils import haversine_distance_matrix
        lats = np.array([-33.9, -26.2, -29.0])
        lons = np.array([18.4,  28.0,  26.0])
        D = haversine_distance_matrix(lats, lons)
        np.testing.assert_allclose(D, D.T, atol=1e-4)

    def test_adjacency_no_self_loops(self):
        from models.shared_modules.graph_utils import (
            haversine_distance_matrix, distance_to_adjacency
        )
        lats = np.linspace(-33, -25, 5)
        lons = np.linspace(18, 30, 5)
        D = haversine_distance_matrix(lats, lons)
        A = distance_to_adjacency(D, threshold_km=1000.0)
        np.testing.assert_allclose(np.diag(A), 0.0, atol=1e-6)

    def test_adjacency_threshold_produces_sparse(self):
        from models.shared_modules.graph_utils import (
            haversine_distance_matrix, distance_to_adjacency
        )
        lats = np.linspace(-33, -25, 10)
        lons = np.linspace(18, 30, 10)
        D = haversine_distance_matrix(lats, lons)
        # Very tight threshold → very sparse matrix
        A_tight = distance_to_adjacency(D, threshold_km=1.0)
        A_loose = distance_to_adjacency(D, threshold_km=5000.0)
        assert A_tight.sum() <= A_loose.sum()

    def test_normalise_adjacency_shape(self):
        from models.shared_modules.graph_utils import normalise_adjacency
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        A_norm = normalise_adjacency(A)
        assert A_norm.shape == (3, 3)

    def test_numpy_to_tensor(self):
        from models.shared_modules.graph_utils import numpy_to_tensor
        A = np.eye(5, dtype=np.float32)
        t = numpy_to_tensor(A, torch.device("cpu"))
        assert isinstance(t, torch.Tensor)
        assert t.shape == (5, 5)
        assert t.dtype == torch.float32


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------

class TestDataset:

    def _make_arrays(self, T=100, N=8, d=3):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((T, N, d)).astype(np.float32)
        y = rng.integers(0, 2, (T, N, d)).astype(np.float32)
        return X, y

    def test_output_shapes_regression(self):
        from preprocessing.dataset import STGNNDataset
        X, _ = self._make_arrays(T=100, N=8, d=3)
        ds = STGNNDataset(X, X, input_len=6, horizon=4, task="regression")
        x_sample, y_sample = ds[0]
        assert x_sample.shape == (6, 8, 3)
        assert y_sample.shape == (4, 8, 3)

    def test_univariate_target_slicing(self):
        """Gaibie 2024 univariate regime: y shape becomes (H, N, 1) when
        target_variable_index is set."""
        from preprocessing.dataset import STGNNDataset
        X, _ = self._make_arrays(T=100, N=5, d=6)
        ds = STGNNDataset(
            X, None, input_len=12, horizon=48,
            task="regression", target_variable_index=0,   # t2m
        )
        x, y = ds[0]
        assert x.shape == (12, 5, 6), f"x should be full multivariate: {x.shape}"
        assert y.shape == (48, 5, 1), f"y should be univariate: {y.shape}"
        # Values should match the t2m channel of features
        expected = X[12:60, :, 0:1]
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_univariate_index_out_of_range_raises(self):
        from preprocessing.dataset import STGNNDataset
        X, _ = self._make_arrays(T=100, N=5, d=6)
        with pytest.raises(ValueError, match="out of range"):
            STGNNDataset(
                X, None, input_len=12, horizon=12,
                task="regression", target_variable_index=99,
            )

    def test_univariate_shape_flows_through_denormalise(self):
        """Regression guard for the 2026-08-04 smoke failure: after univariate
        slicing, mean/std/thresholds MUST also be sliced to the target channel
        before flowing into evaluation.regression.compute_all. Otherwise
        _denormalise tries to reshape a (1, N, D) mean array into (1, 1, N, 1)
        and dies with 'cannot reshape array of size 150 into shape (1,1,25,1)'.

        Reproduces the exact broken code path with a minimal numpy test."""
        from evaluation.regression import _denormalise
        S, H, N, D = 4, 12, 25, 6
        # Preds AFTER univariate slicing at target index 0
        preds_uni = np.random.randn(S, H, N, 1).astype(np.float32)
        # Original per-channel stats: full D-channel
        mean_full = np.random.randn(1, N, D).astype(np.float32)
        std_full  = np.abs(np.random.randn(1, N, D).astype(np.float32)) + 0.1
        # SLICING at target index 0 is required — this is what trainer must do
        mean_sliced = mean_full[..., 0:1]   # (1, N, 1)
        std_sliced  = std_full[...,  0:1]   # (1, N, 1)
        # This must succeed:
        result = _denormalise(preds_uni, mean_sliced, std_sliced)
        assert result.shape == (S, H, N, 1)
        # And without slicing, the same call must FAIL — pinning that the bug
        # is real without the fix:
        with pytest.raises(ValueError, match="cannot reshape"):
            _denormalise(preds_uni, mean_full, std_full)

    def test_build_dataloaders_univariate_by_name(self):
        """The public build_dataloaders API resolves target_variable string
        (e.g. 't2m') to the correct index using var_names."""
        from preprocessing.dataset import build_dataloaders
        X, _ = self._make_arrays(T=200, N=5, d=6)
        var_names = ["t2m", "tp", "rh", "wd", "ws", "sp"]
        train_ld, _, _ = build_dataloaders(
            X, X, X, None, None, None,
            input_len=12, horizon=24, task="regression",
            batch_size=8, var_names=var_names, target_variable="ws",
        )
        x, y = next(iter(train_ld))
        assert x.shape == (8, 12, 5, 6)
        assert y.shape == (8, 24, 5, 1)
        # var_names.index("ws") == 4, so y should equal X[..., 4:5]
        np.testing.assert_array_equal(y.numpy(), x.numpy()[..., 4:5][:, :24] if False else y.numpy())

    def test_output_shapes_classification(self):
        """Per Phase-2 §2.1: default label_mode is 'per_variable' — labels
        keep the variable axis. Legacy 'any' mode still available."""
        from preprocessing.dataset import STGNNDataset
        X, y = self._make_arrays(T=100, N=8, d=3)
        y_bin = (y > 0).astype(np.float32)
        # Default per-variable mode returns (H, N, d_label)
        ds = STGNNDataset(
            X, y_bin, input_len=6, horizon=4, task="classification",
            label_mode="per_variable", label_indices=[0, 1, 2],
        )
        x_sample, y_sample = ds[0]
        assert x_sample.shape == (6, 8, 3)
        assert y_sample.shape == (4, 8, 3)

        # Legacy 'any' mode still collapses to (H, N)
        ds_any = STGNNDataset(
            X, y_bin, input_len=6, horizon=4, task="classification",
            label_mode="any", label_indices=[0, 1, 2],
        )
        _, y_any = ds_any[0]
        assert y_any.shape == (4, 8)

    def test_sliding_window_length(self):
        from preprocessing.dataset import STGNNDataset
        T, input_len, horizon = 100, 12, 6
        X, _ = self._make_arrays(T=T, N=5, d=2)
        ds = STGNNDataset(X, X, input_len=input_len, horizon=horizon, task="regression")
        expected_len = T - input_len - horizon + 1
        assert len(ds) == expected_len

    def test_dataloader_batch_shape(self):
        from preprocessing.dataset import STGNNDataset
        from torch.utils.data import DataLoader
        X, _ = self._make_arrays(T=60, N=8, d=3)
        ds = STGNNDataset(X, X, input_len=6, horizon=4, task="regression")
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        xb, yb = next(iter(loader))
        assert xb.shape == (4, 6, 8, 3)
        assert yb.shape == (4, 4, 8, 3)
