"""
Integration tests for the derive layer across all preprocessing pipeline stages.

Every stage — Derive → Impute → Split → Label → Normalise → DataLoaders —
is exercised with d=8 data.  The full end-to-end run_pipeline is tested by
patching parse_era5_files so no ERA5 NetCDF files are required.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch

# ─── constants ─────────────────────────────────────────────────────────────────

ERA5_VARS = ["t2m", "d2m", "tp", "u10", "v10", "sp"]
D_RAW     = len(ERA5_VARS)   # 6
D_DERIVED = D_RAW + 2        # 8: +ws10, +rh

T, N       = 200, 15   # small enough to be fast; big enough for all splits
INPUT_LEN  = 12
HORIZON    = 6


# ─── shared synthetic data helpers ─────────────────────────────────────────────

def _synthetic_era5_X(seed: int = 0) -> np.ndarray:
    """Physically plausible (T, N, 6) float32 ERA5 array — no NaNs."""
    rng = np.random.default_rng(seed)
    X = np.zeros((T, N, D_RAW), dtype=np.float32)
    t2m = rng.uniform(275, 310, (T, N)).astype(np.float32)
    X[..., ERA5_VARS.index("t2m")] = t2m
    X[..., ERA5_VARS.index("d2m")] = t2m - rng.uniform(0, 20, (T, N)).astype(np.float32)
    X[..., ERA5_VARS.index("u10")] = rng.uniform(-15, 15, (T, N)).astype(np.float32)
    X[..., ERA5_VARS.index("v10")] = rng.uniform(-15, 15, (T, N)).astype(np.float32)
    X[..., ERA5_VARS.index("sp")]  = rng.uniform(95000, 102000, (T, N)).astype(np.float32)
    X[..., ERA5_VARS.index("tp")]  = rng.uniform(0, 0.01, (T, N)).astype(np.float32)
    return X


def _synthetic_coords() -> pd.DataFrame:
    return pd.DataFrame({
        "node_id": np.arange(N),
        "lat":     np.linspace(-34, -25, N, dtype=np.float32),
        "lon":     np.linspace(18,  30,  N, dtype=np.float32),
    })


def _derived():
    """Return (X_raw, X_derived, var_names_with_derived)."""
    from preprocessing.derive import derive_era5_variables
    X_raw = _synthetic_era5_X()
    X_der, names = derive_era5_variables(X_raw, list(ERA5_VARS))
    return X_raw, X_der, names


def _pipeline_config(derive: bool = True, batch_size: int = 8) -> dict:
    return {
        "dataset": "era5",
        "task": "regression",
        "data": {"input_len": INPUT_LEN, "horizon": HORIZON},
        "training": {"batch_size": batch_size},
        "preprocessing": {
            "derive_variables": derive,
            "distance_threshold_km": 5000.0,  # connect all N=15 nodes
            "sigma_km": 200.0,
        },
    }


def _mock_parse_return():
    """Mimics parse_era5_files output."""
    return (
        _synthetic_era5_X(),
        np.arange(T).astype("datetime64[h]"),
        _synthetic_coords(),
        list(ERA5_VARS),
    )


# ─── stage: impute (d=8) ───────────────────────────────────────────────────────

class TestImpute:

    def test_idw_fills_nans(self):
        from preprocessing.impute import idw_impute
        _, X_der, _ = _derived()
        rng = np.random.default_rng(1)
        X_nan = X_der.copy()
        mask = rng.random((T, N)) < 0.05
        X_nan[mask, -2] = np.nan   # ws10
        X_filled = idw_impute(X_nan, _synthetic_coords())
        assert not np.isnan(X_filled).any()

    def test_idw_shape_preserved(self):
        from preprocessing.impute import idw_impute
        _, X_der, _ = _derived()
        X_nan = X_der.copy()
        X_nan[0, 0, -1] = np.nan   # single NaN in rh
        X_filled = idw_impute(X_nan, _synthetic_coords())
        assert X_filled.shape == (T, N, D_DERIVED)

    def test_non_nan_values_unchanged(self):
        from preprocessing.impute import idw_impute
        _, X_der, _ = _derived()
        X_nan = X_der.copy()
        X_nan[0, 0, -2] = np.nan   # NaN only at [0,0,ws10]
        X_filled = idw_impute(X_nan, _synthetic_coords())
        # Every value that was NOT NaN must stay unchanged
        original_mask = ~np.isnan(X_nan)
        np.testing.assert_array_equal(X_filled[original_mask], X_nan[original_mask])


# ─── stage: split (d=8) ────────────────────────────────────────────────────────

class TestSplitD8:

    def test_split_shapes(self):
        from preprocessing.split import chronological_split
        _, X_der, _ = _derived()
        tr, va, te = chronological_split(X_der, 0.6, 0.2)
        assert tr.shape == (int(T * 0.6), N, D_DERIVED)
        assert va.shape == (int(T * 0.2), N, D_DERIVED)
        assert te.shape[-1] == D_DERIVED

    def test_no_time_overlap(self):
        from preprocessing.split import chronological_split
        _, X_der, _ = _derived()
        tr, va, te = chronological_split(X_der, 0.6, 0.2)
        assert len(tr) + len(va) + len(te) == T

    def test_feature_values_identical_after_split(self):
        """Split must be a view/copy — values must match the original array."""
        from preprocessing.split import chronological_split
        _, X_der, _ = _derived()
        tr, va, te = chronological_split(X_der, 0.6, 0.2)
        np.testing.assert_array_equal(tr, X_der[:len(tr)])
        np.testing.assert_array_equal(va, X_der[len(tr):len(tr) + len(va)])


# ─── stage: label (d=8) ────────────────────────────────────────────────────────

class TestLabelD8:

    @pytest.fixture
    def splits(self):
        from preprocessing.split import chronological_split
        _, X_der, _ = _derived()
        tr, va, te = chronological_split(X_der, 0.6, 0.2)
        return tr, va, te

    def test_threshold_shape(self, splits):
        from preprocessing.labels import label_all_splits
        tr, va, te = splits
        _, _, _, thresholds = label_all_splits(tr, va, te)
        assert thresholds.shape == (N, D_DERIVED)

    def test_label_shapes(self, splits):
        from preprocessing.labels import label_all_splits
        tr, va, te = splits
        y_tr, y_va, y_te, _ = label_all_splits(tr, va, te)
        assert y_tr.shape == (len(tr), N, D_DERIVED)
        assert y_va.shape == (len(va), N, D_DERIVED)
        assert y_te.shape == (len(te), N, D_DERIVED)

    def test_labels_binary(self, splits):
        from preprocessing.labels import label_all_splits
        tr, va, te = splits
        y_tr, y_va, y_te, _ = label_all_splits(tr, va, te)
        for name, y in [("train", y_tr), ("val", y_va), ("test", y_te)]:
            assert set(np.unique(y)).issubset({0, 1}), f"Non-binary labels in {name}"

    def test_ws10_threshold_positive(self, splits):
        """95th-percentile wind speed must be positive — ws10 ≥ 0 always."""
        from preprocessing.labels import compute_thresholds
        _, X_der, names = _derived()
        tr = X_der[:int(T * 0.6)]
        thresholds = compute_thresholds(tr)
        ws_thresh = thresholds[:, names.index("ws10")]
        assert (ws_thresh > 0).all()

    def test_rh_threshold_bounded(self, splits):
        """95th-percentile RH must remain in [0, 100]."""
        from preprocessing.labels import compute_thresholds
        _, X_der, names = _derived()
        tr = X_der[:int(T * 0.6)]
        thresholds = compute_thresholds(tr)
        rh_thresh = thresholds[:, names.index("rh")]
        assert (rh_thresh >= 0).all() and (rh_thresh <= 100).all()

    def test_threshold_from_train_only(self, splits):
        """Thresholds must match those computed from train subset only."""
        from preprocessing.labels import label_all_splits, compute_thresholds
        tr, va, te = splits
        _, _, _, thresholds = label_all_splits(tr, va, te)
        expected = compute_thresholds(tr)
        np.testing.assert_array_equal(thresholds, expected)


# ─── stage: normalise (d=8) ────────────────────────────────────────────────────

class TestNormaliseD8:

    @pytest.fixture
    def splits_and_norm(self):
        from preprocessing.split import chronological_split
        from preprocessing.normalise import normalise_all_splits
        _, X_der, _ = _derived()
        tr, va, te = chronological_split(X_der, 0.6, 0.2)
        tr_n, va_n, te_n, mean, std = normalise_all_splits(tr, va, te)
        return tr, va, te, tr_n, va_n, te_n, mean, std

    def test_mean_std_shape(self, splits_and_norm):
        *_, mean, std = splits_and_norm
        assert mean.shape == (1, N, D_DERIVED)
        assert std.shape  == (1, N, D_DERIVED)

    def test_train_mean_near_zero(self, splits_and_norm):
        tr, _, _, tr_n, *_ = splits_and_norm
        np.testing.assert_allclose(tr_n.mean(axis=0), 0.0, atol=0.15)

    def test_train_std_near_one(self, splits_and_norm):
        tr, _, _, tr_n, *_ = splits_and_norm
        np.testing.assert_allclose(tr_n.std(axis=0), 1.0, atol=0.15)

    def test_ws10_normalised_mean_near_zero(self, splits_and_norm):
        _, _, _, tr_n, _, _, _, _ = splits_and_norm
        ws_col = tr_n[..., -2]   # ws10 is second-to-last
        assert abs(ws_col.mean()) < 0.2

    def test_rh_normalised_mean_near_zero(self, splits_and_norm):
        _, _, _, tr_n, _, _, _, _ = splits_and_norm
        rh_col = tr_n[..., -1]   # rh is last
        assert abs(rh_col.mean()) < 0.2

    def test_no_nan_after_normalise(self, splits_and_norm):
        _, _, _, tr_n, va_n, te_n, _, _ = splits_and_norm
        for name, arr in [("train", tr_n), ("val", va_n), ("test", te_n)]:
            assert not np.isnan(arr).any(), f"NaN in {name} after normalise"

    def test_stats_from_train_only(self, splits_and_norm):
        tr, _, _, _, _, _, mean, std = splits_and_norm
        np.testing.assert_allclose(
            mean, tr.mean(axis=0, keepdims=True), rtol=1e-4
        )


# ─── stage: dataset / dataloader (d=8) ─────────────────────────────────────────

class TestDatasetD8:

    @pytest.fixture
    def normalised(self):
        from preprocessing.split import chronological_split
        from preprocessing.normalise import normalise_all_splits
        _, X_der, _ = _derived()
        tr, va, te = chronological_split(X_der, 0.6, 0.2)
        tr_n, va_n, te_n, _, _ = normalise_all_splits(tr, va, te)
        return tr_n, va_n, te_n

    def test_regression_batch_shape(self, normalised):
        from preprocessing.dataset import STGNNDataset
        from torch.utils.data import DataLoader
        tr_n, _, _ = normalised
        ds = STGNNDataset(tr_n, tr_n, INPUT_LEN, HORIZON, "regression")
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        xb, yb = next(iter(loader))
        assert xb.shape == (4, INPUT_LEN, N, D_DERIVED)
        assert yb.shape == (4, HORIZON,   N, D_DERIVED)

    def test_classification_label_collapses_vars(self, normalised):
        """With d=8, classification labels collapse all vars → (B, H, N)."""
        from preprocessing.dataset import STGNNDataset
        from preprocessing.labels import apply_labels, compute_thresholds
        from torch.utils.data import DataLoader
        tr_n, _, _ = normalised
        thresh = compute_thresholds(tr_n)
        y_int8 = apply_labels(tr_n, thresh)
        ds = STGNNDataset(tr_n, y_int8, INPUT_LEN, HORIZON, "classification")
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        xb, yb = next(iter(loader))
        assert xb.shape == (4, INPUT_LEN, N, D_DERIVED)
        assert yb.shape == (4, HORIZON,   N)   # d collapsed by max

    def test_no_nan_in_loader(self, normalised):
        from preprocessing.dataset import STGNNDataset
        from torch.utils.data import DataLoader
        tr_n, _, _ = normalised
        ds = STGNNDataset(tr_n, tr_n, INPUT_LEN, HORIZON, "regression")
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        for xb, yb in loader:
            assert not torch.isnan(xb).any()
            assert not torch.isnan(yb).any()

    def test_sliding_window_count(self, normalised):
        from preprocessing.dataset import STGNNDataset
        tr_n, _, _ = normalised
        ds = STGNNDataset(tr_n, tr_n, INPUT_LEN, HORIZON, "regression")
        expected = len(tr_n) - INPUT_LEN - HORIZON + 1
        assert len(ds) == expected


# ─── physical sanity before normalise ──────────────────────────────────────────

class TestPhysicalConstraints:

    def test_ws10_nonneg(self):
        _, X_der, names = _derived()
        ws = X_der[..., names.index("ws10")]
        assert (ws >= 0).all()

    def test_rh_bounded_0_100(self):
        _, X_der, names = _derived()
        rh = X_der[..., names.index("rh")]
        assert (rh >= 0).all() and (rh <= 100).all()

    def test_raw_vars_untouched(self):
        """derive must not modify the original 6 columns."""
        X_raw, X_der, _ = _derived()
        np.testing.assert_array_equal(X_raw, X_der[..., :D_RAW])

    def test_no_nan_in_derived(self):
        _, X_der, _ = _derived()
        assert not np.isnan(X_der).any()

    def test_ws10_formula_consistent(self):
        """ws10 values must equal sqrt(u10²+v10²) computed independently."""
        _, X_der, names = _derived()
        X_raw = _synthetic_era5_X()
        u = X_raw[..., ERA5_VARS.index("u10")]
        v = X_raw[..., ERA5_VARS.index("v10")]
        expected_ws = np.sqrt(u**2 + v**2)
        np.testing.assert_allclose(
            X_der[..., names.index("ws10")], expected_ws, rtol=1e-5
        )


# ─── full pipeline integration (mocked parser) ─────────────────────────────────

@pytest.fixture(scope="module")
def pipeline_out():
    """Run run_pipeline once with mocked parser. Reused across all tests below."""
    from preprocessing.pipeline import run_pipeline
    with patch("preprocessing.pipeline.parse_era5_files",
               return_value=_mock_parse_return()), \
         patch("preprocessing.pipeline.save_cache"), \
         patch("preprocessing.pipeline._write_hash"):
        result = run_pipeline(_pipeline_config(derive=True), force_rerun=True)
    return result  # (train_loader, val_loader, test_loader, adj, meta)


class TestFullPipeline:

    def test_meta_n_vars_is_8(self, pipeline_out):
        *_, meta = pipeline_out
        assert meta["n_vars"] == D_DERIVED

    def test_meta_varnames(self, pipeline_out):
        *_, meta = pipeline_out
        assert "ws10" in meta["var_names"]
        assert "rh"   in meta["var_names"]
        assert meta["var_names"] == ["t2m", "d2m", "tp", "u10", "v10", "sp", "ws10", "rh"]

    def test_train_batch_x_shape(self, pipeline_out):
        train_loader, *_ = pipeline_out
        xb, _ = next(iter(train_loader))
        assert xb.shape == torch.Size([8, INPUT_LEN, N, D_DERIVED])

    def test_train_batch_y_shape(self, pipeline_out):
        train_loader, *_ = pipeline_out
        _, yb = next(iter(train_loader))
        assert yb.shape == torch.Size([8, HORIZON, N, D_DERIVED])

    def test_val_batch_shape(self, pipeline_out):
        _, val_loader, *_ = pipeline_out
        xb, yb = next(iter(val_loader))
        assert xb.shape[-1] == D_DERIVED
        assert yb.shape[-1] == D_DERIVED

    def test_no_nan_in_all_train_batches(self, pipeline_out):
        train_loader, *_ = pipeline_out
        for xb, yb in train_loader:
            assert not torch.isnan(xb).any(), "NaN in x batch"
            assert not torch.isnan(yb).any(), "NaN in y batch"

    def test_adj_shape(self, pipeline_out):
        _, _, _, adj, _ = pipeline_out
        assert adj.shape == (N, N)

    def test_adj_nonneg(self, pipeline_out):
        _, _, _, adj, _ = pipeline_out
        assert (adj >= 0).all()

    def test_derive_off_gives_6_vars(self):
        """Disabling derivation must yield n_vars=6."""
        from preprocessing.pipeline import run_pipeline
        with patch("preprocessing.pipeline.parse_era5_files",
                   return_value=_mock_parse_return()), \
             patch("preprocessing.pipeline.save_cache"), \
             patch("preprocessing.pipeline._write_hash"):
            result = run_pipeline(_pipeline_config(derive=False), force_rerun=True)
        *_, meta = result
        assert meta["n_vars"] == D_RAW
        assert "ws10" not in meta["var_names"]
        assert "rh"   not in meta["var_names"]

    def test_classification_pipeline_label_shape(self):
        """Classification task: y batches must be (B, H, N) with d=8 input."""
        from preprocessing.pipeline import run_pipeline
        cfg = _pipeline_config(derive=True, batch_size=4)
        cfg["task"] = "classification"
        with patch("preprocessing.pipeline.parse_era5_files",
                   return_value=_mock_parse_return()), \
             patch("preprocessing.pipeline.save_cache"), \
             patch("preprocessing.pipeline._write_hash"):
            result = run_pipeline(cfg, force_rerun=True)
        train_loader, *_ = result
        xb, yb = next(iter(train_loader))
        assert xb.shape[-1] == D_DERIVED          # x still carries 8 vars
        assert len(yb.shape) == 3                 # (B, H, N) — no var dimension
        assert yb.shape == torch.Size([4, HORIZON, N])


# ─── cache hash invalidation ───────────────────────────────────────────────────

class TestHashInvalidation:

    def test_derive_true_vs_false_hash_differs(self):
        from preprocessing.pipeline import compute_data_hash
        h_on  = compute_data_hash("era5", {"input_len": 12, "horizon": 6, "derive_variables": True})
        h_off = compute_data_hash("era5", {"input_len": 12, "horizon": 6, "derive_variables": False})
        assert h_on != h_off

    def test_old_cache_hash_invalidated(self):
        """
        Caches built before derive_variables was added (key absent → None) must
        be invalidated by caches built with derive_variables=True.
        """
        from preprocessing.pipeline import compute_data_hash
        h_old = compute_data_hash("era5", {"input_len": 12, "horizon": 6})
        h_new = compute_data_hash("era5", {"input_len": 12, "horizon": 6, "derive_variables": True})
        assert h_old != h_new

    def test_same_config_same_hash(self):
        from preprocessing.pipeline import compute_data_hash
        cfg = {"input_len": 12, "horizon": 6, "derive_variables": True}
        assert compute_data_hash("era5", cfg) == compute_data_hash("era5", cfg)

    def test_hash_uses_stems_not_mtimes(self, tmp_path, monkeypatch):
        """Touching a file's mtime must NOT change the hash."""
        import time
        from preprocessing.pipeline import compute_data_hash, _manifest_fingerprint
        monkeypatch.setattr("preprocessing.pipeline.ERA5_RAW_DIR", tmp_path)
        f = tmp_path / "era5_southafrica_2010_01.nc"
        f.touch()
        h1 = compute_data_hash("era5", {"input_len": 12, "horizon": 6})
        time.sleep(0.02)
        f.touch()   # update mtime only
        h2 = compute_data_hash("era5", {"input_len": 12, "horizon": 6})
        assert h1 == h2, "Hash changed after mtime-only touch — still using mtimes"

    def test_new_month_changes_hash(self, tmp_path, monkeypatch):
        """Adding a new monthly file must invalidate the hash."""
        from preprocessing.pipeline import compute_data_hash
        monkeypatch.setattr("preprocessing.pipeline.ERA5_RAW_DIR", tmp_path)
        (tmp_path / "era5_southafrica_2010_01.nc").touch()
        h1 = compute_data_hash("era5", {"input_len": 12, "horizon": 6})
        (tmp_path / "era5_southafrica_2010_02.nc").touch()
        h2 = compute_data_hash("era5", {"input_len": 12, "horizon": 6})
        assert h1 != h2
