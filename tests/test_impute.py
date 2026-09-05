"""
Regression tests for preprocessing.impute.

Covers the two things that failed silently in the SAWS 2020-12-31 outage:
  1. Spatial IDW leaves NaN when all k neighbours are simultaneously missing.
  2. There was no temporal-fill fallback, so residual NaN reached the model
     evaluation stage and poisoned per_station_var_mae (which uses .mean()
     rather than nanmean, propagating NaN across the entire (station, variable)
     column).

The fix (impute.py) adds a ffill+bfill pass per (station, variable) after
spatial IDW. These tests pin that behaviour.
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing.impute import exclude_sparse_nodes, idw_impute


def _linear_coords(n_nodes: int) -> pd.DataFrame:
    """n stations on a 1° lat line at fixed longitude — deterministic distances."""
    return pd.DataFrame({
        "node_id": range(n_nodes),
        "lat": np.linspace(-33.0, -33.0 + (n_nodes - 1) * 1.0, n_nodes),
        "lon": np.full(n_nodes, 18.0),
    })


class TestExcludeSparseNodes:

    def test_no_exclusion_when_all_clean(self):
        X = np.ones((100, 5, 3), dtype=np.float32)
        coords = _linear_coords(5)
        X2, c2 = exclude_sparse_nodes(X, coords, exclusion_threshold=0.05)
        assert X2.shape == X.shape
        assert len(c2) == 5

    def test_drops_high_missing_node(self):
        X = np.ones((100, 5, 3), dtype=np.float32)
        X[:, 2, :] = np.nan   # station 2 fully missing
        coords = _linear_coords(5)
        X2, c2 = exclude_sparse_nodes(X, coords, exclusion_threshold=0.05)
        assert X2.shape == (100, 4, 3)
        assert 2 not in c2["node_id"].values
        assert len(c2) == 4

    def test_at_threshold_boundary(self):
        # Station 1: exactly 5% missing → kept at threshold=0.05
        X = np.ones((100, 3, 2), dtype=np.float32)
        X[:5, 1, 0] = np.nan
        coords = _linear_coords(3)
        X2, c2 = exclude_sparse_nodes(X, coords, exclusion_threshold=0.05)
        assert 1 in c2["node_id"].values


class TestIDWImpute:

    def test_no_op_when_no_nan(self):
        X = np.arange(60, dtype=np.float32).reshape(20, 3, 1)
        coords = _linear_coords(3)
        out = idw_impute(X.copy(), coords)
        np.testing.assert_array_equal(out, X)

    def test_spatial_fill_uses_neighbours(self):
        """Single missing cell — should be filled by weighted mean of neighbours."""
        X = np.array([[[10.0], [20.0], [30.0]]], dtype=np.float32)  # T=1, N=3, d=1
        X[0, 1, 0] = np.nan
        coords = _linear_coords(3)
        out = idw_impute(X.copy(), coords)
        assert not np.isnan(out).any()
        # Station 1 sits between stations 0 (10) and 2 (30) at equal distance,
        # so IDW = (10 + 30) / 2 = 20
        assert out[0, 1, 0] == pytest.approx(20.0, abs=0.5)

    def test_temporal_fill_activates_when_all_neighbours_missing(self):
        """SAWS 2020-12-31 regression: at an hour where ALL k neighbours share
        the outage, spatial IDW cannot fill. Temporal ffill/bfill must kick in
        so no NaN escapes into the downstream tensor.
        """
        T, N, d = 100, 5, 2
        X = np.ones((T, N, d), dtype=np.float32) * 10.0
        # Emulate a 3-hour regional outage: ALL 5 stations, ALL variables,
        # NaN at hours 50, 51, 52. Adjacent hours (49, 53) hold value 10.
        X[50:53, :, :] = np.nan
        coords = _linear_coords(N)
        out = idw_impute(X.copy(), coords, k_neighbours=5)
        assert not np.isnan(out).any(), (
            "temporal-fill fallback must eliminate residual NaN when spatial "
            "IDW cannot help (all neighbours simultaneously missing)"
        )
        # Since neighbours in time are 10.0 on both sides, ffill inherits from
        # hour 49 → all 3 outage hours become 10.0.
        np.testing.assert_array_equal(out[50:53, :, :], np.ones((3, N, d)) * 10.0)

    def test_temporal_fallback_bfill_at_leading_edge(self):
        """If the outage is at t=0..N (no earlier hour to ffill from), bfill
        must inherit from the first valid hour instead. Otherwise the head
        of the series would remain NaN."""
        T, N, d = 100, 5, 1
        X = np.ones((T, N, d), dtype=np.float32) * 7.5
        X[:10, :, :] = np.nan   # first 10 hours entirely missing
        coords = _linear_coords(N)
        out = idw_impute(X.copy(), coords, k_neighbours=5)
        assert not np.isnan(out).any()
        # bfill inherits hour 10's value (7.5)
        np.testing.assert_allclose(out[:10, :, :], 7.5)

    def test_temporal_fallback_ffill_at_trailing_edge(self):
        """Mirror of the leading-edge case — this is literally the SAWS bug."""
        T, N, d = 100, 5, 1
        X = np.ones((T, N, d), dtype=np.float32) * 3.14
        X[-23:, :, :] = np.nan   # 23-hour tail outage (SAWS 2020-12-31 pattern)
        coords = _linear_coords(N)
        out = idw_impute(X.copy(), coords, k_neighbours=5)
        assert not np.isnan(out).any()
        np.testing.assert_allclose(out[-23:, :, :], 3.14)

    def test_spatial_still_preferred_over_temporal(self):
        """If only ONE station is missing at a given hour, spatial IDW must
        still handle it (using the other stations' values at the same hour),
        not fall through to temporal fill."""
        T, N, d = 100, 5, 1
        X = np.zeros((T, N, d), dtype=np.float32)
        # station 0 has constant 100 over time; others have constant 0
        X[:, 0, 0] = 100.0
        # Missing at hour 50 for station 0 only — 4 spatial neighbours all have 0
        X[50, 0, 0] = np.nan
        # AND temporally adjacent hours for station 0 are 100
        coords = _linear_coords(N)
        out = idw_impute(X.copy(), coords, k_neighbours=4)
        # Spatial IDW fills with weighted mean of neighbours (all 0) → ~0
        # If temporal ffill had run first, out[50,0,0] would be 100 (wrong).
        assert out[50, 0, 0] == pytest.approx(0.0, abs=1e-3), (
            "spatial IDW must run before temporal fallback; the temporal path "
            "should only handle cells IDW couldn't fill"
        )

    def test_assertion_when_entire_column_nan(self):
        """If exclusion didn't run and a (station, variable) column is 100%
        NaN, the temporal fallback cannot help — assertion must trip loudly
        rather than silently returning NaN."""
        T, N, d = 20, 3, 2
        X = np.ones((T, N, d), dtype=np.float32)
        X[:, 1, 0] = np.nan   # station 1, variable 0 completely missing
        coords = _linear_coords(N)
        # k=1 forces IDW to try to use ONE neighbour; both other stations have
        # data for variable 0, so IDW WILL fill this. Instead flag a case
        # where ALL stations are NaN for a variable simultaneously.
        X = np.ones((T, N, d), dtype=np.float32)
        X[:, :, 1] = np.nan    # variable 1 entirely missing across all stations
        # No spatial fill possible, no temporal fill possible → assertion trips.
        with pytest.raises(AssertionError, match="temporal fallback left"):
            idw_impute(X.copy(), coords)
