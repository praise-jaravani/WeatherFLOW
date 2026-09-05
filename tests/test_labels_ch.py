"""Wraparound + pos-rate distribution tests for calendar-hour thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing.labels_ch import (
    DAY_WIN,
    HOUR_WIN,
    _cell_sample_index,
    _wrapped_offsets,
    apply_ch_labels,
    compute_ch_thresholds,
    compute_ch_thresholds_nontp,
    doy_hour_of,
    label_all_splits_ch,
)


# ---------------------------------------------------------------------------
# Wraparound units
# ---------------------------------------------------------------------------

def test_wrapped_offsets_day_boundary():
    """Day 1 (doy0=0) window pulls doy0 in {359..365, 0..6}."""
    offsets = (_wrapped_offsets(7, 366) + 0) % 366
    expected = set([(i - 7) % 366 for i in range(15)])
    assert set(offsets.tolist()) == expected
    # Contains both wrap-around and near-zero days
    assert 365 in offsets
    assert 0 in offsets
    assert 6 in offsets


def test_wrapped_offsets_hour_boundary():
    """Hour 0 window with radius 1 pulls hours {23, 0, 1}."""
    offsets = (_wrapped_offsets(1, 24) + 0) % 24
    assert set(offsets.tolist()) == {23, 0, 1}
    # Hour 23 window with radius 1 pulls {22, 23, 0}
    offsets2 = (_wrapped_offsets(1, 24) + 23) % 24
    assert set(offsets2.tolist()) == {22, 23, 0}


def test_cell_sample_index_wraps_day():
    """
    A doy0=0 target with day_win=7 pulls a symmetric 15-sample window:
    doy0 in {359..365, 0..7}. Radius 7 means ±7 inclusive → 15 offsets.
    """
    doy0 = np.arange(366, dtype=np.int16)
    hour = np.full(366, 12, dtype=np.int8)
    idx = _cell_sample_index(doy0, hour, target_doy0=0, target_hour=12,
                              day_win=7, hour_win=1)
    assert len(idx) == 15
    assert set(idx.tolist()) == set(list(range(0, 8)) + list(range(359, 366)))
    assert 8 not in idx        # outside window
    assert 358 not in idx      # outside window


def test_cell_sample_index_wraps_hour():
    """A target hour=0 should pull hour=23 samples too."""
    # 24 samples at doy0=0 across all hours
    doy0 = np.zeros(24, dtype=np.int16)
    hour = np.arange(24, dtype=np.int8)
    idx = _cell_sample_index(doy0, hour, target_doy0=0, target_hour=0,
                              day_win=7, hour_win=1)
    picked = sorted(hour[idx].tolist())
    assert picked == [0, 1, 23]


def test_doy_hour_of_leap_and_non_leap():
    """doy0 numbering handles leap years correctly."""
    # 2020 (leap): Dec 31 = doy 366 → doy0 = 365
    ts_leap_end = pd.Timestamp("2020-12-31 00:00")
    d, h = doy_hour_of(np.array([ts_leap_end.to_numpy()]))
    assert d[0] == 365
    # 2021 (non-leap): Dec 31 = doy 365 → doy0 = 364
    ts_non_end = pd.Timestamp("2021-12-31 00:00")
    d, h = doy_hour_of(np.array([ts_non_end.to_numpy()]))
    assert d[0] == 364


# ---------------------------------------------------------------------------
# Threshold construction sanity
# ---------------------------------------------------------------------------

def test_ch_thresholds_shape_and_finite():
    """Synthetic 2-year hourly series produces a (N, D, 366, 24) finite tensor."""
    rng = np.random.default_rng(0)
    N, D = 4, 3
    ts = pd.date_range("2018-01-01 00:00", "2019-12-31 23:00", freq="h").to_numpy()
    X = rng.standard_normal((len(ts), N, D)).astype(np.float32)
    T, info = compute_ch_thresholds_nontp(X, ts, percentile=95.0)
    assert T.shape == (N, D, 366, 24)
    assert np.isfinite(T).all()
    # Median cell samples for a 2-year daily grid: about 2 years × 15 days × 3 hours = 90
    # (below the MIN_SAMPLES_PER_CELL of 100 — so day_win_used should have widened)
    assert info["day_win_used"] >= 7


def test_ch_labels_train_positive_rate_near_5pct():
    """
    By construction of the 95th-percentile threshold, the training positive rate
    within each cell should be ~5%. Aggregate over cells, expect median in
    [4.5%, 6.5%] (loose for synthetic noise + strict-`>` comparator).
    """
    rng = np.random.default_rng(0)
    N, D = 4, 3
    # 3 years so windowed cells have enough samples
    ts = pd.date_range("2018-01-01 00:00", "2020-12-31 23:00", freq="h").to_numpy()
    X = rng.standard_normal((len(ts), N, D)).astype(np.float32)
    T, _ = compute_ch_thresholds_nontp(X, ts, percentile=95.0)
    y = apply_ch_labels(X, ts, T)
    pos_rate = y.mean()
    assert 0.03 < pos_rate < 0.07, f"train pos rate {pos_rate:.4f} out of expected 5%±band"


def test_apply_ch_labels_shape_and_dtype():
    ts = pd.date_range("2020-01-01", periods=48, freq="h").to_numpy()
    N, D = 2, 3
    X = np.zeros((48, N, D), dtype=np.float32)
    T = np.ones((N, D, 366, 24), dtype=np.float32) * 0.5
    y = apply_ch_labels(X, ts, T)
    assert y.shape == (48, N, D)
    assert y.dtype == np.int8
    # All values < threshold → all zeros
    assert y.sum() == 0


def test_label_all_splits_ch_orchestrator():
    """End-to-end orchestrator returns the expected 6-tuple with right shapes."""
    rng = np.random.default_rng(0)
    N, D = 3, 2
    ts_train = pd.date_range("2018-01-01", "2019-12-31 23:00", freq="h").to_numpy()
    ts_val   = pd.date_range("2020-01-01", "2020-06-30 23:00", freq="h").to_numpy()
    ts_test  = pd.date_range("2020-07-01", "2020-12-31 23:00", freq="h").to_numpy()
    X_train = rng.standard_normal((len(ts_train), N, D)).astype(np.float32)
    X_val   = rng.standard_normal((len(ts_val),   N, D)).astype(np.float32)
    X_test  = rng.standard_normal((len(ts_test),  N, D)).astype(np.float32)
    y_tr, y_va, y_te, T, rung, info = label_all_splits_ch(
        X_train, X_val, X_test, ts_train, ts_val, ts_test,
        var_names=["t2m", "rh"],
    )
    assert y_tr.shape == (len(ts_train), N, D)
    assert y_va.shape == (len(ts_val),   N, D)
    assert y_te.shape == (len(ts_test),  N, D)
    assert T.shape == (N, D, 366, 24)
    # rung_map returns -1 when tp absent
    assert (rung == -1).all()
    assert info["scheme"] == "calendar_hour_relative"
    assert info["percentile"] == 95.0
