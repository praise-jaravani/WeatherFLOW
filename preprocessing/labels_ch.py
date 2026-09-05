"""
Calendar-day / hour-of-day windowed thresholds for Phase-2 classification.

Motivation
----------
Static per-(station, variable) 95th-percentile thresholds concentrate positives
in climatological peak season/hours (t2m → summer afternoons; sp → winter fronts).
A model then scores AUC by learning the calendar, not the weather — a well-
documented failure mode in ML-heatwave and marine-heatwave forecasting.

Fix (ETCCDI-family, extended to hourly): the threshold for a value at
(station s, variable v, timestamp t) is the 95th percentile of training values
of v at s among timestamps whose (day-of-year, hour-of-day) fall within
`(doy(t) ± DAY_WIN, hour(t) ± HOUR_WIN)` — wrapping at both boundaries
(day 366 → day 1; hour 23 → hour 0).

Cache shape: `thresholds_ch[N_stations, N_vars, 366, 24]` float32 (~2.4 MB
for N=32, D=6). Labels have the same shape as static-scheme labels but are
computed by indexing the (doy, hour) cell for each timestamp.

tp wet-hour rule (§2.5 of Phase-2 plan, extended)
-------------------------------------------------
Precipitation is 90%+ dry at hourly resolution. A cell's 95th percentile over
all training samples is trivially "any rain". So for tp, we compute the
percentile over WET hours only (tp ≥ TP_WET_THRESH_M), WITHIN each cell.
Sparse cells fall back through a ladder:

    rung 0: cell wet-percentile (day±DAY_WIN, hour±HOUR_WIN) if wet_count ≥ TP_MIN_WET
    rung 1: widen day window to ±DAY_WIN_WIDE
    rung 2: pool all hours (doy±DAY_WIN_WIDE only)
    rung 3: season-level wet-percentile (3-month bucket)
    rung 4: fallback to global static wet-hour threshold

Every cell's rung is logged and returned as a `(N, 366, 24)` int8 map, and a
per-variable histogram of rung usage is emitted for the report.

Comparator: strictly `>`. Matches static scheme (`preprocessing/labels.py:116`).

Wrap semantics
--------------
Day-of-year is 1..366. On non-leap years, day 366 does not occur in
`timestamps`, so no sample is stored at index 365 (0-based) for those years —
but the WINDOW around a target day still wraps into that region, pulling from
leap-year day-366 samples when they exist. This matches the ETCCDI convention
of pooling across years and lets the seasonal cycle close.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Window widths — SWEEP-SELECTED (2026-08-13).
#
# Pre-registered rule: candidates ordered lexicographically by
# (hour_window, day_window) ascending; winner = first candidate satisfying
# (a) median non-tp cell count >= 200 AND (b) max non-tp climatology-
# opponent PR-AUC / prevalence <= 1.15 on real data. Grid searched:
# hour ∈ {0, 1, 2}, day ∈ {3, 7, 10, 15}.
#
# Winner: hour_window=1, day_window=7 — first candidate to clear the 200-
# cell floor (270 cells) and the 1.15× real-deseasonalisation bound (0.933×).
# Real max ratio stayed in [0.90, 1.16] across 8 of 12 candidates, so the
# CH scheme is not sensitive to window choice within the swept range.
# Physical justification: seasonal cycle is slow (±7d spans little drift),
# diurnal cycle is sharp (±1h is at the edge of defensible smearing).
# Following the ETCCDI convention widened on the day axis for our shorter
# 6-year baseline (ETCCDI uses 30-year baselines × 5 days = 150 samples;
# we need 6 years × 15 days × 3 hours = 270 to clear the ~200 floor).
#
# See:
#   - results/diagnostics/phase2_labels/window_sweep.md   (full appendix table)
#   - docs/PHASE2_FORMULATION.md § 2.1.1                  (selection prose)
#   - results/diagnostics/phase2_labels/REPORT.md § 7.5   (sweep verdict)
DAY_WIN: int = 7                     # ± days around target doy (window = 15 days)
DAY_WIN_WIDE: int = 15               # widening for sparse tp cells (window = 31)
HOUR_WIN: int = 1                    # ± hours around target hour (window = 3 hours)
PERCENTILE: float = 95.0
MIN_SAMPLES_PER_CELL: int = 100      # non-tp; expected ≈ 3 × 15 × 6 ≈ 270

# tp-specific — carried over from static scheme
from preprocessing.labels import (   # noqa: E402
    TP_WET_MM_PER_H,
    TP_WET_THRESH_M,
    TP_MIN_WET_HOURS,
)

TP_MIN_WET_PER_CELL: int = 30        # min wet-hour count for rung-0 acceptance

# Rung codes for the fallback ladder (also written into meta)
RUNG_CELL         = 0                # doy ±7,  hour ±1  (default)
RUNG_WIDE_DAY     = 1                # doy ±15, hour ±1
RUNG_HOUR_POOLED  = 2                # doy ±15, all hours
RUNG_SEASONAL     = 3                # 3-month season bucket
RUNG_GLOBAL_WET   = 4                # static wet-hour threshold from labels.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def doy_hour_of(timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert datetime64 array → (doy-1, hour) integer arrays.

    doy is 1..366 in pandas; we return 0..365 for direct array indexing.

    Args:
        timestamps: (T,) datetime64[ns] hourly array.

    Returns:
        doy0: (T,) int16, values in [0, 365]
        hour: (T,) int8,  values in [0, 23]
    """
    ts = pd.DatetimeIndex(timestamps)
    doy0 = (ts.dayofyear.to_numpy() - 1).astype(np.int16)
    hour = ts.hour.to_numpy().astype(np.int8)
    return doy0, hour


def _wrapped_offsets(radius: int, size: int) -> np.ndarray:
    """
    Return an int array of relative offsets in [-radius, +radius] used for
    modular indexing. Caller is responsible for applying `% size`.
    """
    return np.arange(-radius, radius + 1, dtype=np.int16)


def _cell_sample_index(
    doy0_train: np.ndarray,
    hour_train: np.ndarray,
    target_doy0: int,
    target_hour: int,
    day_win: int,
    hour_win: int,
    pool_hours: bool = False,
) -> np.ndarray:
    """
    Return training-sample indices falling within the wrapped window around
    (target_doy0, target_hour).

    Args:
        doy0_train, hour_train: (T,) per-training-sample doy0/hour.
        target_doy0: int in [0, 365].
        target_hour: int in [0, 23].
        day_win, hour_win: window radii.
        pool_hours: if True, ignore hour_win and accept any hour.

    Returns:
        indices: (K,) int64 into training arrays.
    """
    day_offsets  = (_wrapped_offsets(day_win, 366) + target_doy0) % 366
    day_mask     = np.isin(doy0_train, day_offsets)
    if pool_hours:
        return np.where(day_mask)[0]
    hour_offsets = (_wrapped_offsets(hour_win, 24)  + target_hour) % 24
    hour_mask    = np.isin(hour_train, hour_offsets)
    return np.where(day_mask & hour_mask)[0]


def _season_of(doy0: int) -> int:
    """DJF=0, MAM=1, JJA=2, SON=3 (Southern Hemisphere calendar convention)."""
    m = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(doy0))  # 2020 has 366 days
    return (m.month % 12) // 3


# ---------------------------------------------------------------------------
# Non-tp threshold builder
# ---------------------------------------------------------------------------

def _cell_indices_all(doy0: np.ndarray, hour: np.ndarray,
                       day_win: int, hour_win: int,
                       pool_hours: bool = False) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Precompute per-cell wrapped-window index arrays.

    Runs in O(366 * 24 * T) but that's ~460M elementwise mask ops for a
    6-year train split — 5–10s on numpy vectorised ops (vs pure Python).
    Reused by both the non-tp threshold builder and the tp fallback ladder.
    """
    idx_map: Dict[Tuple[int, int], np.ndarray] = {}
    for d in range(366):
        day_offsets = (_wrapped_offsets(day_win, 366) + d) % 366
        day_mask    = np.isin(doy0, day_offsets)
        if pool_hours:
            common_idx = np.where(day_mask)[0]
            for h in range(24):
                idx_map[(d, h)] = common_idx        # identical across h when pooling
            continue
        for h in range(24):
            hour_offsets = (_wrapped_offsets(hour_win, 24) + h) % 24
            hour_mask    = np.isin(hour, hour_offsets)
            idx_map[(d, h)] = np.where(day_mask & hour_mask)[0]
    return idx_map


def compute_ch_thresholds_nontp(
    train_data: np.ndarray,
    timestamps_train: np.ndarray,
    percentile: float = PERCENTILE,
    day_win: int = DAY_WIN,
    hour_win: int = HOUR_WIN,
    min_samples: int = MIN_SAMPLES_PER_CELL,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Compute the (N, D, 366, 24) threshold surface for non-tp variables.

    Args:
        train_data:       (T_train, N, D) physical-units training tensor.
        timestamps_train: (T_train,) datetime64[ns] matching train_data.
        percentile:       percentile per cell (default 95).
        day_win, hour_win: window radii.
        min_samples:      minimum training samples for a valid cell; if any
                          cell has fewer, we widen day window globally and try
                          again. Records the deviation.

    Returns:
        T:      (N, D, 366, 24) float32 threshold surface.
        info:   dict with 'day_win_used', 'min_cell_samples', 'median_cell_samples'.
    """
    T_train, N, D = train_data.shape
    doy0, hour = doy_hour_of(timestamps_train)

    used_day_win = day_win
    while True:
        idx_map = _cell_indices_all(doy0, hour, used_day_win, hour_win)
        cell_counts = np.array([[len(idx_map[(d, h)]) for h in range(24)] for d in range(366)],
                                dtype=np.int32)
        if cell_counts.min() >= min_samples or used_day_win >= 30:
            break
        used_day_win += 3
        if verbose:
            print(f"    [nontp] widening day_win to {used_day_win}")

    T = np.zeros((N, D, 366, 24), dtype=np.float32)
    for d in range(366):
        for h in range(24):
            idx = idx_map[(d, h)]
            if idx.size == 0:
                continue
            T[:, :, d, h] = np.percentile(train_data[idx], percentile, axis=0).astype(np.float32)

    info = {
        "day_win_used":         int(used_day_win),
        "min_cell_samples":     int(cell_counts.min()),
        "median_cell_samples":  int(np.median(cell_counts)),
    }
    return T, info


# ---------------------------------------------------------------------------
# tp threshold builder (wet-hour with fallback ladder)
# ---------------------------------------------------------------------------

def compute_ch_thresholds_tp(
    train_data: np.ndarray,
    timestamps_train: np.ndarray,
    var_names: List[str],
    percentile: float = PERCENTILE,
    day_win: int = DAY_WIN,
    hour_win: int = HOUR_WIN,
    day_win_wide: int = DAY_WIN_WIDE,
    min_wet: int = TP_MIN_WET_PER_CELL,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Compute the (N, 366, 24) tp threshold surface using the wet-hour rule with
    a fallback ladder for sparse cells.

    Args:
        train_data:       (T_train, N, D)
        timestamps_train: (T_train,)
        var_names:        list of length D; must contain 'tp' or this returns
                          empty tensors + a note in the rung dict.
        percentile, day_win, hour_win, day_win_wide, min_wet: as documented in
        module docstring.

    Returns:
        tp_thresh:  (N, 366, 24) float32, threshold per cell for tp.
        rung_map:   (N, 366, 24) int8, which rung was used for each cell.
        rung_hist:  {rung: count} across all cells (all stations flattened).
    """
    if "tp" not in var_names:
        # Return empty tensors with the right shapes so callers can concatenate
        N = train_data.shape[1]
        return (np.zeros((N, 366, 24), dtype=np.float32),
                np.full((N, 366, 24), -1, dtype=np.int8),
                {})

    tp_idx = var_names.index("tp")
    tp_all = train_data[..., tp_idx]                         # (T_train, N)
    T_train, N = tp_all.shape
    doy0, hour = doy_hour_of(timestamps_train)
    is_wet = tp_all >= TP_WET_THRESH_M                       # (T_train, N)

    # Global wet-hour static threshold per station — the rung-4 fallback.
    global_wet = np.full(N, np.nan, dtype=np.float32)
    for n in range(N):
        wet_vals = tp_all[is_wet[:, n], n]
        if wet_vals.size >= TP_MIN_WET_HOURS:
            global_wet[n] = np.percentile(wet_vals, percentile)

    # ---- Precompute all cell index arrays (shared across rungs 0-2) ----
    idx_cell_all = _cell_indices_all(doy0, hour, day_win, hour_win)
    idx_wide_all = _cell_indices_all(doy0, hour, day_win_wide, hour_win)
    idx_pool_all = _cell_indices_all(doy0, hour, day_win_wide, hour_win,
                                       pool_hours=True)
    # ---- Precompute season-of-doy vector + per-(season, station) wet vectors ----
    doy_season = np.array([_season_of(int(d)) for d in doy0], dtype=np.int8)   # (T_train,)
    # season_wet_vals[season][n] → np.ndarray of wet-hour values
    season_wet_vals = [
        [tp_all[(doy_season == s) & is_wet[:, n], n] for n in range(N)]
        for s in range(4)
    ]
    season_wet_pct = np.full((4, N), np.nan, dtype=np.float32)
    for s in range(4):
        for n in range(N):
            v = season_wet_vals[s][n]
            if v.size >= min_wet:
                season_wet_pct[s, n] = np.percentile(v, percentile)

    tp_thresh = np.zeros((N, 366, 24), dtype=np.float32)
    rung_map  = np.full((N, 366, 24), -1, dtype=np.int8)

    for d in range(366):
        season = _season_of(d)
        for h in range(24):
            idx_cell = idx_cell_all[(d, h)]
            idx_wide = idx_wide_all[(d, h)]
            idx_pool = idx_pool_all[(d, h)]
            # slice tp values ONCE per cell — reuse across stations
            tp_cell = tp_all[idx_cell]   # (K_cell, N)
            tp_wide = tp_all[idx_wide]   # (K_wide, N)
            tp_pool = tp_all[idx_pool]   # (K_pool, N)
            for n in range(N):
                # rung 0
                v = tp_cell[:, n];  v = v[v >= TP_WET_THRESH_M]
                if v.size >= min_wet:
                    tp_thresh[n, d, h] = np.percentile(v, percentile)
                    rung_map[n, d, h]  = RUNG_CELL
                    continue
                # rung 1
                v = tp_wide[:, n];  v = v[v >= TP_WET_THRESH_M]
                if v.size >= min_wet:
                    tp_thresh[n, d, h] = np.percentile(v, percentile)
                    rung_map[n, d, h]  = RUNG_WIDE_DAY
                    continue
                # rung 2
                v = tp_pool[:, n];  v = v[v >= TP_WET_THRESH_M]
                if v.size >= min_wet:
                    tp_thresh[n, d, h] = np.percentile(v, percentile)
                    rung_map[n, d, h]  = RUNG_HOUR_POOLED
                    continue
                # rung 3 — precomputed
                if np.isfinite(season_wet_pct[season, n]):
                    tp_thresh[n, d, h] = season_wet_pct[season, n]
                    rung_map[n, d, h]  = RUNG_SEASONAL
                    continue
                # rung 4 — global wet or wet cutoff
                if np.isfinite(global_wet[n]):
                    tp_thresh[n, d, h] = global_wet[n]
                else:
                    tp_thresh[n, d, h] = TP_WET_THRESH_M
                rung_map[n, d, h] = RUNG_GLOBAL_WET

    # Rung histogram (all cells flattened)
    rung_hist: Dict[int, int] = {}
    for r in (RUNG_CELL, RUNG_WIDE_DAY, RUNG_HOUR_POOLED, RUNG_SEASONAL, RUNG_GLOBAL_WET):
        rung_hist[r] = int((rung_map == r).sum())

    return tp_thresh, rung_map, rung_hist


# ---------------------------------------------------------------------------
# Combined threshold builder
# ---------------------------------------------------------------------------

def compute_ch_thresholds(
    train_data: np.ndarray,
    timestamps_train: np.ndarray,
    var_names: List[str],
    percentile: float = PERCENTILE,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute the (N, D, 366, 24) threshold surface for all variables, applying
    the tp wet-hour rule (with fallback ladder) where applicable.

    Returns:
        T:         (N, D, 366, 24) float32.
        rung_map:  (N, 366, 24) int8, rung used per cell for tp (or -1 if no tp).
        info:      dict — nontp_day_win, min_cell_samples, rung_hist, etc.
    """
    T_nontp, info_nontp = compute_ch_thresholds_nontp(
        train_data, timestamps_train, percentile=percentile,
    )
    tp_thresh, rung_map, rung_hist = compute_ch_thresholds_tp(
        train_data, timestamps_train, var_names, percentile=percentile,
    )
    T = T_nontp.copy()
    if "tp" in var_names:
        tp_idx = var_names.index("tp")
        T[:, tp_idx, :, :] = tp_thresh

    info = {
        **info_nontp,
        "rung_hist": rung_hist,
        "percentile": float(percentile),
        "day_win": DAY_WIN,
        "hour_win": HOUR_WIN,
        "day_win_wide": DAY_WIN_WIDE,
        "min_samples_per_cell": MIN_SAMPLES_PER_CELL,
        "tp_min_wet_per_cell": TP_MIN_WET_PER_CELL,
        "tp_wet_thresh_m": TP_WET_THRESH_M,
        "comparator": ">",
        "scheme": "calendar_hour_relative",
        "scheme_version": "1.0",
    }
    return T, rung_map, info


# ---------------------------------------------------------------------------
# Label application
# ---------------------------------------------------------------------------

def apply_ch_labels(
    data: np.ndarray,
    timestamps: np.ndarray,
    thresholds_ch: np.ndarray,
) -> np.ndarray:
    """
    Produce binary labels for `data` by indexing the (doy, hour) cell of
    `thresholds_ch` at each timestamp.

    Args:
        data:          (T, N, D) physical-units array.
        timestamps:    (T,) datetime64[ns] matching data.
        thresholds_ch: (N, D, 366, 24) float32 threshold surface.

    Returns:
        labels: (T, N, D) int8. Same comparator as static scheme (strict `>`).
    """
    doy0, hour = doy_hour_of(timestamps)
    # T[n, v, doy, h] — we need T[doy_of(t), h_of(t)] for each t, broadcast over (N, D).
    # Rearrange to (366, 24, N, D) then index.
    T_indexed = np.transpose(thresholds_ch, (2, 3, 0, 1))    # (366, 24, N, D)
    thresholds_for_data = T_indexed[doy0, hour]              # (T, N, D)
    return (data > thresholds_for_data).astype(np.int8)


# ---------------------------------------------------------------------------
# Orchestrator — analogue of preprocessing.labels.label_all_splits
# ---------------------------------------------------------------------------

def label_all_splits_ch(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    timestamps_train: np.ndarray,
    timestamps_val: np.ndarray,
    timestamps_test: np.ndarray,
    var_names: List[str],
    percentile: float = PERCENTILE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    End-to-end: fit thresholds on train, label all splits, return artefacts.

    Returns:
        y_train, y_val, y_test: (T_*, N, D) int8
        thresholds_ch:          (N, D, 366, 24) float32
        rung_map:                (N, 366, 24) int8 (tp rungs; -1 if no tp)
        info:                   dict for meta.json
    """
    thresholds_ch, rung_map, info = compute_ch_thresholds(
        train, timestamps_train, var_names, percentile=percentile,
    )
    y_train = apply_ch_labels(train, timestamps_train, thresholds_ch)
    y_val   = apply_ch_labels(val,   timestamps_val,   thresholds_ch)
    y_test  = apply_ch_labels(test,  timestamps_test,  thresholds_ch)
    return y_train, y_val, y_test, thresholds_ch, rung_map, info
