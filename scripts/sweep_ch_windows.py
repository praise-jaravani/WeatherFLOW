#!/usr/bin/env python3
"""
Window-size sweep for Update-1 (calendar/hour-relative thresholds).

For each candidate (hour_window, day_window) in the grid:
    hour_window in {0, 1, 2},  day_window in {3, 7, 10, 15}
compute:
    - median non-tp cell sample count (stability floor)
    - non-tp threshold-surface variance (fidelity proxy — collapses toward
      zero as window widens; the seasonal cycle amplitude sets an upper limit)
    - non-tp adjacent-cell threshold MAD along both axes (smoothness /
      noise-vs-signal proxy — small = smooth = real climate; large = ragged
      = sampling noise)
    - seasonal-climatology opponent PR-AUC / prevalence per label variable
      on REAL data (t2m, tp, rh, ws), macro across non-tp
    - the same on SYNTHETIC data (pure seasonal + diurnal + Gaussian white
      noise, no weather signal). This is the estimation-noise floor —
      residual opponent skill on real data should be interpreted as
      (real − synthetic).

Pre-registered selection rule (per 2026-08-13 reviewer call, reproduced
verbatim in the report):

    Candidates ordered lexicographically by (hour_window, day_window)
    ascending; winner = first candidate with
        median non-tp cell count >= MIN_CELL_SAMPLES  (=200)
      AND
        max non-tp climatology-opponent PR-AUC/prevalence <= REAL_MAX_RATIO
                                                              (=1.15)
      on REAL data. Winners at day_window > 10 are flagged for a physics
      review before acceptance. All 12 candidates' cell counts, real ratios,
      and synthetic ratios are tabulated in the appendix.

    Synthetic control: computed for all candidates as a calibration
    reference, NOT a selection criterion; interpretation = real ratio minus
    synthetic excess; hard stop only if the SELECTED window's synthetic
    ratio > SYNTH_BUG_TRIPWIRE (=1.15) — that's an implementation defect,
    not a window property. Investigate before proceeding.

Output:
    results/diagnostics/phase2_labels/window_sweep.json    (full table)
    results/diagnostics/phase2_labels/window_sweep.md      (human-readable)

This is a diagnostic. It does NOT rebuild the production cache; the
follow-up (if the winning window is not the current (day=7, hour=1)) is a
separate action.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.labels_ch import (             # noqa: E402
    _cell_indices_all,
    apply_ch_labels,
    doy_hour_of,
)
from preprocessing.saws_parser import WINDOW_START, WINDOW_END   # noqa: E402
from evaluation.classification import compute_all, select_thresholds  # noqa: E402


# ---------------------------------------------------------------------------
# Selection rule (pre-registered, do NOT modify without re-registering)
# ---------------------------------------------------------------------------

DAY_WINDOWS  = [3, 7, 10, 15]
HOUR_WINDOWS = [0, 1, 2]
MIN_CELL_SAMPLES    = 200
REAL_MAX_RATIO      = 1.15
SYNTH_BUG_TRIPWIRE  = 1.15
DAY_PHYSICS_FLAG    = 10

LABEL_VARS = ["t2m", "tp", "rh", "ws"]      # Phase-2 §2.1
NONTP_VARS = ["t2m", "rh", "ws"]

INPUT_LEN = 12
HORIZON   = 48
PERCENTILE = 95.0


# ---------------------------------------------------------------------------
# Threshold / label / opponent helpers (windowed, non-tp only)
# ---------------------------------------------------------------------------

def _build_nontp_thresholds(train_data: np.ndarray,
                             ts_train: np.ndarray,
                             day_win: int, hour_win: int,
                             percentile: float = PERCENTILE):
    """
    (N, D_nontp, 366, 24) float32 threshold surface for non-tp variables.
    Returns thresholds + cell-count array + info dict.
    """
    T_train, N, D = train_data.shape
    doy0, hour = doy_hour_of(ts_train)
    idx_map = _cell_indices_all(doy0, hour, day_win, hour_win)
    cell_counts = np.array([[len(idx_map[(d, h)]) for h in range(24)] for d in range(366)],
                            dtype=np.int32)

    T_out = np.zeros((N, D, 366, 24), dtype=np.float32)
    for d in range(366):
        for h in range(24):
            idx = idx_map[(d, h)]
            if idx.size == 0:
                continue
            T_out[:, :, d, h] = np.percentile(train_data[idx], percentile, axis=0).astype(np.float32)

    return T_out, cell_counts


def _fidelity_stats(T: np.ndarray) -> Dict[str, float]:
    """
    T: (N, D, 366, 24) — threshold surface.
    Returns:
        variance_over_calendar: mean over (N, D) of std_across_(doy,hour)(T)
        adj_day_mad:  mean |T[.,.,d+1,h] - T[.,.,d,h]| over doy axis, wrapped
        adj_hour_mad: mean |T[.,.,d,h+1] - T[.,.,d,h]| over hour axis, wrapped
    """
    # Std over calendar axes → measures how much the threshold varies across
    # the year × day. Collapses as window widens.
    std_calendar = T.reshape(*T.shape[:2], -1).std(axis=-1).mean()
    # Adjacent-cell MAD (roll gives wraparound comparison)
    dday_mad  = np.abs(np.roll(T, -1, axis=2) - T).mean()
    dhour_mad = np.abs(np.roll(T, -1, axis=3) - T).mean()
    return {
        "variance_over_calendar_axes": float(std_calendar),
        "adjacent_day_mad":           float(dday_mad),
        "adjacent_hour_mad":          float(dhour_mad),
    }


def _apply_labels_nontp(data: np.ndarray, ts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Strictly `>` per (doy, hour) cell. T: (N, D, 366, 24). data: (T, N, D)."""
    doy0, hour = doy_hour_of(ts)
    Tperm = T.transpose(2, 3, 0, 1)                # (366, 24, N, D)
    return (data > Tperm[doy0, hour]).astype(np.int8)


def _windowed(y: np.ndarray, ts: np.ndarray, input_len: int, horizon: int):
    T = y.shape[0]
    n_samples = T - input_len - horizon + 1
    starts = np.arange(n_samples)
    tidx = starts[:, None] + input_len + np.arange(horizon)[None, :]
    return y[tidx], ts[tidx]


def _build_freq_map(y_train: np.ndarray, ts_train: np.ndarray,
                     day_win: int, hour_win: int) -> np.ndarray:
    """Cell exceedance frequency: (N, D, 366, 24). Uses same window as thresholds."""
    T_train, N, D = y_train.shape
    doy0, hour = doy_hour_of(ts_train)
    idx_map = _cell_indices_all(doy0, hour, day_win, hour_win)
    freq = np.zeros((N, D, 366, 24), dtype=np.float32)
    for d in range(366):
        for h in range(24):
            idx = idx_map[(d, h)]
            if idx.size == 0:
                continue
            freq[:, :, d, h] = y_train[idx].astype(np.float32).mean(axis=0)
    return freq


def _climatology_probs(freq_map: np.ndarray, ts_win: np.ndarray) -> np.ndarray:
    doy0, hour = doy_hour_of(ts_win.ravel())
    doy0 = doy0.reshape(ts_win.shape); hour = hour.reshape(ts_win.shape)
    return freq_map.transpose(2, 3, 0, 1)[doy0, hour]     # (n_samples, H, N, D)


def _opponent_ratios(train_data, ts_train, val_data, ts_val, test_data, ts_test,
                      var_names_input, label_vars,
                      day_win: int, hour_win: int) -> Dict:
    """
    Build thresholds → labels → freq map → opponent probs → val τ selection →
    test compute_all → PR-AUC/prev per label variable. Returns dict.

    Only non-tp variables are computed here (tp behaviour is dominated by
    small-sample noise; the reviewer's rule uses non-tp for gating).
    """
    # Reduce inputs to non-tp variables
    nontp_idx = [var_names_input.index(v) for v in label_vars]
    train_nontp = train_data[..., nontp_idx]
    val_nontp   = val_data[..., nontp_idx]
    test_nontp  = test_data[..., nontp_idx]

    T, cell_counts = _build_nontp_thresholds(train_nontp, ts_train, day_win, hour_win)
    fidelity = _fidelity_stats(T)

    # Labels
    y_train = _apply_labels_nontp(train_nontp, ts_train, T)
    y_val   = _apply_labels_nontp(val_nontp,   ts_val,   T)
    y_test  = _apply_labels_nontp(test_nontp,  ts_test,  T)

    # Windowed targets
    val_tgt, val_tsw   = _windowed(y_val,  ts_val,  INPUT_LEN, HORIZON)
    test_tgt, test_tsw = _windowed(y_test, ts_test, INPUT_LEN, HORIZON)
    prev = test_tgt.astype(np.float32).mean(axis=(0, 1, 2))    # (D_nontp,)

    # Opponent: cell frequency map
    freq_map = _build_freq_map(y_train, ts_train, day_win, hour_win)
    val_probs   = _climatology_probs(freq_map, val_tsw)
    test_probs  = _climatology_probs(freq_map, test_tsw)

    # Metrics protocol identical to model path
    taus = select_thresholds(val_probs, val_tgt, label_vars)
    metrics = compute_all(test_probs, test_tgt, var_names=label_vars, taus=taus)

    per_var_ratios = {}
    for v in label_vars:
        pr_auc = float(metrics["per_variable"][v]["pr_auc"])
        p = float(prev[label_vars.index(v)])
        per_var_ratios[v] = {
            "prevalence": p,
            "pr_auc":     pr_auc,
            "ratio":      (pr_auc / p) if p > 0 else float("nan"),
        }
    max_ratio = max(v["ratio"] for v in per_var_ratios.values() if not np.isnan(v["ratio"]))

    return {
        "cell_count_median":  int(np.median(cell_counts)),
        "cell_count_min":     int(cell_counts.min()),
        "cell_count_max":     int(cell_counts.max()),
        "fidelity":           fidelity,
        "per_variable":       per_var_ratios,
        "max_ratio":          float(max_ratio),
        "prev":               {v: per_var_ratios[v]["prevalence"] for v in label_vars},
    }


# ---------------------------------------------------------------------------
# Synthetic control — pure seasonal + diurnal + Gaussian noise
# ---------------------------------------------------------------------------

def build_synthetic_series(timestamps: np.ndarray, N: int, seed: int = 0,
                            noise_sigma: float = 1.0):
    """
    Build a synthetic (T, N, D_nontp) series that is purely:
      - annual sinusoid (seasonal cycle)
      - diurnal sinusoid (daily cycle)
      - Gaussian white noise
    NO persistence, NO autocorrelation beyond the calendar.

    Per-station random phase offsets so stations differ but share the same
    calendar structure. Returned in the same variable order as NONTP_VARS.
    """
    rng = np.random.default_rng(seed)
    ts = pd.DatetimeIndex(timestamps)
    doy = ts.dayofyear.to_numpy() - 1
    hour = ts.hour.to_numpy()
    T = len(ts)
    D = len(NONTP_VARS)
    # Base calendar signal, station-shared, variable-specific
    ann = np.sin(2 * np.pi * doy / 366)[:, None]         # (T, 1)
    dir_ = np.sin(2 * np.pi * hour / 24)[:, None]         # (T, 1)
    # Per-variable amplitude choices — arbitrary but different so channels
    # aren't identical
    amp_ann = np.array([5.0, 3.0, 1.5], dtype=np.float32)  # t2m stronger seasonal
    amp_dir = np.array([4.0, 2.0, 1.0], dtype=np.float32)  # t2m stronger diurnal
    signal = amp_ann[None, :] * ann + amp_dir[None, :] * dir_          # (T, D)
    # Per-station phase jitter (small — keep stations comparable)
    phase = rng.uniform(-0.2, 0.2, size=(N, D)).astype(np.float32)
    signal_by_station = signal[:, None, :] + phase[None, :, :]         # (T, N, D)
    noise = rng.normal(0.0, noise_sigma, size=(T, N, D)).astype(np.float32)
    X = signal_by_station.astype(np.float32) + noise
    return X


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def reconstruct_saws_timestamps(n_train, n_val, n_test):
    canonical = pd.date_range(WINDOW_START, WINDOW_END, freq="h")
    if n_train + n_val + n_test != len(canonical):
        raise ValueError("cache split sizes don't match canonical grid")
    ts = canonical.to_numpy()
    return ts[:n_train], ts[n_train:n_train+n_val], ts[n_train+n_val:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="preprocessing/cache/saws")
    ap.add_argument("--out-dir", default="results/diagnostics/phase2_labels")
    ap.add_argument("--label-vars-nontp", nargs="+", default=NONTP_VARS)
    ap.add_argument("--synth-seed", type=int, default=0)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("[sweep] loading cache")
    X_train = np.load(cache / "X_train.npy")
    X_val   = np.load(cache / "X_val.npy")
    X_test  = np.load(cache / "X_test.npy")
    with open(cache / "meta.json") as f:
        meta = json.load(f)
    var_names = meta["var_names"]
    ts_train, ts_val, ts_test = reconstruct_saws_timestamps(
        X_train.shape[0], X_val.shape[0], X_test.shape[0],
    )
    print(f"[sweep] X_train {X_train.shape}  var_names={var_names}")

    # Synthetic series matched to shapes
    print(f"[sweep] building synthetic series (seed={args.synth_seed})")
    N = X_train.shape[1]
    synth_train = build_synthetic_series(ts_train, N, seed=args.synth_seed)
    synth_val   = build_synthetic_series(ts_val,   N, seed=args.synth_seed + 1)
    synth_test  = build_synthetic_series(ts_test,  N, seed=args.synth_seed + 2)

    # Candidate grid — enumerated in the pre-registered selection order
    # (hour ascending, then day ascending)
    candidates: List[Tuple[int, int]] = [
        (h, d) for h in HOUR_WINDOWS for d in DAY_WINDOWS
    ]
    print(f"[sweep] {len(candidates)} candidates")

    rows = []
    total_t0 = time.time()
    for i, (hour_win, day_win) in enumerate(candidates, 1):
        print(f"\n[sweep] {i}/{len(candidates)}  (hour={hour_win}, day={day_win})")
        t0 = time.time()
        real = _opponent_ratios(
            X_train, ts_train, X_val, ts_val, X_test, ts_test,
            var_names, args.label_vars_nontp, day_win, hour_win,
        )
        real_elapsed = time.time() - t0
        real_vars_str = "  ".join(f"{k}={v['ratio']:.2f}x" for k, v in real["per_variable"].items())
        print(f"    real  : median_cells={real['cell_count_median']:>4d}  "
              f"max_ratio={real['max_ratio']:.3f}  {real_vars_str}  ({real_elapsed:.0f}s)")

        # Reviewer's spec: compute synthetic for ALL candidates so the noise
        # floor is visible in the appendix table.
        t0 = time.time()
        synth = _opponent_ratios(
            synth_train, ts_train, synth_val, ts_val, synth_test, ts_test,
            NONTP_VARS, args.label_vars_nontp, day_win, hour_win,
        )
        synth_elapsed = time.time() - t0
        synth_vars_str = "  ".join(f"{k}={v['ratio']:.2f}x" for k, v in synth["per_variable"].items())
        print(f"    synth : median_cells={synth['cell_count_median']:>4d}  "
              f"max_ratio={synth['max_ratio']:.3f}  {synth_vars_str}  ({synth_elapsed:.0f}s)")

        rows.append({
            "hour_window":       hour_win,
            "day_window":        day_win,
            "cell_median":       real["cell_count_median"],
            "cell_min":          real["cell_count_min"],
            "fidelity":          real["fidelity"],
            "real_max_ratio":    real["max_ratio"],
            "real_per_var":      real["per_variable"],
            "synth_max_ratio":   synth["max_ratio"],
            "synth_per_var":     synth["per_variable"],
            "prevalence":        real["prev"],
        })
    total_elapsed = time.time() - total_t0
    print(f"\n[sweep] all candidates done in {total_elapsed:.0f}s")

    # ---- Apply pre-registered selection rule ----
    print("\n[sweep] applying pre-registered selection rule…")
    print("        (hour_window, day_window) ascending; first candidate with")
    print(f"        median non-tp cell count >= {MIN_CELL_SAMPLES} AND")
    print(f"        max non-tp PR-AUC/prev <= {REAL_MAX_RATIO} on REAL data")

    winner = None
    for row in rows:
        cell_ok  = row["cell_median"] >= MIN_CELL_SAMPLES
        ratio_ok = row["real_max_ratio"] <= REAL_MAX_RATIO
        row["cell_floor_pass"]  = cell_ok
        row["real_bound_pass"]  = ratio_ok
        row["passes"]           = cell_ok and ratio_ok
        if winner is None and row["passes"]:
            winner = row

    if winner is None:
        verdict = "NO_WINNER"
        print("\n[sweep] NO candidate passed BOTH conditions.")
    else:
        verdict = "PASS"
        print(f"\n[sweep] WINNER: hour_window={winner['hour_window']}, "
              f"day_window={winner['day_window']}")
        print(f"        median cell count = {winner['cell_median']} (≥{MIN_CELL_SAMPLES})")
        print(f"        max real PR-AUC/prev = {winner['real_max_ratio']:.3f} (≤{REAL_MAX_RATIO})")
        print(f"        synthetic PR-AUC/prev = {winner['synth_max_ratio']:.3f} "
              f"({'BUG TRIPWIRE' if winner['synth_max_ratio'] > SYNTH_BUG_TRIPWIRE else 'ok'})")

        if winner["day_window"] > DAY_PHYSICS_FLAG:
            print(f"        ⚠ day_window={winner['day_window']} > {DAY_PHYSICS_FLAG} — "
                  "physics review required before acceptance.")
        if winner["synth_max_ratio"] > SYNTH_BUG_TRIPWIRE:
            print(f"        ⚠ HARD STOP: synthetic ratio {winner['synth_max_ratio']:.3f} > "
                  f"{SYNTH_BUG_TRIPWIRE} indicates a pipeline defect.")
            verdict = "SYNTH_TRIPWIRE"

    # ---- Serialise ----
    out_json = out / "window_sweep.json"
    with open(out_json, "w") as f:
        json.dump({
            "selection_rule": {
                "order":              "(hour_window, day_window) ascending",
                "cell_floor":         MIN_CELL_SAMPLES,
                "real_max_ratio":     REAL_MAX_RATIO,
                "synth_bug_tripwire": SYNTH_BUG_TRIPWIRE,
                "day_physics_flag":   DAY_PHYSICS_FLAG,
                "grid_hour_windows":  HOUR_WINDOWS,
                "grid_day_windows":   DAY_WINDOWS,
                "label_vars_nontp":   args.label_vars_nontp,
                "percentile":         PERCENTILE,
            },
            "verdict":  verdict,
            "winner":   winner,
            "rows":     rows,
        }, f, indent=2)
    print(f"\n[sweep] wrote {out_json}")

    # ---- Human-readable Markdown table ----
    md_lines = ["# Phase 2 window-sweep table\n",
                f"**Verdict:** {verdict}\n"]
    if winner:
        md_lines.append(f"**Winner:** hour_window={winner['hour_window']}, day_window={winner['day_window']}\n")
        md_lines.append(f"- Median non-tp cell count: {winner['cell_median']} (floor {MIN_CELL_SAMPLES})")
        md_lines.append(f"- Max real PR-AUC/prev: {winner['real_max_ratio']:.3f} (bound {REAL_MAX_RATIO})")
        md_lines.append(f"- Synthetic PR-AUC/prev: {winner['synth_max_ratio']:.3f} (bug tripwire {SYNTH_BUG_TRIPWIRE})\n")
    md_lines.append("## Full table (sorted by pre-registered order)\n")
    md_lines.append("| hour | day | cell_med | fidelity_var | real max/prev | synth max/prev | leak (real−synth) | pass |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        leak = r["real_max_ratio"] - r["synth_max_ratio"]
        mark = "✅" if r["passes"] else ("↓cell" if not r["cell_floor_pass"] else "↑bound")
        md_lines.append(
            f"| {r['hour_window']} | {r['day_window']} | {r['cell_median']} | "
            f"{r['fidelity']['variance_over_calendar_axes']:.3f} | "
            f"{r['real_max_ratio']:.3f} | {r['synth_max_ratio']:.3f} | {leak:+.3f} | {mark} |"
        )
    md_lines.append("\nMarkers: ✅ passes both bounds. ↓cell = cell count < 200. ↑bound = max real ratio > 1.15.")
    md_lines.append("\nInterpretation: `leak = real − synth`. Synthetic column is the estimation-noise")
    md_lines.append("floor from a pure-calendar-plus-noise series. Positive leak = residual calendar")
    md_lines.append("signal beyond estimator noise; near-zero or negative = no meaningful leakage.")

    out_md = out / "window_sweep.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))
    print(f"[sweep] wrote {out_md}")

    return 0 if verdict == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
