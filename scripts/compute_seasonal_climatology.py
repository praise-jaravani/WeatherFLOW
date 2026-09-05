#!/usr/bin/env python3
"""
Seasonal-climatology and companion classification baselines under a chosen
label scheme (static or calendar_hour_relative).

Baselines emitted (all under the SAME label definition per invocation):
  - seasonal-climatology: pred = training exceedance frequency in the same
    (doy±7, hour±1) cell (with the same fallback ladder as the CH threshold
    builder for tp sparse cells).
  - persistence: pred = label at time t (repeated H times) — reference for
    the model's `identity` baseline.
  - base-rate:  pred = training positive rate (constant per station × var) —
    reference for a purely-unconditional classifier.

Metrics pipeline is identical to the model classification path
(`evaluation.classification.compute_all` + `select_thresholds`), so JSONs
have the same schema as model summaries.

Expected pattern (validation of Update 1)
-----------------------------------------
- Under calendar_hour_relative labels: seasonal-climatology should ~= base
  rate (PR-AUC ≤ 1.15 × prevalence per var). If materially above → the
  deseasonalisation leaked.
- Under static labels: seasonal-climatology substantially above base rate
  — this quantifies the calendar confound the new scheme removes.

Usage:
    python scripts/compute_seasonal_climatology.py \\
        --dataset saws \\
        --scheme calendar_hour_relative        # or 'static'
        [--out results/baselines]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.labels_ch import (               # noqa: E402
    DAY_WIN, DAY_WIN_WIDE, HOUR_WIN, _cell_indices_all, doy_hour_of,
    RUNG_CELL, RUNG_WIDE_DAY, RUNG_HOUR_POOLED, RUNG_SEASONAL, RUNG_GLOBAL_WET,
    _season_of,
)
from preprocessing.saws_parser import WINDOW_START, WINDOW_END      # noqa: E402
from evaluation.classification import compute_all, select_thresholds  # noqa: E402


LABEL_VARS_DEFAULT = ["t2m", "tp", "rh", "ws"]      # Phase-2 §2.1 target set
INPUT_LEN = 12                                       # matches cli/pipeline
HORIZON = 48


# ---------------------------------------------------------------------------
# Frequency-map builder — analogue of compute_ch_thresholds but for LABEL FREQ
# ---------------------------------------------------------------------------

def build_cell_freq_map(
    y_train: np.ndarray,
    timestamps_train: np.ndarray,
    var_names: list,
    day_win: int = DAY_WIN,
    hour_win: int = HOUR_WIN,
    day_win_wide: int = DAY_WIN_WIDE,
    min_samples: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute exceedance frequency per (station, variable, doy, hour) cell
    from TRAIN labels.

    Fallback ladder (same as CH threshold builder) for cells with < min_samples:
        rung 0: cell (doy±7, hour±1)
        rung 1: wide day (doy±15, hour±1)
        rung 2: hour-pooled (doy±15, any hour)
        rung 3: season-level (all days in season, all hours) — per (var, station)
        rung 4: global train rate per (var, station)

    Args:
        y_train: (T_train, N, D_label) int8. Already subsetted to label vars
                 if scheme is calendar/label-restricted.
        timestamps_train: (T_train,) datetime64[ns]
        var_names: names matching D_label axis of y_train.
        day_win, hour_win, day_win_wide, min_samples: as above.

    Returns:
        freq_map: (N, D, 366, 24) float32 exceedance frequency in [0, 1].
        rung_map: (N, D, 366, 24) int8 rung code used.
    """
    T, N, D = y_train.shape
    doy0, hour = doy_hour_of(timestamps_train)
    idx_cell = _cell_indices_all(doy0, hour, day_win,     hour_win)
    idx_wide = _cell_indices_all(doy0, hour, day_win_wide, hour_win)
    idx_pool = _cell_indices_all(doy0, hour, day_win_wide, hour_win, pool_hours=True)

    # Precompute per-station-var season pooled rate (rung 3) and global rate (rung 4)
    doy_season = np.array([_season_of(int(d)) for d in doy0], dtype=np.int8)
    season_rate = np.zeros((4, N, D), dtype=np.float32)
    for s in range(4):
        mask = doy_season == s
        if mask.any():
            season_rate[s] = y_train[mask].astype(np.float32).mean(axis=0)  # (N, D)
    global_rate = y_train.astype(np.float32).mean(axis=0)     # (N, D)

    freq = np.zeros((N, D, 366, 24), dtype=np.float32)
    rung = np.full((N, D, 366, 24), -1, dtype=np.int8)

    for d in range(366):
        season = _season_of(d)
        for h in range(24):
            ic = idx_cell[(d, h)]; iw = idx_wide[(d, h)]; ip = idx_pool[(d, h)]
            # Vectorised per-station-var rung selection
            cell_sz  = ic.size
            wide_sz  = iw.size
            pool_sz  = ip.size
            if cell_sz >= min_samples:
                freq[:, :, d, h] = y_train[ic].astype(np.float32).mean(axis=0)
                rung[:, :, d, h] = RUNG_CELL
                continue
            if wide_sz >= min_samples:
                freq[:, :, d, h] = y_train[iw].astype(np.float32).mean(axis=0)
                rung[:, :, d, h] = RUNG_WIDE_DAY
                continue
            if pool_sz >= min_samples:
                freq[:, :, d, h] = y_train[ip].astype(np.float32).mean(axis=0)
                rung[:, :, d, h] = RUNG_HOUR_POOLED
                continue
            freq[:, :, d, h] = season_rate[season]
            rung[:, :, d, h] = RUNG_SEASONAL

    # Whatever's left with 0 samples upstream will now be at rung=RUNG_SEASONAL,
    # but the season array might itself have a 0 for tp-dry stations — fall
    # back to global rate as a last resort. For non-tp variables season_rate is
    # always finite (many samples).
    zero_mask = (freq == 0) & (rung <= RUNG_SEASONAL)
    if zero_mask.any():
        # For those cells, use global rate; if it's also 0, leave 0.
        freq_global = np.broadcast_to(global_rate[:, :, None, None],
                                       (N, D, 366, 24)).copy()
        freq[zero_mask] = freq_global[zero_mask]
        rung[zero_mask] = RUNG_GLOBAL_WET

    return freq, rung


# ---------------------------------------------------------------------------
# Windowed sample construction (matches trainer's classification path)
# ---------------------------------------------------------------------------

def _windowed_labels(y: np.ndarray, timestamps: np.ndarray,
                      input_len: int, horizon: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over y to produce (n_samples, horizon, N, D) targets and
    the timestamp array of shape (n_samples, horizon).
    """
    T = y.shape[0]
    n_samples = T - input_len - horizon + 1
    starts = np.arange(n_samples)
    # For each sample, target hours are [start+INPUT_LEN, start+INPUT_LEN+H)
    target_idx = starts[:, None] + input_len + np.arange(horizon)[None, :]  # (n_samples, H)
    y_windowed = y[target_idx]                                                # (n_samples, H, N, D)
    ts_windowed = timestamps[target_idx]                                      # (n_samples, H)
    return y_windowed, ts_windowed


def climatology_probs(freq_map: np.ndarray, ts_windowed: np.ndarray) -> np.ndarray:
    """Emit (n_samples, H, N, D) probabilities via cell-lookup."""
    doy0, hour = doy_hour_of(ts_windowed.ravel())
    doy0 = doy0.reshape(ts_windowed.shape)
    hour = hour.reshape(ts_windowed.shape)
    # freq_map: (N, D, 366, 24) → transpose to (366, 24, N, D) for gather
    T = freq_map.transpose(2, 3, 0, 1)                                         # (366, 24, N, D)
    return T[doy0, hour]                                                        # (n_samples, H, N, D)


def persistence_probs(y: np.ndarray, input_len: int, horizon: int) -> np.ndarray:
    """label̂(t+h) = label(t+input_len-1) broadcast across horizon steps."""
    T = y.shape[0]
    n_samples = T - input_len - horizon + 1
    starts = np.arange(n_samples)
    last_obs = y[starts + input_len - 1]                # (n_samples, N, D)
    return np.broadcast_to(
        last_obs[:, None, :, :].astype(np.float32),
        (n_samples, horizon, *last_obs.shape[1:])
    ).copy()


def base_rate_probs(base_rate: np.ndarray, n_samples: int, horizon: int) -> np.ndarray:
    """Constant per-(N, D) probability."""
    return np.broadcast_to(
        base_rate[None, None, :, :].astype(np.float32),
        (n_samples, horizon, *base_rate.shape),
    ).copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def reconstruct_saws_timestamps(n_train: int, n_val: int, n_test: int):
    canonical = pd.date_range(WINDOW_START, WINDOW_END, freq="h")
    if n_train + n_val + n_test != len(canonical):
        raise ValueError(
            f"cache split sizes ({n_train}+{n_val}+{n_test}) "
            f"don't match canonical ({len(canonical)})"
        )
    ts = canonical.to_numpy()
    return ts[:n_train], ts[n_train:n_train+n_val], ts[n_train+n_val:]


def load_labels(cache: Path, scheme: str, label_vars: list, var_names: list):
    """Return (y_train, y_val, y_test) already subsetted to label_vars axis."""
    if scheme == "static":
        y_train = np.load(cache / "y_train.npy")
        y_val   = np.load(cache / "y_val.npy")
        y_test  = np.load(cache / "y_test.npy")
    elif scheme == "calendar_hour_relative":
        y_train = np.load(cache / "y_train_ch.npy")
        y_val   = np.load(cache / "y_val_ch.npy")
        y_test  = np.load(cache / "y_test_ch.npy")
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    idx = [var_names.index(v) for v in label_vars]
    return y_train[..., idx], y_val[..., idx], y_test[..., idx]


def emit_baseline(name: str, val_probs: np.ndarray, val_targets: np.ndarray,
                   test_probs: np.ndarray, test_targets: np.ndarray,
                   label_vars: list, scheme: str, out_dir: Path,
                   prevalence: np.ndarray, extra: dict = None):
    """Run τ selection on val, metrics on test, write JSON."""
    print(f"\n=== Baseline: {name}  (scheme={scheme}) ===", flush=True)
    print(f"    selecting τ per variable on val…", flush=True)
    taus = select_thresholds(val_probs, val_targets, label_vars)
    print(f"    τ selected: {dict(zip(label_vars, taus.tolist()))}", flush=True)
    print(f"    computing test metrics…", flush=True)
    metrics = compute_all(test_probs, test_targets, var_names=label_vars, taus=taus)
    # Report headline
    pv = metrics["per_variable"]
    macro = metrics["macro"]
    print(f"    macro F1={macro['f1']:.4f}  PR-AUC={macro['pr_auc']:.4f}  "
          f"CSI={macro['csi']:.4f}  ROC-AUC={macro['auc_roc']:.4f}")
    for v in label_vars:
        pv_v = pv[v]
        prev = float(prevalence[label_vars.index(v)])
        ratio = pv_v["pr_auc"] / prev if prev > 0 else np.nan
        print(f"    {v:>4s}: F1={pv_v['f1']:.4f}  PR-AUC={pv_v['pr_auc']:.4f} "
              f"(prev={prev:.4f}, PR-AUC/prev={ratio:.2f})")

    # Serialise — strip per_station np arrays into lists
    def _clean(o):
        if isinstance(o, dict):  return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, list):  return [_clean(v) for v in o]
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return o
    summary = {
        "baseline":         name,
        "scheme":           scheme,
        "label_variables":  label_vars,
        "per_variable":     _clean(metrics["per_variable"]),
        "macro":            _clean(metrics["macro"]),
        "taus":             _clean(metrics["taus"]),
        "prevalence":       _clean(prevalence.tolist()),
        "pr_auc_over_prevalence": _clean({
            v: (pv[v]["pr_auc"] / prevalence[i]) if prevalence[i] > 0 else None
            for i, v in enumerate(label_vars)
        }),
        "n_test_samples":   int(test_probs.shape[0]),
        "input_len":        INPUT_LEN,
        "horizon":          HORIZON,
    }
    if extra:
        summary["extra"] = _clean(extra)
    out_path = out_dir / f"classification-{name}-{scheme}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    wrote {out_path}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="saws", choices=["saws"])
    ap.add_argument("--scheme", required=True, choices=["static", "calendar_hour_relative"],
                    help="which label scheme to build baselines against")
    ap.add_argument("--cache-dir", default="preprocessing/cache/saws")
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--label-vars", nargs="+", default=LABEL_VARS_DEFAULT)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cache / "meta.json") as f:
        meta = json.load(f)
    var_names = meta["var_names"]
    print(f"[baselines] scheme={args.scheme}  label_vars={args.label_vars}  var_names={var_names}")

    # ---- load labels + reconstruct timestamps ----
    y_train, y_val, y_test = load_labels(cache, args.scheme, args.label_vars, var_names)
    print(f"[baselines] label shapes: train {y_train.shape}  val {y_val.shape}  test {y_test.shape}")
    ts_train, ts_val, ts_test = reconstruct_saws_timestamps(
        y_train.shape[0], y_val.shape[0], y_test.shape[0],
    )

    # ---- windowed targets (val + test) ----
    val_targets,  val_ts_win  = _windowed_labels(y_val,  ts_val,  INPUT_LEN, HORIZON)
    test_targets, test_ts_win = _windowed_labels(y_test, ts_test, INPUT_LEN, HORIZON)
    print(f"[baselines] windowed: val_targets {val_targets.shape}  test_targets {test_targets.shape}")

    # ---- prevalence per variable on TEST windowed labels ----
    prevalence = test_targets.astype(np.float32).mean(axis=(0, 1, 2))    # (D_label,)
    print(f"[baselines] test prevalence per var: {dict(zip(args.label_vars, prevalence.tolist()))}")

    # ---- 1. seasonal-climatology (frequency map) ----
    t0 = time.time()
    freq_map, freq_rung = build_cell_freq_map(y_train, ts_train, args.label_vars)
    print(f"[baselines] freq map built in {time.time() - t0:.1f}s  shape={freq_map.shape}")
    val_probs   = climatology_probs(freq_map, val_ts_win)
    test_probs  = climatology_probs(freq_map, test_ts_win)
    emit_baseline("seasonal-climatology", val_probs, val_targets, test_probs, test_targets,
                   args.label_vars, args.scheme, out_dir, prevalence,
                   extra={"freq_map_rung_hist": {
                       int(r): int((freq_rung == r).sum())
                       for r in (RUNG_CELL, RUNG_WIDE_DAY, RUNG_HOUR_POOLED,
                                 RUNG_SEASONAL, RUNG_GLOBAL_WET)
                   }})

    # ---- 2. persistence ----
    val_probs   = persistence_probs(y_val,  INPUT_LEN, HORIZON)
    test_probs  = persistence_probs(y_test, INPUT_LEN, HORIZON)
    emit_baseline("persistence", val_probs, val_targets, test_probs, test_targets,
                   args.label_vars, args.scheme, out_dir, prevalence)

    # ---- 3. base-rate ----
    base_rate = y_train.astype(np.float32).mean(axis=0)   # (N, D_label)
    val_probs   = base_rate_probs(base_rate, val_targets.shape[0],  HORIZON)
    test_probs  = base_rate_probs(base_rate, test_targets.shape[0], HORIZON)
    emit_baseline("base-rate", val_probs, val_targets, test_probs, test_targets,
                   args.label_vars, args.scheme, out_dir, prevalence)

    print(f"\n[baselines] done — wrote 3 JSONs under {out_dir} (scheme={args.scheme})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
