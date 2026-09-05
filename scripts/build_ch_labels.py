#!/usr/bin/env python3
"""
Build calendar/hour-relative label artefacts and add them to the existing
SAWS cache without modifying any static-scheme artefact.

Additive files (all in preprocessing/cache/saws/):
    thresholds_ch.npy       (N, D, 366, 24) float32
    y_train_ch.npy          (T_train, N, D) int8
    y_val_ch.npy            (T_val,   N, D) int8
    y_test_ch.npy           (T_test,  N, D) int8
    tp_wet_trigger_ch.npy   (N, 366, 24)    int8   (rung code per cell; -1 if no tp)
    labels_ch_meta.json     scheme metadata + rung histogram + sample-count stats

Notes on reproducibility
------------------------
Training timestamps are reconstructed from the canonical SAWS window
(2016-01-01 00:00 → 2025-12-31 23:00 hourly) and the cache's train/val/test
slice sizes. No raw SAWS files are read; no network I/O.

Nothing existing is overwritten. Static-scheme artefacts (y_train.npy,
thresholds.npy, tp_wet_trigger.npy) are left untouched so Phase-1 stratified
metrics remain reproducible.

Usage:
    python scripts/build_ch_labels.py [--dataset saws] [--percentile 95.0] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.labels_ch import (       # noqa: E402
    DAY_WIN, DAY_WIN_WIDE, HOUR_WIN, PERCENTILE,
    RUNG_CELL, RUNG_WIDE_DAY, RUNG_HOUR_POOLED, RUNG_SEASONAL, RUNG_GLOBAL_WET,
    label_all_splits_ch,
)
from preprocessing.saws_parser import WINDOW_START, WINDOW_END   # noqa: E402


def reconstruct_saws_timestamps(n_train: int, n_val: int, n_test: int) -> tuple:
    """Split the canonical SAWS hourly grid the same way the pipeline did."""
    canonical = pd.date_range(WINDOW_START, WINDOW_END, freq="h")
    total = len(canonical)
    if n_train + n_val + n_test != total:
        raise ValueError(
            f"cache split sizes ({n_train}+{n_val}+{n_test}={n_train+n_val+n_test}) "
            f"don't match canonical grid ({total}). Cache built from a different "
            "window — this script needs updating."
        )
    ts = canonical.to_numpy()
    return ts[:n_train], ts[n_train:n_train+n_val], ts[n_train+n_val:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="saws", choices=["saws"],
                    help="only saws supported (ERA5 uses monthly manifest → different reconstruction)")
    ap.add_argument("--percentile", type=float, default=PERCENTILE)
    ap.add_argument("--cache-dir", default="preprocessing/cache/saws")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + report but do not write artefacts")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing _ch artefacts if present")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    if not cache.is_dir():
        print(f"[FAIL] cache dir not found: {cache}", file=sys.stderr); return 2

    # Refuse if artefacts already exist and --force not passed
    ch_files = [cache / f for f in ("thresholds_ch.npy", "y_train_ch.npy",
                                     "y_val_ch.npy", "y_test_ch.npy",
                                     "tp_wet_trigger_ch.npy", "labels_ch_meta.json")]
    existing = [f for f in ch_files if f.exists()]
    if existing and not args.force and not args.dry_run:
        print(f"[FAIL] _ch artefacts already present ({len(existing)}/6): {[f.name for f in existing]}",
              file=sys.stderr)
        print("       pass --force to overwrite, or --dry-run to inspect", file=sys.stderr)
        return 2

    print(f"[build-ch] loading cache from {cache}")
    X_train = np.load(cache / "X_train.npy")
    X_val   = np.load(cache / "X_val.npy")
    X_test  = np.load(cache / "X_test.npy")
    print(f"[build-ch]   X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")

    with open(cache / "meta.json") as f:
        meta = json.load(f)
    var_names = meta["var_names"]
    print(f"[build-ch]   var_names = {var_names}")

    ts_train, ts_val, ts_test = reconstruct_saws_timestamps(
        X_train.shape[0], X_val.shape[0], X_test.shape[0],
    )
    print(f"[build-ch] train span: {ts_train[0]} → {ts_train[-1]}  ({len(ts_train)} hours)")

    # ---- fit + label ----
    t0 = time.time()
    y_train, y_val, y_test, thresh_ch, rung_map, info = label_all_splits_ch(
        X_train, X_val, X_test, ts_train, ts_val, ts_test,
        var_names=var_names, percentile=args.percentile,
    )
    elapsed = time.time() - t0
    print(f"[build-ch] label computation done in {elapsed:.1f}s")

    # ---- report: per-variable pos rates + per-cell sample stats ----
    print()
    print("[build-ch] positive rates per variable (macro-mean over cells → close to 5%):")
    per_var_train = {}
    per_var_val   = {}
    per_var_test  = {}
    for i, v in enumerate(var_names):
        pr_tr = float(y_train[..., i].mean())
        pr_va = float(y_val  [..., i].mean())
        pr_te = float(y_test [..., i].mean())
        per_var_train[v] = pr_tr; per_var_val[v] = pr_va; per_var_test[v] = pr_te
        print(f"    {v:>4s}:  train={pr_tr:.4f}  val={pr_va:.4f}  test={pr_te:.4f}")

    # non-tp positive-rate sanity: within-window per-cell rate should be ~5%
    # by construction of the 95th-percentile threshold + strict `>` comparator.
    #
    # For each (n, v, d, h) cell, the threshold was fit from ~270 training
    # samples in the (d±7, h±1) window. Within those 270 samples, the
    # fraction strictly above threshold ≈ 5% - small edge effects (ties,
    # np.percentile linear interpolation). This is the check we want.
    print()
    nontp_idx = [i for i, v in enumerate(var_names) if v != "tp"]
    if nontp_idx:
        from preprocessing.labels_ch import _cell_indices_all, DAY_WIN, HOUR_WIN
        doy0 = pd.DatetimeIndex(ts_train).dayofyear.to_numpy() - 1
        hour = pd.DatetimeIndex(ts_train).hour.to_numpy()
        idx_map = _cell_indices_all(doy0.astype(np.int16), hour.astype(np.int8),
                                     DAY_WIN, HOUR_WIN)
        # Sample cells uniformly across the year to avoid a 366×24-way sweep
        sampled_cells = [(d, h) for d in range(0, 366, 15) for h in range(0, 24, 4)]  # 25×6=150
        rates = []
        for (d, h) in sampled_cells:
            idx = idx_map[(d, h)]
            if idx.size == 0:
                continue
            # For each non-tp variable and each station, compute the fraction
            # of samples in this cell's window that received label=1.
            cell_labels = y_train[idx][..., nontp_idx]     # (K_cell, N, D_nontp)
            rates.append(cell_labels.mean())
        cell_rates = np.array(rates)
        med = float(np.median(cell_rates)); mn = float(cell_rates.mean())
        p25 = float(np.percentile(cell_rates, 25)); p75 = float(np.percentile(cell_rates, 75))
        print(f"[build-ch] non-tp within-window per-cell positive rate ({len(rates)} sampled cells):")
        print(f"    mean={mn:.4f}  median={med:.4f}  IQR=[{p25:.4f},{p75:.4f}]")
        # Tolerance widened: strict `>` + interpolation typically yields
        # median in [4.3%, 5.1%]; anything outside [3.5%, 6.5%] is a red flag.
        if 0.045 <= med <= 0.055:
            print(f"    [ok] median cell rate ~5% by construction")
        elif 0.035 <= med <= 0.065:
            print(f"    [ok, loose] median cell rate {med:.4f} — expected {5}% ± threshold-interp edge effects")
        else:
            print(f"    [WARN] median cell rate {med:.4f} outside [3.5%, 6.5%] — possible bug")

    print()
    print("[build-ch] rung usage histogram (tp cells only):")
    rung_hist = info.get("rung_hist", {})
    total_cells = sum(rung_hist.values()) or 1
    labels = {
        RUNG_CELL: "cell (±7d,±1h)", RUNG_WIDE_DAY: "wide day (±15d)",
        RUNG_HOUR_POOLED: "hour-pooled", RUNG_SEASONAL: "seasonal",
        RUNG_GLOBAL_WET: "global wet static",
    }
    for r, name in labels.items():
        cnt = rung_hist.get(r, 0)
        print(f"    rung {r} [{name:<20s}]: {cnt:>8d}  ({100*cnt/total_cells:5.1f}%)")

    print()
    print(f"[build-ch] threshold-fit info: {info}")

    if args.dry_run:
        print("\n[build-ch] --dry-run: skipping writes"); return 0

    # ---- write artefacts ----
    print()
    print("[build-ch] writing artefacts:")
    np.save(cache / "thresholds_ch.npy",     thresh_ch)
    print(f"    thresholds_ch.npy     {thresh_ch.shape} float32 ({thresh_ch.nbytes/1024:.0f} KB)")
    np.save(cache / "y_train_ch.npy",        y_train)
    print(f"    y_train_ch.npy        {y_train.shape} int8 ({y_train.nbytes/1024:.0f} KB)")
    np.save(cache / "y_val_ch.npy",          y_val)
    print(f"    y_val_ch.npy          {y_val.shape} int8 ({y_val.nbytes/1024:.0f} KB)")
    np.save(cache / "y_test_ch.npy",         y_test)
    print(f"    y_test_ch.npy         {y_test.shape} int8 ({y_test.nbytes/1024:.0f} KB)")
    np.save(cache / "tp_wet_trigger_ch.npy", rung_map)
    print(f"    tp_wet_trigger_ch.npy {rung_map.shape} int8 ({rung_map.nbytes/1024:.0f} KB)")

    meta_out = {
        **info,
        "positive_rates": {
            "train_per_var": per_var_train,
            "val_per_var":   per_var_val,
            "test_per_var":  per_var_test,
        },
        "rung_labels": labels,
        "train_span": {
            "start": str(pd.Timestamp(ts_train[0])),
            "end":   str(pd.Timestamp(ts_train[-1])),
            "n_hours": int(len(ts_train)),
        },
        "generated_by": "scripts/build_ch_labels.py",
        "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
    }
    # numpy → python types
    def _clean(o):
        if isinstance(o, dict):  return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, list):  return [_clean(v) for v in o]
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return o
    with open(cache / "labels_ch_meta.json", "w") as f:
        json.dump(_clean(meta_out), f, indent=2)
    print(f"    labels_ch_meta.json   (scheme + rung hist + pos rates + train span)")

    print()
    print("[build-ch] done. Existing static-scheme artefacts (y_train.npy, thresholds.npy,")
    print("           tp_wet_trigger.npy) are unchanged and remain the Phase-1 reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
