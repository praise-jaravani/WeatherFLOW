#!/usr/bin/env python3
"""Build STGODE's semantic-adjacency DTW distance matrix from the SAWS cache.

Corrected 2026-08-11 per peer-review finding C2 (see
specs/STGODE_INTEGRATION_AUDIT.md).

Upstream `STGODE/utils.py:51` computes:

    data_mean = np.mean([data[24*12*i : 24*12*(i+1)] for i in ...], axis=0)

which is `mean OVER DAYS at each within-day step` — the result is the average
diurnal profile per station (length = steps-per-day; 288 for PEMS 5-min data).
DTW is then computed between these per-station diurnal profiles.

Our earlier (broken) implementation computed `.reshape(n_days, day_len, N).mean(axis=1)`
— the mean WITHIN each day, yielding a seasonality-dominated daily-mean time
series of length n_days. The two are orthogonal similarity notions: stations
with the same diurnal shape but opposite seasonal phase are IDENTICAL under
the paper's construction and OPPOSITE under the broken one (rank inversion).

This corrected script:
  1. Slices X_train to `--variable` (default t2m).
  2. Reshapes to (n_days, hours_per_day, N).
  3. Averages ACROSS days at each within-day step → (hours_per_day, N).
  4. Transposes to (N, hours_per_day) = one 24-step diurnal profile per station.
  5. Computes pairwise fastdtw with `radius=6` (upstream default).
  6. Writes `dtw_matrix_{variable}.npy` — variable-suffixed so each target's
     graph is available (semantic similarity is variable-specific).

At length 24 the DTW computation is milliseconds per pair — the whole matrix
takes seconds, not minutes. So `--force` is cheap and per-target builds are
trivial to add.

Usage:
    # Default: t2m diurnal profile
    python scripts/build_stgode_semantic_adj.py

    # Per-target rebuild for the confirmation stage
    for v in t2m sp rh ws; do
        python scripts/build_stgode_semantic_adj.py --variable $v --force
    done
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from fastdtw import fastdtw


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="preprocessing/cache/saws",
                   help="preprocessing cache dir (default preprocessing/cache/saws)")
    p.add_argument("--variable", default="t2m",
                   help="which variable's diurnal profile to DTW (default t2m)")
    p.add_argument("--hours-per-day", type=int, default=24,
                   help="steps per day at the data cadence (default 24 for 1-hour SAWS)")
    p.add_argument("--radius", type=int, default=6,
                   help="fastdtw radius parameter (default 6, matches upstream)")
    p.add_argument("--force", action="store_true",
                   help="recompute even if output exists")
    args = p.parse_args()

    cache = Path(args.cache)
    out = cache / f"dtw_matrix_{args.variable}.npy"
    meta_out = cache / f"dtw_matrix_{args.variable}.meta.json"

    if out.exists() and not args.force:
        print(f"[skip] {out} already exists (use --force to recompute)")
        return 0

    # Load cache
    x_train = np.load(cache / "X_train.npy", mmap_mode="r")
    with open(cache / "meta.json") as f:
        meta = json.load(f)
    var_names = meta["var_names"]

    if args.variable not in var_names:
        print(f"[fatal] variable {args.variable!r} not in cache var_names {var_names}",
              file=sys.stderr)
        return 2
    var_idx = var_names.index(args.variable)

    T, N, D = x_train.shape
    print(f"Cache: T={T} N={N} D={D}  var_names={var_names}")
    print(f"DTW source: variable={args.variable} (idx={var_idx})  hours_per_day={args.hours_per_day}")

    # Slice to the single variable
    sig = np.asarray(x_train[:, :, var_idx], dtype=np.float32)  # (T, N)

    # ---- CORRECTED axis: mean OVER days at each within-day step ----
    hpd = args.hours_per_day
    n_days = T // hpd
    if n_days < 2:
        print(f"[fatal] not enough data for hours_per_day={hpd}: T={T} → {n_days} days",
              file=sys.stderr)
        return 3
    # (T, N) → (n_days, hpd, N) → mean over axis=0 → (hpd, N)  — the diurnal profile
    sig = sig[: n_days * hpd].reshape(n_days, hpd, N).mean(axis=0)
    per_station = sig.T  # (N, hpd) = one 24-step diurnal profile per station

    print(f"per-station diurnal-profile shape (N, hours_per_day) = {per_station.shape}")
    print(f"computing {N*(N-1)//2} pairwise DTW distances (radius={args.radius})...")

    dtw = np.zeros((N, N), dtype=np.float32)
    t0 = time.time()
    total_pairs = N * (N - 1) // 2
    done = 0
    for i in range(N):
        for j in range(i + 1, N):
            d, _ = fastdtw(per_station[i], per_station[j], radius=args.radius)
            dtw[i, j] = d
            dtw[j, i] = d
            done += 1
            if done % max(1, total_pairs // 10) == 0:
                pct = 100 * done / total_pairs
                elapsed = time.time() - t0
                print(f"  {done}/{total_pairs} ({pct:.1f}%) elapsed={elapsed:.2f}s")

    elapsed = time.time() - t0
    off = dtw[~np.eye(N, dtype=bool)]
    print(f"DTW distance matrix computed in {elapsed:.2f}s")
    print(f"  off-diag min/mean/max = {off.min():.3f} / {off.mean():.3f} / {off.max():.3f}")

    np.save(out, dtw)
    with open(meta_out, "w") as f:
        json.dump({
            "variable": args.variable, "var_idx": var_idx,
            "hours_per_day": hpd,
            "n_days_averaged": n_days,
            "diurnal_profile_length": hpd,
            "axis_convention": "mean_across_days_per_within_day_step",
            "radius": args.radius, "T_train": T, "N": N,
            "elapsed_seconds": elapsed,
        }, f, indent=2)

    print(f"wrote {out}")
    print(f"wrote {meta_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
