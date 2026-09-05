"""
Step 0 — Inventory what per_station.npz files exist and what metrics they contain.

Run this before the backfill to decide what actually needs to be recomputed.
No GPU, no model loading, ~30 seconds.

Usage:
    python scripts/inventory_per_station.py [--runs-dir runs]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    runs = (ROOT / args.runs_dir)
    if not runs.exists():
        print(f"runs dir not found: {runs}")
        sys.exit(1)

    reg_dirs = sorted(
        d for d in runs.iterdir()
        if d.is_dir()
        and (d / "summary.json").exists()
        and "_cls_" not in d.name
        and any(d.name.split("_")[1:2] == [v] for v in ("t2m", "sp", "rh", "ws"))
    )

    key_freq:  Counter = Counter()
    nan_issues: list   = []
    missing:    list   = []
    sample_shown = 0

    for d in reg_dirs:
        npz = d / "per_station.npz"
        if not npz.exists():
            missing.append(d.name)
            continue
        try:
            z = np.load(str(npz), allow_pickle=False)
        except Exception as e:
            print(f"  ERROR loading {d.name}: {e}")
            continue

        keys = tuple(sorted(z.files))
        key_freq[keys] += 1

        if sample_shown < 3:
            print(f"{d.name}:")
            for k in z.files:
                arr = z[k]
                nan_frac = float(np.isnan(arr).mean())
                print(f"  {k:30s}  shape={str(arr.shape):15s}  nan%={nan_frac:.1%}")
            print()
            sample_shown += 1

        for k in z.files:
            nan_frac = float(np.isnan(z[k]).mean())
            if nan_frac > 0:
                nan_issues.append((d.name, k, nan_frac))

    # Summary
    print("\n=== Key-set frequencies ===")
    for ks, n in sorted(key_freq.items(), key=lambda x: -x[1]):
        print(f"  {n:4d} runs: {list(ks)}")

    if nan_issues:
        print(f"\n=== Runs with NaN values ({len(nan_issues)} array/run pairs) ===")
        for name, k, frac in nan_issues[:20]:
            print(f"  {name}  key={k}  nan%={frac:.1%}")
        if len(nan_issues) > 20:
            print(f"  ... and {len(nan_issues) - 20} more")

    print(f"\n=== Missing per_station.npz ({len(missing)} runs) ===")
    for m in missing[:20]:
        print(f"  {m}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")

    print(f"\n=== Totals ===")
    total = len(reg_dirs)
    print(f"  Regression runs found: {total}")
    print(f"  Have per_station.npz:  {total - len(missing)}")
    print(f"  Missing npz:           {len(missing)}")
    print(f"  Runs with any NaN:     {len(set(n for n,_,_ in nan_issues))}")

    # Decision guidance
    print()
    extreme_keys = {"mae_extreme", "mae_nonextreme", "peak_intensity_error"}
    has_extreme = sum(
        1 for ks in key_freq.elements()
        if extreme_keys.issubset(set(ks))
    )
    if len(missing) == 0 and has_extreme == total and not nan_issues:
        print("→ All runs have complete per_station.npz with no NaN.")
        print("  Run aggregate_per_station.py directly — no backfill needed for per-station data.")
    elif len(missing) > 0:
        print(f"→ {len(missing)} runs missing per_station.npz entirely — full backfill required.")
    else:
        print(f"→ Some runs have NaN or missing extreme keys — targeted backfill required.")
        print("  Run: python scripts/backfill_extreme_metrics.py --models all --dry-run")


if __name__ == "__main__":
    main()
