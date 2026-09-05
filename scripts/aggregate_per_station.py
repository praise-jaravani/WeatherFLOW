"""
Step 1 — Aggregate per_station.npz files into PER_STATION_all_runs.csv.

Produces:
  results/PER_STATION_all_runs.csv   — one row per (model, target, seed, station, metric)
  results/PER_STATION_spread.csv     — seed-averaged worst/Q1/median/Q3/best + ratio per cell

The 'ratio' column (worst-station MAE ÷ best-station MAE) is the single number that
discharges the §3.1 geographic-skew disclosure: a ratio near 1 means skill is uniform;
a large ratio means the pooled number masks a strong geographic gradient.

Usage:
    python scripts/aggregate_per_station.py [--seeds 42-48] [--runs-dir runs]

    --seeds  inclusive range, e.g. 42-48 (default) or 42-46
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METRICS = ("mae", "rmse", "smape", "mae_extreme", "mae_nonextreme", "peak_intensity_error")
REG_VARS = {"t2m", "sp", "rh", "ws"}


def _parse_seed_range(s: str):
    lo, hi = s.split("-")
    return set(range(int(lo), int(hi) + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",    default="42-48",
                        help="Inclusive seed range to include (default: 42-48, the n=7 comparable set)")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    seed_set  = _parse_seed_range(args.seeds)
    runs_root = ROOT / args.runs_dir
    out_dir   = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    rows = []
    missing_npz, missing_metric = [], []

    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        sj = d / "summary.json"
        if not sj.exists():
            continue
        s = json.load(open(sj))
        if s.get("task") != "regression":
            continue
        if s["seed"] not in seed_set:
            continue
        if s.get("target") not in REG_VARS:
            continue

        npz = d / "per_station.npz"
        if not npz.exists():
            missing_npz.append(d.name)
            continue

        z = np.load(str(npz), allow_pickle=False)
        for metric in METRICS:
            if metric not in z.files:
                missing_metric.append((d.name, metric))
                continue
            vals = np.asarray(z[metric]).reshape(-1)   # (N,)
            for station_idx, val in enumerate(vals):
                rows.append({
                    "model":   s["model"],
                    "target":  s["target"],
                    "seed":    s["seed"],
                    "station": station_idx,
                    "metric":  metric,
                    "value":   float(val),
                })

    if not rows:
        print("No data collected — check that per_station.npz files exist.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    out_csv = out_dir / "PER_STATION_all_runs.csv"
    df.to_csv(out_csv, index=False)
    print(f"Written: {out_csv}  ({len(df):,} rows)")

    # Seed-averaged spread: worst / Q1 / median / Q3 / best + ratio
    mae_df  = df[df.metric == "mae"].copy()
    mean_by = (mae_df
               .groupby(["model", "target", "station"])["value"]
               .mean()
               .reset_index(name="seed_mean_mae"))

    def _spread(x):
        return pd.Series({
            "min":    x.min(),
            "q1":     x.quantile(0.25),
            "median": x.median(),
            "q3":     x.quantile(0.75),
            "max":    x.max(),
            "ratio":  x.max() / x.min() if x.min() > 0 else float("nan"),
        })

    spread = (mean_by
              .groupby(["model", "target"])["seed_mean_mae"]
              .apply(_spread)
              .reset_index())

    out_spread = out_dir / "PER_STATION_spread.csv"
    spread.to_csv(out_spread, index=False)
    print(f"Written: {out_spread}")

    print("\n=== Spread table (seed-averaged MAE across stations) ===")
    print(spread.to_string(index=False))

    if missing_npz:
        print(f"\nWARNING: {len(missing_npz)} runs missing per_station.npz (excluded):")
        for m in missing_npz[:10]:
            print(f"  {m}")

    if missing_metric:
        print(f"\nWARNING: {len(missing_metric)} (run, metric) pairs missing from npz:")
        for m, k in missing_metric[:10]:
            print(f"  {m}  key={k}")


if __name__ == "__main__":
    main()
