"""
Stage-0 label audit for Phase 2 sign-off.

Produces the diagnostic table that the team + supervisor must sign off before
any Phase 2 production seeds launch. Covers the three §2 gates from the plan:

  §2.4  label variable set (default excludes u10/v10 signed components)
  §2.5  tp wet-hour rule — flag stations where naive 95th-percentile threshold
        degenerates to trace level (>90% dry hours)
  §2.6  seasonal / diurnal concentration of positives

Inputs:
  --cache  directory containing X_train.npy, X_val.npy, X_test.npy,
           mean.npy, std.npy, thresholds.npy, coords.npy, meta.json.
           Defaults to the local preprocessing/cache/era5/ path (on HPC use
           /scratch/.../preprocessing/cache/era5).
  --out    directory to write the audit artifacts. Default: outputs/audit_labels/.

Outputs (written to --out):
  thresholds_table.csv           — per-station, per-variable thresholds, naive vs
                                   wet-hour variant for tp; dry-hour fraction per
                                   station for tp; wet-hour rule TRIGGER flag
  positive_rates.csv             — positive rate per (variable, split) both under
                                   the naive threshold and (for tp) under the
                                   wet-hour override
  positive_rates_seasonal.csv    — positive rate by season × variable (train)
  positive_rates_diurnal.csv     — positive rate by hour-of-day × variable (train)
  or_rate.csv                    — for reference: fraction of hours flagged by
                                   the OR-of-all definition (paper won't use it)
  SUMMARY.md                     — human-readable digest of everything above

The script does not modify any cache file. It reads only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# ETCCDI wet-hour convention adapted from R95p (wet days ≥ 1 mm/day).
# At hourly resolution we use ≥ 0.1 mm/h. ERA5 tp is in metres (m/h).
TP_WET_MM_PER_H = 0.1
TP_WET_THRESH_M = TP_WET_MM_PER_H / 1000.0        # 0.1 mm/h → 1e-4 m/h
TP_DRY_FRACTION_TRIGGER = 0.90                     # >90% dry → apply wet-hour rule

# Default label variable set from §2.4
DEFAULT_LABEL_VARIABLES = ["t2m", "d2m", "tp", "ws10", "rh", "sp"]


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def load_cache(cache_dir: Path) -> Dict[str, np.ndarray]:
    """Load the preprocessing cache. Only the fields this audit needs."""
    meta = json.loads((cache_dir / "meta.json").read_text())
    required = ["X_train", "X_val", "X_test", "mean", "std", "thresholds", "coords"]
    missing = [f for f in required if not (cache_dir / f"{f}.npy").exists()]
    if missing:
        raise FileNotFoundError(
            f"Cache at {cache_dir} missing files: {missing}. "
            "This audit must run where the full cache lives (HPC scratch)."
        )
    return {
        "X_train":    np.load(cache_dir / "X_train.npy"),
        "X_val":      np.load(cache_dir / "X_val.npy"),
        "X_test":     np.load(cache_dir / "X_test.npy"),
        "mean":       np.load(cache_dir / "mean.npy"),
        "std":        np.load(cache_dir / "std.npy"),
        "thresholds": np.load(cache_dir / "thresholds.npy"),
        "coords":     np.load(cache_dir / "coords.npy"),
        "var_names":  list(meta["var_names"]),
        "meta":       meta,
    }


def denormalise(X_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Undo z-scoring. Broadcasts (1, N, d) stats over (T, N, d) features."""
    return X_norm * std + mean


# --------------------------------------------------------------------------- #
# §2.5 — precipitation wet-hour rule
# --------------------------------------------------------------------------- #

def compute_tp_wet_hour_thresholds(
    train_phys: np.ndarray,
    tp_idx: int,
    percentile: float = 95.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every station, compute:
      dry_frac      : fraction of train hours with tp < TP_WET_THRESH_M
      naive_thresh  : 95th-percentile of all hours
      wet_thresh    : 95th-percentile of wet hours only (NaN where too few wet)

    Returns
    -------
    dry_frac       : (N,)
    naive_thresh   : (N,)
    wet_thresh     : (N,), NaN at stations with <20 wet hours (statistically
                     meaningless).
    """
    tp = train_phys[..., tp_idx]                        # (T_train, N)
    is_dry = tp < TP_WET_THRESH_M
    dry_frac = is_dry.mean(axis=0)                      # (N,)
    naive_thresh = np.percentile(tp, percentile, axis=0)  # (N,)
    N = tp.shape[1]
    wet_thresh = np.full(N, np.nan, dtype=np.float32)
    for n in range(N):
        wet_hours = tp[~is_dry[:, n], n]
        if wet_hours.size >= 20:
            wet_thresh[n] = np.percentile(wet_hours, percentile)
    return dry_frac.astype(np.float32), naive_thresh.astype(np.float32), wet_thresh


def apply_wet_hour_override(
    thresholds: np.ndarray,
    tp_idx: int,
    dry_frac: np.ndarray,
    naive_thresh_tp: np.ndarray,
    wet_thresh_tp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Where dry_frac > TP_DRY_FRACTION_TRIGGER AND we have a valid wet_thresh,
    replace the tp column with wet_thresh; elsewhere keep naive_thresh.

    Returns
    -------
    thresholds_updated : (N, d), the tp column overridden
    trigger_mask       : (N,) bool, True where the rule fired
    """
    out = thresholds.copy()
    trigger = (dry_frac > TP_DRY_FRACTION_TRIGGER) & np.isfinite(wet_thresh_tp)
    # Fall back to naive where the override doesn't fire.
    new_tp = np.where(trigger, wet_thresh_tp, naive_thresh_tp)
    out[:, tp_idx] = new_tp
    return out, trigger


# --------------------------------------------------------------------------- #
# §2.6 — positive rates
# --------------------------------------------------------------------------- #

def labels_from_thresholds(phys: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Binary labels: 1 where value > threshold. Shape (T, N, d)."""
    return (phys > thresholds[None, :, :]).astype(np.int8)


def positive_rate_by_variable(labels: np.ndarray, var_names: List[str]) -> Dict[str, float]:
    """Overall positive rate per variable (mean over T, N)."""
    return {v: float(labels[..., i].mean()) for i, v in enumerate(var_names)}


def positive_rate_by_hour_of_day(
    labels: np.ndarray,
    var_names: List[str],
    start_year: int,
    start_month: int,
) -> pd.DataFrame:
    """
    Positive rate stratified by hour-of-day (0..23) per variable.

    Assumes the cache time axis is hourly and begins at `start_year-start_month-01
    00:00` UTC. We only need hour-of-day, so the exact calendar doesn't matter as
    long as the hourly cadence is honoured — this holds for ERA5.
    """
    T = labels.shape[0]
    # Hour-of-day for each of T contiguous hourly timestamps starting at hour 0.
    hod = np.arange(T) % 24
    rows = []
    for h in range(24):
        mask = hod == h
        row = {"hour_of_day": h}
        for i, v in enumerate(var_names):
            row[v] = float(labels[mask, :, i].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def positive_rate_by_season(
    labels: np.ndarray,
    var_names: List[str],
    start_year: int,
    start_month: int,
) -> pd.DataFrame:
    """
    Positive rate stratified by meteorological season (DJF/MAM/JJA/SON) per
    variable. Season labels use Southern Hemisphere convention names but the
    definition is the standard Dec-Feb / Mar-May / Jun-Aug / Sep-Nov grouping.
    """
    T = labels.shape[0]
    # Approximate month index for each hourly timestamp: build a
    # start-anchored monthly calendar. Uses pandas for calendar-correctness.
    start = pd.Timestamp(year=start_year, month=start_month, day=1)
    ts = pd.date_range(start=start, periods=T, freq="h")
    month = ts.month.to_numpy()
    season_of_month = {12: "DJF", 1: "DJF", 2: "DJF",
                        3: "MAM", 4: "MAM", 5: "MAM",
                        6: "JJA", 7: "JJA", 8: "JJA",
                        9: "SON", 10: "SON", 11: "SON"}
    season = np.array([season_of_month[m] for m in month])
    rows = []
    for s in ("DJF", "MAM", "JJA", "SON"):
        mask = season == s
        row = {"season": s}
        for i, v in enumerate(var_names):
            row[v] = float(labels[mask, :, i].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# OR-rate for reference
# --------------------------------------------------------------------------- #

def or_rate_all_variables(labels: np.ndarray) -> float:
    """P(any variable extreme) across (T, N)."""
    return float(labels.max(axis=-1).mean())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cache", type=Path,
        default=REPO / "preprocessing" / "cache" / "era5",
        help="Preprocessing cache directory (must contain the full .npy set).",
    )
    parser.add_argument(
        "--out", type=Path,
        default=REPO / "outputs" / "audit_labels",
        help="Where to write the audit artifacts.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Reading cache: {args.cache}")
    cache = load_cache(args.cache)
    var_names = cache["var_names"]
    N = cache["thresholds"].shape[0]
    d = cache["thresholds"].shape[1]
    coords = cache["coords"]
    meta = cache["meta"]
    print(f"  {N} stations × {d} variables")
    print(f"  vars: {var_names}")

    # Recover the raw physical values from the normalised cache.
    print("\nDenormalising splits to physical units...")
    train_phys = denormalise(cache["X_train"], cache["mean"], cache["std"])
    val_phys   = denormalise(cache["X_val"],   cache["mean"], cache["std"])
    test_phys  = denormalise(cache["X_test"],  cache["mean"], cache["std"])

    naive_thresholds = cache["thresholds"]  # (N, d), same as labels.py would emit

    # ---------- §2.5 tp wet-hour audit ----------
    if "tp" not in var_names:
        print("  WARNING: no 'tp' variable in cache; skipping wet-hour audit.")
        wet_updated_thresholds = naive_thresholds
        tp_trigger = np.zeros(N, dtype=bool)
        dry_frac = np.zeros(N, dtype=np.float32)
        wet_thresh_tp = np.full(N, np.nan)
        naive_thresh_tp = naive_thresholds[:, 0]
    else:
        tp_idx = var_names.index("tp")
        print(f"\n§2.5  Auditing tp wet-hour rule (var index {tp_idx})...")
        dry_frac, naive_thresh_tp, wet_thresh_tp = compute_tp_wet_hour_thresholds(
            train_phys, tp_idx=tp_idx, percentile=95.0
        )
        wet_updated_thresholds, tp_trigger = apply_wet_hour_override(
            naive_thresholds, tp_idx, dry_frac, naive_thresh_tp, wet_thresh_tp
        )
        n_trigger = int(tp_trigger.sum())
        print(f"  dry-hour fraction: median {np.median(dry_frac):.3f}, "
              f"max {dry_frac.max():.3f}")
        print(f"  wet-hour rule fires at {n_trigger} / {N} stations "
              f"({100 * n_trigger / N:.1f}%).")

    # ---------- Emit thresholds_table.csv ----------
    thresh_rows = []
    for n in range(N):
        row = {
            "station":   int(n),
            "lat":       float(coords[n, 0]),
            "lon":       float(coords[n, 1]),
            "tp_dry_frac_train": float(dry_frac[n]),
            "tp_naive_thresh":   float(naive_thresh_tp[n]),
            "tp_wet_thresh":     float(wet_thresh_tp[n]) if np.isfinite(wet_thresh_tp[n]) else None,
            "tp_wet_rule_fires": bool(tp_trigger[n]),
        }
        for i, v in enumerate(var_names):
            row[f"{v}_thresh_naive"] = float(naive_thresholds[n, i])
            row[f"{v}_thresh_used"]  = float(wet_updated_thresholds[n, i])
        thresh_rows.append(row)
    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv(args.out / "thresholds_table.csv", index=False)
    print(f"\nWrote {args.out / 'thresholds_table.csv'}")

    # ---------- Build labels under both threshold sets ----------
    print("\nComputing labels under naive vs wet-hour thresholds...")
    labels_naive_train = labels_from_thresholds(train_phys, naive_thresholds)
    labels_naive_val   = labels_from_thresholds(val_phys,   naive_thresholds)
    labels_naive_test  = labels_from_thresholds(test_phys,  naive_thresholds)

    labels_used_train = labels_from_thresholds(train_phys, wet_updated_thresholds)
    labels_used_val   = labels_from_thresholds(val_phys,   wet_updated_thresholds)
    labels_used_test  = labels_from_thresholds(test_phys,  wet_updated_thresholds)

    # ---------- positive_rates.csv ----------
    print("Computing positive rates per (variable, split)...")
    rate_rows = []
    for split, lab_naive, lab_used in [
        ("train", labels_naive_train, labels_used_train),
        ("val",   labels_naive_val,   labels_used_val),
        ("test",  labels_naive_test,  labels_used_test),
    ]:
        for i, v in enumerate(var_names):
            rate_rows.append({
                "variable":       v,
                "split":          split,
                "pos_rate_naive": float(lab_naive[..., i].mean()),
                "pos_rate_used":  float(lab_used[..., i].mean()),
            })
    pd.DataFrame(rate_rows).to_csv(args.out / "positive_rates.csv", index=False)
    print(f"Wrote {args.out / 'positive_rates.csv'}")

    # ---------- season × hour-of-day (train, used labels) ----------
    date_range = meta.get("date_range", {})
    start_str = date_range.get("start", "2000-01")
    start_year, start_month = map(int, start_str.split("-"))
    print(f"Stratifying train positive rate by season / hour-of-day "
          f"(cache starts {start_year}-{start_month:02d})...")

    train_frac = train_phys.shape[0] / (train_phys.shape[0] + val_phys.shape[0] + test_phys.shape[0])
    del train_frac  # only used mentally to remember train_phys is the first block

    season_df = positive_rate_by_season(labels_used_train, var_names, start_year, start_month)
    season_df.to_csv(args.out / "positive_rates_seasonal.csv", index=False)
    print(f"Wrote {args.out / 'positive_rates_seasonal.csv'}")

    hod_df = positive_rate_by_hour_of_day(labels_used_train, var_names, start_year, start_month)
    hod_df.to_csv(args.out / "positive_rates_diurnal.csv", index=False)
    print(f"Wrote {args.out / 'positive_rates_diurnal.csv'}")

    # ---------- OR-rate for reference ----------
    default_label_idx = [
        var_names.index(v) for v in DEFAULT_LABEL_VARIABLES if v in var_names
    ]
    print(f"OR-rate reference (default label vars {DEFAULT_LABEL_VARIABLES})...")
    or_rows = []
    for split, lab in [("train", labels_used_train),
                       ("val",   labels_used_val),
                       ("test",  labels_used_test)]:
        or_rows.append({
            "split":    split,
            "or_rate_default_labels": float(lab[..., default_label_idx].max(axis=-1).mean()),
            "or_rate_all_vars":       or_rate_all_variables(lab),
        })
    pd.DataFrame(or_rows).to_csv(args.out / "or_rate.csv", index=False)
    print(f"Wrote {args.out / 'or_rate.csv'}")

    # ---------- SUMMARY.md ----------
    lines = []
    lines.append("# Phase 2 label audit — sign-off table\n")
    lines.append(f"- cache: `{args.cache}`")
    lines.append(f"- stations: {N}")
    lines.append(f"- variables (in cache order): {var_names}")
    lines.append(f"- default label variables (§2.4): {DEFAULT_LABEL_VARIABLES}")
    lines.append("")
    lines.append("## §2.5 tp wet-hour rule")
    lines.append(f"- rule triggers where train dry-hour fraction > {TP_DRY_FRACTION_TRIGGER:.2f}")
    lines.append(f"- dry-hour fraction (train): median {np.median(dry_frac):.3f}, "
                 f"max {dry_frac.max():.3f}, min {dry_frac.min():.3f}")
    lines.append(f"- rule fires at {int(tp_trigger.sum())} / {N} stations "
                 f"({100 * tp_trigger.sum() / N:.1f}%)")
    if tp_trigger.any():
        naive_mean = float(naive_thresh_tp[tp_trigger].mean())
        wet_mean = float(wet_thresh_tp[tp_trigger].mean())
        lines.append(f"- at triggered stations: mean naive threshold "
                     f"{naive_mean:.3e} m/h → mean wet-hour threshold "
                     f"{wet_mean:.3e} m/h "
                     f"(shift factor ×{wet_mean / max(naive_mean, 1e-12):.1f})")
    lines.append("")
    lines.append("## Positive rates per variable × split")
    rate_pivot = pd.DataFrame(rate_rows).pivot(
        index="variable", columns="split", values="pos_rate_used"
    ).reindex(index=var_names)
    lines.append(rate_pivot.round(4).to_markdown())
    lines.append("")
    lines.append("## OR-rate reference (analysis-only; not used for training)")
    lines.append(pd.DataFrame(or_rows).round(4).to_markdown(index=False))
    lines.append("")
    lines.append("## Positive rate by season (train, wet-hour override applied)")
    lines.append(season_df.round(4).to_markdown(index=False))
    lines.append("")
    lines.append("## Positive rate by hour of day (train, wet-hour override applied)")
    lines.append(hod_df.round(4).to_markdown(index=False))
    lines.append("")
    (args.out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.out / 'SUMMARY.md'}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
