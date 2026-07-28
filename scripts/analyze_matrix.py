"""
Analyze the WeatherFLOW production matrix.

Walks `outputs/`, gathers all completed per-seed test_metrics.json + per_station.npz
files from the production runs (2026-06-23 onwards by default), and produces:

  - Per-model summary table (mean +/- std across seeds, every variable)
  - Model x variable comparison heatmap (normalised to best-in-column)
  - Pairwise Wilcoxon signed-rank tests for every model pair x variable
  - Per-horizon NRMSE curve plot
  - Geographic per-station MAE map per model
  - Markdown summary written to outputs/figures/SUMMARY.md

Idempotent: re-running after new seeds land just refreshes outputs/figures/.

Usage:
  python scripts/analyze_matrix.py                              # all production runs
  python scripts/analyze_matrix.py --since 2026-06-25           # only runs from this date
  python scripts/analyze_matrix.py --outputs outputs/ --figures outputs/figures/
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

REPO = Path(__file__).resolve().parent.parent

REGRESSION_VARS = ["t2m", "d2m", "tp", "u10", "v10", "sp", "ws10", "rh"]
# climode (the "ClimODE-inspired" advection graph-ODE) archived on 2026-07-28
# pending faithful reimplementation — see results/archive/climode_inspired/.
# climode_faithful joins this list once its production runs land.
MODEL_ORDER = ["agcrn", "clcrn", "mtgode"]

RUN_DIR_RE = re.compile(
    r"(?P<model>agcrn|clcrn|mtgode|climode|climode_faithful)_era5_regression_(?P<ts>\d{8})_(?P<hms>\d{6})"
)

RUN_DIR_RE_CLASSIFICATION = re.compile(
    r"(?P<model>agcrn|clcrn|mtgode|climode|climode_faithful)_era5_classification_(?P<ts>\d{8})_(?P<hms>\d{6})"
)

# Phase-2 metric set (see evaluation/classification.py). See §4 of the plan.
CLASSIFICATION_METRICS = ["f1", "precision", "recall", "csi", "auc_roc", "pr_auc", "accuracy"]


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def discover_runs(outputs_dir: Path, since: str) -> pd.DataFrame:
    """Return DataFrame [model, seed, ts, path] for every completed production run."""
    rows = []
    for run_dir in outputs_dir.glob("*_era5_regression_*"):
        m = RUN_DIR_RE.match(run_dir.name)
        if not m:
            continue
        date = m.group("ts")
        if date < since.replace("-", ""):
            continue
        for seed_dir in run_dir.glob("run_*"):
            metrics_path = seed_dir / "test_metrics.json"
            if not metrics_path.exists():
                continue
            seed_match = re.match(r"run_(\d+)", seed_dir.name)
            if not seed_match:
                continue
            rows.append({
                "model": m.group("model"),
                "seed": int(seed_match.group(1)),
                "ts": f"{m.group('ts')}_{m.group('hms')}",
                "path": seed_dir,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # If multiple completed runs exist for the same (model, seed), keep the newest.
    df = df.sort_values("ts").drop_duplicates(subset=["model", "seed"], keep="last")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def load_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """Long-format DataFrame: [model, seed, variable, metric, value]."""
    rows = []
    for _, r in runs.iterrows():
        metrics = json.loads((r["path"] / "test_metrics.json").read_text())
        for var in REGRESSION_VARS:
            if var not in metrics.get("mae", {}):
                continue
            row = {"model": r["model"], "seed": r["seed"], "variable": var}
            for metric_key in ["mae", "rmse", "mse", "smape", "vpt",
                               "mae_extreme", "mae_nonextreme", "peak_intensity_error"]:
                if metric_key in metrics:
                    row[metric_key] = metrics[metric_key].get(var, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std MAE per model x variable."""
    grouped = metrics.groupby(["model", "variable"])["mae"].agg(["mean", "std", "count"])
    return grouped.reset_index()


# --------------------------------------------------------------------------- #
# Statistical tests
# --------------------------------------------------------------------------- #

def wilcoxon_pairs(metrics: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests per variable. NaN if n<3 per group."""
    rows = []
    for var in metrics["variable"].unique():
        sub = metrics[metrics["variable"] == var]
        models = sorted(sub["model"].unique())
        seed_lookup = {m: sub[sub["model"] == m].set_index("seed")[metric] for m in models}
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                common_seeds = seed_lookup[a].index.intersection(seed_lookup[b].index)
                if len(common_seeds) < 3:
                    p = np.nan
                    stat = np.nan
                else:
                    x = seed_lookup[a].loc[common_seeds].values
                    y = seed_lookup[b].loc[common_seeds].values
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            stat, p = stats.wilcoxon(x, y, zero_method="wilcox")
                        except ValueError:
                            stat, p = np.nan, np.nan
                rows.append({
                    "variable": var,
                    "model_a": a,
                    "model_b": b,
                    "n": len(common_seeds),
                    "wilcoxon_stat": stat,
                    "p_value": p,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_mae_heatmap(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Heatmap of mean MAE (model x variable), normalised to best per column."""
    piv = summary.pivot(index="model", columns="variable", values="mean")
    piv = piv.reindex(index=[m for m in MODEL_ORDER if m in piv.index])
    piv = piv[[v for v in REGRESSION_VARS if v in piv.columns]]

    # normalised (relative to best per column) — easier to read across scales
    normed = piv / piv.min(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(3, 0.6 * len(piv) + 2)))
    sns.heatmap(piv, annot=True, fmt=".3g", cmap="Reds", ax=axes[0],
                cbar_kws={"label": "MAE (physical units)"})
    axes[0].set_title("Mean MAE across seeds")
    sns.heatmap(normed, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=axes[1],
                vmin=1.0, cbar_kws={"label": "MAE relative to best"})
    axes[1].set_title("Relative to best-in-variable (lower = better)")
    plt.tight_layout()
    out = fig_dir / "mae_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_variance_box(metrics: pd.DataFrame, fig_dir: Path) -> None:
    """Box plot of MAE across seeds per model, faceted by variable."""
    g = sns.catplot(
        data=metrics, x="model", y="mae", col="variable",
        col_wrap=4, kind="box", height=2.6, aspect=1.1,
        order=[m for m in MODEL_ORDER if m in metrics["model"].unique()],
        sharey=False,
    )
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=20)
    out = fig_dir / "mae_variance_box.png"
    g.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(g.fig)
    print(f"  wrote {out}")


def plot_nrmse_horizon(runs: pd.DataFrame, fig_dir: Path) -> None:
    """Mean NRMSE-per-horizon curve across seeds, one panel per variable."""
    # Collect per-(model, variable) curves across all seeds
    curves = defaultdict(lambda: defaultdict(list))  # curves[var][model] = [curve_per_seed]
    for _, r in runs.iterrows():
        m = json.loads((r["path"] / "test_metrics.json").read_text())
        if "nrmse_per_horizon" not in m:
            continue
        for var, curve in m["nrmse_per_horizon"].items():
            curves[var][r["model"]].append(np.array(curve, dtype=float))

    if not curves:
        return

    vars_present = [v for v in REGRESSION_VARS if v in curves]
    n = len(vars_present)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for i, var in enumerate(vars_present):
        ax = axes[i // ncols][i % ncols]
        for model in MODEL_ORDER:
            if model not in curves[var]:
                continue
            arr = np.stack(curves[var][model])   # (n_seeds, H)
            mean = arr.mean(axis=0)
            ax.plot(range(len(mean)), mean, marker="o", label=f"{model} (n={arr.shape[0]})")
            if arr.shape[0] > 1:
                ax.fill_between(range(len(mean)), arr.min(axis=0), arr.max(axis=0), alpha=0.1)
        ax.set_title(var)
        ax.set_xlabel("horizon step (h)")
        ax.set_ylabel("NRMSE")
        ax.axhline(0.3, color="grey", linestyle=":", linewidth=0.8)
        ax.legend(fontsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    plt.tight_layout()
    out = fig_dir / "nrmse_per_horizon.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_geographic(runs: pd.DataFrame, fig_dir: Path, coords: np.ndarray, var: str = "t2m") -> None:
    """Per-station MAE map, one panel per model. Averaged across seeds."""
    var_idx = REGRESSION_VARS.index(var)
    by_model = defaultdict(list)  # model -> list of (N,) arrays
    for _, r in runs.iterrows():
        npz_path = r["path"] / "per_station.npz"
        if not npz_path.exists():
            continue
        ps = np.load(npz_path)
        if "mae" not in ps:
            continue
        by_model[r["model"]].append(ps["mae"][:, var_idx])

    if not by_model:
        return

    models = [m for m in MODEL_ORDER if m in by_model]
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), squeeze=False)
    # use shared colour scale across models so absolute differences are visible
    all_vals = np.concatenate([np.stack(by_model[m]).mean(axis=0) for m in models])
    vmin, vmax = all_vals.min(), all_vals.max()
    for i, m in enumerate(models):
        ax = axes[0][i]
        mean = np.stack(by_model[m]).mean(axis=0)  # (N,)
        sc = ax.scatter(coords[:, 1], coords[:, 0], c=mean, s=18,
                        vmin=vmin, vmax=vmax, cmap="Reds")
        ax.set_title(f"{m}  (n={len(by_model[m])} seeds)")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_xlim(14, 34)
        ax.set_ylim(-36, -21)
        ax.set_aspect("equal")
    fig.colorbar(sc, ax=axes[0], shrink=0.7, label=f"MAE ({var})")
    fig.suptitle(f"Per-station MAE — variable: {var}", fontsize=12)
    out = fig_dir / f"geographic_{var}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #

def write_summary_md(runs: pd.DataFrame, summary: pd.DataFrame, wilcoxon: pd.DataFrame,
                     fig_dir: Path) -> None:
    md = ["# WeatherFLOW production matrix — analysis report", ""]
    counts = runs.groupby("model")["seed"].count()
    md.append("## Seed counts")
    md.append("")
    md.append("| model | n_seeds |")
    md.append("|-------|---------|")
    for m in MODEL_ORDER:
        md.append(f"| {m} | {counts.get(m, 0)} |")
    md.append("")

    md.append("## Mean MAE per variable (mean +/- std across seeds)")
    md.append("")
    piv_mean = summary.pivot(index="model", columns="variable", values="mean")
    piv_std = summary.pivot(index="model", columns="variable", values="std")
    cols = [v for v in REGRESSION_VARS if v in piv_mean.columns]
    md.append("| model | " + " | ".join(cols) + " |")
    md.append("|" + "---|" * (len(cols) + 1))
    for m in MODEL_ORDER:
        if m not in piv_mean.index:
            continue
        cells = []
        for v in cols:
            mu = piv_mean.loc[m, v]
            sd = piv_std.loc[m, v]
            if pd.isna(sd):
                cells.append(f"{mu:.3g}")
            else:
                cells.append(f"{mu:.3g} +/- {sd:.2g}")
        md.append(f"| {m} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("## Pairwise Wilcoxon signed-rank tests (paired by seed, p-values)")
    md.append("")
    if wilcoxon.empty:
        md.append("_(no model pair has n>=3 seeds yet; Wilcoxon skipped)_")
    else:
        usable = wilcoxon[wilcoxon["p_value"].notna()]
        skipped = wilcoxon[wilcoxon["p_value"].isna()]
        if not usable.empty:
            md.append("| variable | model_a | model_b | n | p |")
            md.append("|----------|---------|---------|---|---|")
            for _, w in usable.sort_values(["variable", "p_value"]).iterrows():
                star = " **" if w["p_value"] < 0.05 else ""
                md.append(f"| {w['variable']} | {w['model_a']} | {w['model_b']} | {w['n']} | {w['p_value']:.3g}{star} |")
        if not skipped.empty:
            md.append("")
            md.append(f"_{len(skipped)} pairs skipped due to n<3 seeds._")
    md.append("")

    md.append("## Figures")
    md.append("")
    for f in sorted(fig_dir.glob("*.png")):
        md.append(f"- ![{f.name}]({f.name})")
    md.append("")
    md.append(f"Generated from {len(runs)} run directories.")

    out = fig_dir / "SUMMARY.md"
    out.write_text("\n".join(md) + "\n")
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# Phase 2 — classification path
# --------------------------------------------------------------------------- #

def discover_runs_classification(outputs_dir: Path, since: str) -> pd.DataFrame:
    """Same schema as `discover_runs` but scans classification directories."""
    rows = []
    for run_dir in outputs_dir.glob("*_era5_classification_*"):
        m = RUN_DIR_RE_CLASSIFICATION.match(run_dir.name)
        if not m:
            continue
        date = m.group("ts")
        if date < since.replace("-", ""):
            continue
        for seed_dir in run_dir.glob("run_*"):
            metrics_path = seed_dir / "test_metrics.json"
            if not metrics_path.exists():
                continue
            seed_match = re.match(r"run_(\d+)", seed_dir.name)
            if not seed_match:
                continue
            rows.append({
                "model": m.group("model"),
                "seed": int(seed_match.group(1)),
                "ts": f"{m.group('ts')}_{m.group('hms')}",
                "path": seed_dir,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("ts").drop_duplicates(subset=["model", "seed"], keep="last")
    return df.reset_index(drop=True)


def load_metrics_classification(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Long-format DataFrame: [model, seed, variable, metric, value] for
    per-variable classification runs. Also emits a "macro" pseudo-variable.
    """
    rows = []
    for _, r in runs.iterrows():
        m = json.loads((r["path"] / "test_metrics.json").read_text())
        per_variable = m.get("per_variable", {})
        for var, mv in per_variable.items():
            row = {"model": r["model"], "seed": r["seed"], "variable": var}
            for k in CLASSIFICATION_METRICS + ["tau"]:
                if k in mv:
                    row[k] = mv[k]
            rows.append(row)
        # macro row for cross-model comparison
        macro = m.get("macro", {})
        if macro:
            row = {"model": r["model"], "seed": r["seed"], "variable": "macro"}
            for k in CLASSIFICATION_METRICS:
                if k in macro:
                    row[k] = macro[k]
            rows.append(row)
    return pd.DataFrame(rows)


def summary_table_classification(metrics: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    grouped = metrics.groupby(["model", "variable"])[metric].agg(["mean", "std", "count"])
    return grouped.reset_index()


def wilcoxon_pairs_classification(
    metrics: pd.DataFrame, metric: str = "f1"
) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank on per-seed values of `metric`, per variable
    (incl. "macro"). Higher is better for classification metrics.
    """
    rows = []
    for var in metrics["variable"].unique():
        sub = metrics[metrics["variable"] == var]
        models = sorted(sub["model"].unique())
        seed_lookup = {mo: sub[sub["model"] == mo].set_index("seed")[metric] for mo in models}
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                common = seed_lookup[a].index.intersection(seed_lookup[b].index)
                if len(common) < 3:
                    p = np.nan
                    stat = np.nan
                else:
                    x = seed_lookup[a].loc[common].values
                    y = seed_lookup[b].loc[common].values
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            stat, p = stats.wilcoxon(x, y, zero_method="wilcox")
                        except ValueError:
                            stat, p = np.nan, np.nan
                rows.append({
                    "variable": var,
                    "model_a": a,
                    "model_b": b,
                    "n": len(common),
                    "wilcoxon_stat": stat,
                    "p_value": p,
                })
    return pd.DataFrame(rows)


def plot_f1_heatmap(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Model × variable heatmap of mean F1 across seeds (higher is better)."""
    piv = summary.pivot(index="model", columns="variable", values="mean")
    piv = piv.reindex(index=[m for m in MODEL_ORDER if m in piv.index])
    vars_first = [c for c in piv.columns if c != "macro"] + (["macro"] if "macro" in piv.columns else [])
    piv = piv[vars_first]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * piv.shape[1]), max(3, 0.6 * len(piv) + 2)))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="Greens", ax=ax,
                cbar_kws={"label": "F1"})
    ax.set_title("Phase 2 — Mean F1 per (model, label variable)")
    plt.tight_layout()
    out = fig_dir / "phase2_f1_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_classification_variance(metrics: pd.DataFrame, fig_dir: Path) -> None:
    g = sns.catplot(
        data=metrics, x="model", y="f1", col="variable",
        col_wrap=4, kind="box", height=2.6, aspect=1.1,
        order=[m for m in MODEL_ORDER if m in metrics["model"].unique()],
        sharey=False,
    )
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=20)
    out = fig_dir / "phase2_f1_variance_box.png"
    g.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(g.fig)
    print(f"  wrote {out}")


def plot_classification_geographic(
    runs: pd.DataFrame, fig_dir: Path, coords: np.ndarray, var: str, var_idx: int
) -> None:
    """Per-station F1 map per model, averaged across seeds."""
    by_model = defaultdict(list)
    for _, r in runs.iterrows():
        npz_path = r["path"] / "per_station.npz"
        if not npz_path.exists():
            continue
        ps = np.load(npz_path)
        if "f1" not in ps:
            continue
        arr = ps["f1"]
        if arr.ndim != 2 or var_idx >= arr.shape[1]:
            continue
        by_model[r["model"]].append(arr[:, var_idx])
    if not by_model:
        return

    models = [mo for mo in MODEL_ORDER if mo in by_model]
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), squeeze=False)
    stacked = [np.stack(by_model[mo]) for mo in models]
    means = [np.nanmean(s, axis=0) for s in stacked]
    vmin = float(np.nanmin([np.nanmin(m) for m in means]))
    vmax = float(np.nanmax([np.nanmax(m) for m in means]))
    for i, (mo, mean_arr) in enumerate(zip(models, means)):
        ax = axes[0][i]
        sc = ax.scatter(coords[:, 1], coords[:, 0], c=mean_arr, s=18,
                        vmin=vmin, vmax=vmax, cmap="Greens")
        ax.set_title(f"{mo}  (n={len(by_model[mo])} seeds)")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_xlim(14, 34)
        ax.set_ylim(-36, -21)
        ax.set_aspect("equal")
    fig.colorbar(sc, ax=axes[0], shrink=0.7, label=f"F1 ({var})")
    fig.suptitle(f"Per-station F1 — variable: {var}", fontsize=12)
    out = fig_dir / f"phase2_geographic_{var}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def write_summary_md_classification(
    runs: pd.DataFrame, summary: pd.DataFrame, wlc: pd.DataFrame, fig_dir: Path,
) -> None:
    md = ["# WeatherFLOW Phase 2 — classification analysis", ""]
    counts = runs.groupby("model")["seed"].count()
    md.append("## Seed counts")
    md.append("")
    md.append("| model | n_seeds |")
    md.append("|-------|---------|")
    for m in MODEL_ORDER:
        md.append(f"| {m} | {counts.get(m, 0)} |")
    md.append("")

    md.append("## Mean F1 per (model, label variable) — mean ± std across seeds")
    md.append("")
    piv_mean = summary.pivot(index="model", columns="variable", values="mean")
    piv_std  = summary.pivot(index="model", columns="variable", values="std")
    cols = [c for c in piv_mean.columns if c != "macro"] + (["macro"] if "macro" in piv_mean.columns else [])
    md.append("| model | " + " | ".join(cols) + " |")
    md.append("|" + "---|" * (len(cols) + 1))
    for m in MODEL_ORDER:
        if m not in piv_mean.index:
            continue
        cells = []
        for v in cols:
            mu = piv_mean.loc[m, v]
            sd = piv_std.loc[m, v]
            if pd.isna(sd):
                cells.append(f"{mu:.3g}")
            else:
                cells.append(f"{mu:.3g} ± {sd:.2g}")
        md.append(f"| {m} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("## Pairwise Wilcoxon on macro-F1 across seeds (higher is better)")
    md.append("")
    if wlc.empty:
        md.append("_(no pair reached n>=3 seeds; Wilcoxon skipped)_")
    else:
        usable = wlc[wlc["p_value"].notna()]
        if not usable.empty:
            md.append("| variable | model_a | model_b | n | p |")
            md.append("|----------|---------|---------|---|---|")
            for _, w in usable.sort_values(["variable", "p_value"]).iterrows():
                star = " **" if w["p_value"] < 0.05 else ""
                md.append(f"| {w['variable']} | {w['model_a']} | {w['model_b']} | {w['n']} | {w['p_value']:.3g}{star} |")
    md.append("")

    md.append("## Figures")
    md.append("")
    for f in sorted(fig_dir.glob("phase2_*.png")):
        md.append(f"- ![{f.name}]({f.name})")
    md.append("")
    md.append(f"Generated from {len(runs)} classification run directories.")

    out = fig_dir / "SUMMARY_phase2.md"
    out.write_text("\n".join(md) + "\n")
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outputs", type=Path, default=REPO / "outputs",
                        help="Root directory containing model_era5_regression_* folders")
    parser.add_argument("--figures", type=Path, default=REPO / "outputs" / "figures",
                        help="Where to write figures and SUMMARY.md")
    parser.add_argument("--since", default="2026-06-23",
                        help="Ignore runs older than this date (default: 2026-06-23 — start of production matrix)")
    parser.add_argument("--coords", type=Path,
                        default=REPO / "preprocessing" / "cache" / "era5" / "coords.npy",
                        help="Path to coords.npy for geographic plots")
    args = parser.parse_args()

    args.figures.mkdir(parents=True, exist_ok=True)

    print(f"Discovering runs in {args.outputs} (since {args.since})...")
    runs = discover_runs(args.outputs, args.since)
    runs_cls = discover_runs_classification(args.outputs, args.since)

    if runs.empty and runs_cls.empty:
        print("No completed runs found.")
        return 1

    if not runs_cls.empty:
        print(f"Found {len(runs_cls)} classification runs:")
        for m, n in runs_cls.groupby("model")["seed"].count().items():
            print(f"  {m}: {n} seed(s)")

        cls_metrics = load_metrics_classification(runs_cls)
        cls_summary = summary_table_classification(cls_metrics, metric="f1")
        print("\nPhase 2 F1 summary:")
        print(cls_summary.to_string(index=False))

        # Wilcoxon on macro F1 only (headline).
        macro_only = cls_metrics[cls_metrics["variable"] == "macro"]
        cls_wlc = wilcoxon_pairs_classification(macro_only, metric="f1")
        if not cls_wlc.empty:
            usable = cls_wlc[cls_wlc["p_value"].notna()]
            if not usable.empty:
                print("\nPhase 2 macro-F1 Wilcoxon:")
                print(usable.to_string(index=False))

        print("\nWriting Phase 2 figures...")
        plot_f1_heatmap(cls_summary, args.figures)
        plot_classification_variance(cls_metrics, args.figures)
        if args.coords.exists():
            coords = np.load(args.coords)
            # Use whatever label variables appear in the first run
            label_vars_seen = list(cls_metrics[cls_metrics["variable"] != "macro"]["variable"].unique())
            for i, v in enumerate(label_vars_seen[:4]):
                plot_classification_geographic(runs_cls, args.figures, coords, v, i)

        write_summary_md_classification(runs_cls, cls_summary, cls_wlc, args.figures)

    if runs.empty:
        # No regression runs to summarise (Phase 2 only).
        print("\nDone (classification only).")
        return 0

    print(f"Found {len(runs)} completed runs:")
    for m, n in runs.groupby("model")["seed"].count().items():
        print(f"  {m}: {n} seed(s)")

    print("\nLoading metrics...")
    metrics = load_metrics(runs)

    print("Summary table...")
    summary = summary_table(metrics)
    print(summary.to_string(index=False))

    print("\nPairwise Wilcoxon tests...")
    wlc = wilcoxon_pairs(metrics)
    if not wlc.empty:
        usable = wlc[wlc["p_value"].notna()]
        if not usable.empty:
            print(usable.to_string(index=False))
        else:
            print("(All pairs n<3; skipped.)")

    print("\nWriting figures...")
    plot_mae_heatmap(summary, args.figures)
    plot_variance_box(metrics, args.figures)
    plot_nrmse_horizon(runs, args.figures)
    if args.coords.exists():
        coords = np.load(args.coords)
        for var in ["t2m", "ws10", "sp", "rh"]:
            plot_geographic(runs, args.figures, coords, var=var)
    else:
        print(f"  (skipping geographic plots — coords.npy not at {args.coords})")

    print("\nWriting markdown report...")
    write_summary_md(runs, summary, wlc, args.figures)

    print(f"\nDone. See {args.figures}/SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
