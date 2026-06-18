"""
Load multiple summary.json files and print a formatted cross-model comparison table.

Accepts one or more output folder paths (or a glob pattern). Useful during Phase 3
for assembling the final results table without opening individual JSON files.

Usage:
  # All runs in outputs/
  python scripts/compare_results.py outputs/

  # Specific pattern
  python scripts/compare_results.py "outputs/*era5*regression*"

  # Export to CSV
  python scripts/compare_results.py outputs/ --format csv --out results.csv

  # Wilcoxon significance tests vs baselines (per report_var)
  python scripts/compare_results.py outputs/ --wilcoxon

  # NRMSE horizon-decay + MAE bar / box plots (per report_var)
  python scripts/compare_results.py outputs/ --plot
"""

import argparse
import csv
import glob
import json
import sys
import warnings
from pathlib import Path

# Ensure project root is on the path so evaluation.stats is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

# Primary metrics shown in the comparison table (in display order)
REGRESSION_METRICS = [
    "per_variable.t2m.mae_mean", "per_variable.t2m.mae_std",
    "per_variable.t2m.vpt_mean",
    "params_M", "inference_ms_per_sample_mean",
]
CLASSIFICATION_METRICS = ["f1_mean", "f1_std", "recall_mean", "precision_mean", "auc_roc_mean", "params_M"]

BASELINES = {"agcrn", "clcrn"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare WEATHER-FLOW experiment results")
    p.add_argument("paths", nargs="+", help="Output folders or glob patterns to compare")
    p.add_argument("--format", choices=["table", "csv"], default="table",
                   help="Output format (default: table)")
    p.add_argument("--out", default=None,
                   help="Output file path (CSV format only). Prints to stdout if omitted.")
    p.add_argument("--task", choices=["regression", "classification", "auto"],
                   default="auto", help="Task type for metric selection (default: auto-detect)")
    p.add_argument("--wilcoxon", action="store_true",
                   help="Run Wilcoxon signed-rank tests vs baselines (AGCRN, CLCRN)")
    p.add_argument("--plot", action="store_true",
                   help="Generate MAE bar, box, and NRMSE horizon-decay plots per report_var")
    return p.parse_args()


def find_summaries(paths: list) -> list:
    """Resolve folder paths and glob patterns to a list of summary.json paths."""
    found = []
    for pattern in paths:
        for match in glob.glob(str(pattern), recursive=True):
            p = Path(match)
            if p.is_dir():
                s = p / "summary.json"
                if s.exists():
                    found.append(s)
            elif p.name == "summary.json" and p.exists():
                found.append(p)
    return sorted(set(found))


def load_summary(path: Path) -> dict:
    """Load and return summary.json, injecting the folder name as 'run_label'."""
    with open(path) as f:
        data = json.load(f)
    data["run_label"] = path.parent.name
    return data


def _get_nested(d: dict, dotted_key: str, default=None):
    """Fetch a value from a nested dict using dot-separated keys."""
    parts = dotted_key.split(".")
    cur = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _fmt(val, precision: int = 4) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.{precision}f}"
    except (TypeError, ValueError):
        return str(val)


def build_table(summaries: list, task: str) -> Table:
    metrics = REGRESSION_METRICS if task == "regression" else CLASSIFICATION_METRICS

    table = Table(show_header=True, header_style="bold cyan", title="WEATHER-FLOW Results")
    table.add_column("Run", style="dim", no_wrap=True)
    table.add_column("Model")
    table.add_column("Dataset")
    table.add_column("Task")
    for m in metrics:
        label = m.split(".")[-1].replace("_mean", "").replace("_", " ").title()
        if "." in m:
            label = f"{m.split('.')[1]}.{label}"
        table.add_column(label, justify="right")

    for s in summaries:
        row = [
            s.get("run_label", "?")[:40],
            s.get("model", "?"),
            s.get("dataset", "?"),
            s.get("task", "?"),
        ]
        for m in metrics:
            val = _get_nested(s, m)
            if m.endswith("_mean") and val is not None:
                std_key = m.replace("_mean", "_std")
                std = _get_nested(s, std_key)
                cell = f"{_fmt(val)} ± {_fmt(std)}" if std is not None else _fmt(val)
                row.append(cell)
            else:
                row.append(_fmt(val, 3))
        table.add_row(*row)

    return table


def export_csv(summaries: list, output_path) -> None:
    if not summaries:
        return
    all_keys = sorted({k for s in summaries for k in s.keys()})

    f = open(output_path, "w", newline="") if output_path else sys.stdout
    try:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k, "") for k in all_keys})
    finally:
        if output_path:
            f.close()

    if output_path:
        console.print(f"CSV written to [bold]{output_path}[/bold]")


# ---------------------------------------------------------------------------
# Wilcoxon significance tests
# ---------------------------------------------------------------------------

def run_wilcoxon(summaries: list, task: str) -> None:
    """
    For each experimental model vs each baseline (AGCRN, CLCRN), run a
    Wilcoxon signed-rank test on per-seed MAE (regression) or F1 (classification)
    for every report_var. Print a p-value table.
    """
    from evaluation.stats import wilcoxon_test  # noqa: PLC0415

    baselines = [s for s in summaries if s.get("model") in BASELINES]
    experiments = [s for s in summaries if s.get("model") not in BASELINES]

    if not baselines:
        console.print("[yellow]No baseline (agcrn/clcrn) summaries found — skipping Wilcoxon.[/yellow]")
        return
    if not experiments:
        console.print("[yellow]No experimental model summaries found — skipping Wilcoxon.[/yellow]")
        return

    if task == "regression":
        metric_key = "mae"
    else:
        metric_key = "f1"

    # Collect per-seed arrays: summary["per_seed"]["mae"]["t2m"] = [s42, s43, ...]
    def get_seeds(s, var=None):
        per_seed = s.get("per_seed", {})
        if metric_key not in per_seed:
            return None
        if var is None:
            # classification: flat list
            return np.array(per_seed[metric_key], dtype=float) if isinstance(per_seed[metric_key], list) else None
        return np.array(per_seed[metric_key].get(var, []), dtype=float)

    # Determine report_vars from first regression summary
    report_vars = None
    if task == "regression":
        for s in summaries:
            rv = s.get("report_vars")
            if rv:
                report_vars = rv
                break
    if report_vars is None:
        report_vars = ["all"]

    table = Table(show_header=True, header_style="bold cyan",
                  title=f"Wilcoxon p-values ({metric_key})")
    table.add_column("Model", style="dim")
    table.add_column("Variable")
    for b in baselines:
        table.add_column(f"vs {b.get('model', '?')}", justify="right")

    for exp in experiments:
        exp_model = exp.get("model", "?")
        for var in report_vars:
            exp_arr = get_seeds(exp, var if task == "regression" else None)
            row = [exp_model, var]
            for b in baselines:
                b_arr = get_seeds(b, var if task == "regression" else None)
                if exp_arr is None or b_arr is None or len(exp_arr) < 2 or len(b_arr) < 2:
                    row.append("—")
                    continue
                n = min(len(exp_arr), len(b_arr))
                if n < 2:
                    row.append("n<2")
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, p = wilcoxon_test(exp_arr[:n], b_arr[:n])
                sig = "*" if p < 0.05 else ""
                row.append(f"{p:.4f}{sig}")
            table.add_row(*row)

    console.print(table)
    console.print("[dim]* p < 0.05[/dim]")


# ---------------------------------------------------------------------------
# Cross-model plots
# ---------------------------------------------------------------------------

def run_plots(summaries: list, task: str) -> None:
    """
    Generate per-report_var figures:
      - Bar chart: MAE mean ± std per model, grouped by family
      - Box plot: seed distribution of MAE per model
      - NRMSE horizon-decay: one line per model, VPT ε reference line
    Saved to plots/ under the first summary's parent directory.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
    except ImportError:
        console.print("[red]matplotlib not installed — run `pip install matplotlib` or use HPC.[/red]")
        return

    if task != "regression":
        console.print("[yellow]--plot currently only supports regression summaries.[/yellow]")
        return

    # Infer output dir from first summary path
    base_dir = Path(summaries[0].get("run_label", ".")).parent
    plot_dir = base_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Determine report_vars
    report_vars = None
    for s in summaries:
        rv = s.get("report_vars")
        if rv:
            report_vars = rv
            break
    if not report_vars:
        console.print("[yellow]No report_vars found in summaries — skipping plots.[/yellow]")
        return

    models = [s.get("model", s.get("run_label", "?")) for s in summaries]
    vpt_epsilon = summaries[0].get("vpt_epsilon", 0.3)

    for var in report_vars:
        mae_means = []
        mae_stds = []
        mae_per_seed = []
        nrmse_curves = []
        valid_models = []

        for s, m in zip(summaries, models):
            pv = s.get("per_variable", {}).get(var, {})
            mm = pv.get("mae_mean")
            ms = pv.get("mae_std", 0.0)
            seeds_arr = s.get("per_seed", {}).get("mae", {}).get(var)
            nrmse = s.get("nrmse_per_horizon", {}).get(var, {})
            curve = nrmse.get("mean") if isinstance(nrmse, dict) else None

            if mm is None:
                continue
            mae_means.append(mm)
            mae_stds.append(ms)
            mae_per_seed.append(np.array(seeds_arr, dtype=float) if seeds_arr else np.array([mm]))
            nrmse_curves.append(curve)
            valid_models.append(m)

        if not valid_models:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Variable: {var}", fontsize=14)

        # Bar chart
        ax = axes[0]
        x = np.arange(len(valid_models))
        ax.bar(x, mae_means, yerr=mae_stds, capsize=4, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_models, rotation=30, ha="right")
        ax.set_ylabel("MAE (physical units)")
        ax.set_title("MAE mean ± std")

        # Box plot
        ax = axes[1]
        ax.boxplot(mae_per_seed, labels=valid_models, vert=True)
        ax.set_xticklabels(valid_models, rotation=30, ha="right")
        ax.set_ylabel("MAE (physical units)")
        ax.set_title("Seed distribution")

        # NRMSE horizon-decay
        ax = axes[2]
        H = max((len(c) for c in nrmse_curves if c is not None), default=0)
        for m_name, curve in zip(valid_models, nrmse_curves):
            if curve is not None:
                ax.plot(range(1, len(curve) + 1), curve, marker="o", markersize=3, label=m_name)
        if H > 0:
            ax.axhline(y=vpt_epsilon, color="gray", linestyle="--", label=f"ε={vpt_epsilon}")
        ax.set_xlabel("Horizon step")
        ax.set_ylabel("NRMSE")
        ax.set_title("NRMSE horizon-decay")
        ax.legend(fontsize=7)

        plt.tight_layout()
        out_path = plot_dir / f"{var}_comparison.png"
        plt.savefig(str(out_path), dpi=150)
        plt.close(fig)
        console.print(f"  Saved {out_path}")

    console.print(f"Plots written to [bold]{plot_dir}[/bold]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    summary_paths = find_summaries(args.paths)
    if not summary_paths:
        console.print("[red]No summary.json files found in the specified paths.[/red]")
        console.print("Run an experiment first: make run MODEL=agcrn DATASET=era5 TASK=regression")
        sys.exit(1)

    summaries = [load_summary(p) for p in summary_paths]
    console.print(f"Found [bold]{len(summaries)}[/bold] result(s)\n")

    if args.task == "auto":
        tasks = {s.get("task", "regression") for s in summaries}
        task = tasks.pop() if len(tasks) == 1 else "regression"
    else:
        task = args.task

    if args.format == "table":
        console.print(build_table(summaries, task))
    else:
        export_csv(summaries, args.out)

    if args.wilcoxon:
        run_wilcoxon(summaries, task)

    if args.plot:
        run_plots(summaries, task)


if __name__ == "__main__":
    main()
