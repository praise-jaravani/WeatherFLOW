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
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# Primary metrics shown in the table (in display order)
REGRESSION_METRICS = ["mae_mean", "mae_std", "rmse_mean", "vpt", "params_M", "inference_ms_per_sample_mean"]
CLASSIFICATION_METRICS = ["f1_mean", "f1_std", "recall_mean", "precision_mean", "auc_roc_mean", "params_M"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare WEATHER-FLOW experiment results")
    p.add_argument("paths", nargs="+", help="Output folders or glob patterns to compare")
    p.add_argument("--format", choices=["table", "csv"], default="table",
                   help="Output format (default: table)")
    p.add_argument("--out", default=None,
                   help="Output file path (CSV format only). Prints to stdout if omitted.")
    p.add_argument("--task", choices=["regression", "classification", "auto"],
                   default="auto", help="Task type for metric selection (default: auto-detect)")
    return p.parse_args()


def find_summaries(paths: list) -> list[Path]:
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
        table.add_column(m.replace("_mean", "").replace("_", " ").title(), justify="right")

    for s in summaries:
        row = [
            s.get("run_label", "?")[:40],
            s.get("model", "?"),
            s.get("dataset", "?"),
            s.get("task", "?"),
        ]
        for m in metrics:
            val = s.get(m)
            if m.endswith("_mean") and val is not None:
                std_key = m.replace("_mean", "_std")
                std = s.get(std_key)
                cell = f"{_fmt(val)} ± {_fmt(std)}" if std is not None else _fmt(val)
                row.append(cell)
            else:
                row.append(_fmt(val, 3))
        table.add_row(*row)

    return table


def export_csv(summaries: list, output_path: str | None) -> None:
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


if __name__ == "__main__":
    main()
