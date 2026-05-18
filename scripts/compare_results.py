"""
Load multiple summary.json files and print a formatted cross-model comparison table.

Accepts a list of output folder paths or a glob pattern. Useful during Phase 3
for assembling the final results table without opening individual JSON files.

Usage:
  # Compare specific runs
  python scripts/compare_results.py outputs/2026-*__saws__regression/

  # Specify output format
  python scripts/compare_results.py outputs/2026-*__saws__regression/ --format csv

Output can be rendered as a rich table in the terminal or exported as CSV.
"""

import argparse
import json
import csv
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def parse_args() -> argparse.Namespace:
    raise NotImplementedError


def load_summary(folder: Path) -> dict:
    """Load and return summary.json from a run output folder."""
    raise NotImplementedError


def build_comparison_table(summaries: list) -> Table:
    """Build a rich Table comparing metrics across all loaded summaries."""
    raise NotImplementedError


def export_csv(summaries: list, output_path: Path) -> None:
    """Write comparison data to a CSV file."""
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
