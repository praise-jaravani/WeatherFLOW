"""
Convenience script: run AGCRN and CLCRN back-to-back on a specified dataset and task.

Used during the Setup phase to establish baseline performance before the
individual workstreams begin. Wraps two sequential calls to cli.py with
consistent config.

Usage:
  python scripts/run_baselines.py --dataset saws --task regression
  python scripts/run_baselines.py --dataset era5 --task classification
"""

import argparse
import subprocess
import sys
from pathlib import Path

CLI = Path(__file__).parent.parent / "cli.py"


def parse_args() -> argparse.Namespace:
    raise NotImplementedError


def run_model(model: str, dataset: str, task: str, extra_args: list) -> int:
    """Run cli.py for a single model. Returns the subprocess return code."""
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
