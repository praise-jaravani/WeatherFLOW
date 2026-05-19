"""
Convenience script: run AGCRN and CLCRN back-to-back on a specified dataset and task.

Used during the Setup phase to establish baseline performance before the individual
workstreams begin. Wraps two sequential calls to cli.py with consistent config.

Usage:
  python scripts/run_baselines.py --dataset era5 --task regression
  python scripts/run_baselines.py --dataset saws --task classification --runs 1
"""

import argparse
import subprocess
import sys
from pathlib import Path

CLI = Path(__file__).parent.parent / "cli.py"
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run AGCRN and CLCRN back-to-back"
    )
    p.add_argument("--dataset", required=True, choices=["saws", "era5", "both"])
    p.add_argument("--task",    required=True, choices=["regression", "classification"])
    p.add_argument("--runs",    type=int, default=10,
                   help="Number of seeds per model (default 10)")
    p.add_argument("--mlflow-off", action="store_true",
                   help="Disable MLflow tracking")
    return p.parse_args()


def run_model(model: str, args: argparse.Namespace) -> int:
    cmd = [
        PYTHON, str(CLI),
        "--model",   model,
        "--dataset", args.dataset,
        "--task",    args.task,
        "--runs",    str(args.runs),
    ]
    if args.mlflow_off:
        cmd.append("--mlflow-off")

    print(f"\n{'='*60}")
    print(f"  Running {model.upper()} | {args.dataset} | {args.task} | {args.runs} seed(s)")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    args = parse_args()
    exit_codes = []

    for model in ["agcrn", "clcrn"]:
        code = run_model(model, args)
        exit_codes.append((model, code))

    print("\n" + "="*60)
    print("  Baselines complete")
    print("="*60)
    for model, code in exit_codes:
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"  {model.upper()}: {status}")

    if any(c != 0 for _, c in exit_codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
