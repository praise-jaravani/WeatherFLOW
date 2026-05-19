"""
Verification script — checks all pipeline components and prints a pass/fail table.

Run from the project root:
  python scripts/verify_setup.py

Every item must pass before implementation begins. This is the gate condition
for moving from setup into active development.

Checks performed:
  1. All 8 cloned repositories present and non-empty
  2. All Python modules importable from project root without errors
  3. All 8 model YAML config files present and valid YAML
  4. tracking/ directory exists
  5. data/saws/raw/ and data/era5/raw/ directories exist
  6. preprocessing/cache/ directory exists
  7. container.def present in project root
"""

import sys
import importlib
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()

REPO_CHECKS = [
    ("AGCRN",      PROJECT_ROOT / "models/baselines/AGCRN"),
    ("CLCRN",      PROJECT_ROOT / "models/baselines/CLCRN"),
    ("MTGODE",     PROJECT_ROOT / "models/continuous_time/MTGODE"),
    ("ClimODE",    PROJECT_ROOT / "models/continuous_time/ClimODE"),
    ("STLGRU",     PROJECT_ROOT / "models/lightweight/STLGRU"),
    ("Lite-STGNN", PROJECT_ROOT / "models/lightweight/Lite-STGNN"),
    ("DiffSTG",    PROJECT_ROOT / "models/diffusion/DiffSTG"),
    ("USTD",       PROJECT_ROOT / "models/diffusion/USTD"),
]

# Modules that must import cleanly (adapters for unimplemented models
# will show as import errors — that's expected for teammates' stubs)
MODULE_CHECKS = [
    "preprocessing.era5_parser",
    "preprocessing.impute",
    "preprocessing.split",
    "preprocessing.labels",
    "preprocessing.normalise",
    "preprocessing.graph",
    "preprocessing.dataset",
    "preprocessing.pipeline",
    "training.seed",
    "training.losses",
    "training.scheduler",
    "training.trainer",
    "evaluation.regression",
    "evaluation.classification",
    "evaluation.computational",
    "evaluation.stats",
    "models.base_adapter",
    "models.baselines.agcrn_adapter",
    "models.baselines.agcrn_model",
    "models.baselines.clcrn_adapter",
    "models.baselines.clcrn_model",
    "models.continuous_time.mtgode_adapter",
    "models.continuous_time.mtgode_model",
    "models.continuous_time.climode_adapter",
    "models.continuous_time.climode_model",
]

CONFIG_CHECKS = [
    PROJECT_ROOT / f"experiments/configs/{name}.yaml"
    for name in ["agcrn", "clcrn", "mtgode", "climode", "stlgru", "litestgnn", "diffstg", "ustd"]
]

DIR_CHECKS = [
    PROJECT_ROOT / "tracking",
    PROJECT_ROOT / "data/saws/raw",
    PROJECT_ROOT / "data/era5/raw",
    PROJECT_ROOT / "preprocessing/cache",
]

FILE_CHECKS = [
    PROJECT_ROOT / "container.def",
]


def _row(label: str, ok: bool, detail: str = "") -> tuple:
    status = "[bold green]PASS[/bold green]" if ok else "[bold red]FAIL[/bold red]"
    return label, status, detail


def check_repos() -> list:
    rows = []
    for name, path in REPO_CHECKS:
        exists = path.exists() and any(path.iterdir())
        rows.append(_row(f"repo: {name}", exists,
                         str(path.relative_to(PROJECT_ROOT)) if not exists else ""))
    return rows


def check_modules() -> list:
    rows = []
    for mod in MODULE_CHECKS:
        try:
            importlib.import_module(mod)
            rows.append(_row(f"import: {mod}", True))
        except Exception as e:
            rows.append(_row(f"import: {mod}", False, str(e)[:80]))
    return rows


def check_configs() -> list:
    rows = []
    for path in CONFIG_CHECKS:
        if not path.exists():
            rows.append(_row(f"config: {path.name}", False, "file missing"))
            continue
        try:
            with open(path) as f:
                yaml.safe_load(f)
            rows.append(_row(f"config: {path.name}", True))
        except Exception as e:
            rows.append(_row(f"config: {path.name}", False, str(e)[:80]))
    return rows


def check_dirs() -> list:
    return [
        _row(f"dir: {p.relative_to(PROJECT_ROOT)}", p.exists())
        for p in DIR_CHECKS
    ]


def check_files() -> list:
    return [
        _row(f"file: {p.relative_to(PROJECT_ROOT)}", p.exists())
        for p in FILE_CHECKS
    ]


def main():
    console.rule("[bold]WEATHER-FLOW setup verification[/bold]")

    all_rows = (
        check_repos()
        + check_modules()
        + check_configs()
        + check_dirs()
        + check_files()
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("Component", style="dim", width=52)
    table.add_column("Status", width=8, justify="center")
    table.add_column("Notes")

    n_pass = n_fail = 0
    for label, status, detail in all_rows:
        table.add_row(label, status, detail)
        if "PASS" in status:
            n_pass += 1
        else:
            n_fail += 1

    console.print(table)
    console.print(f"\n  [bold green]{n_pass} passed[/bold green]  "
                  f"[bold red]{n_fail} failed[/bold red]\n")

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
