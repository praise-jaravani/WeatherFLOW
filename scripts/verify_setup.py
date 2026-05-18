"""
Verification script — checks all pipeline components and prints a pass/fail table.

Run from the project root:
  python scripts/verify_setup.py

Every item must pass before implementation begins. This is the gate condition
for moving from setup into active development.

Checks performed:
  1. All 8 cloned repositories present and non-empty
  2. All stub Python modules importable from project root without errors
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

MODULE_CHECKS = [
    "preprocessing.saws_parser",
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
    "models.shared_modules.graph_utils",
    "models.baselines.agcrn_adapter",
    "models.baselines.agcrn_model",
    "models.baselines.clcrn_adapter",
    "models.baselines.clcrn_model",
    "models.continuous_time.mtgode_adapter",
    "models.continuous_time.mtgode_model",
    "models.continuous_time.climode_adapter",
    "models.continuous_time.climode_model",
    "models.continuous_time.modules.ode_integrator",
    "models.lightweight.stlgru_adapter",
    "models.lightweight.stlgru_model",
    "models.lightweight.litestgnn_adapter",
    "models.lightweight.litestgnn_model",
    "models.diffusion.diffstg_adapter",
    "models.diffusion.diffstg_model",
    "models.diffusion.ustd_adapter",
    "models.diffusion.ustd_model",
    "models.diffusion.modules.denoising",
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


def check_repos() -> list:
    raise NotImplementedError


def check_modules() -> list:
    raise NotImplementedError


def check_configs() -> list:
    raise NotImplementedError


def check_dirs() -> list:
    raise NotImplementedError


def check_files() -> list:
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
