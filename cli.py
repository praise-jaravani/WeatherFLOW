"""
Single entry point for all WEATHER-FLOW experiments.

Usage:
  python cli.py --model <name> --dataset <saws|era5|both> --task <regression|classification> [options]

Available models: agcrn, clcrn, mtgode, climode, stlgru, litestgnn, diffstg, ustd

Optional flags:
  --runs N             Number of seeds to run (default 10, seeds 42–51)
  --horizon H          Forecast horizon in timesteps
  --config.<key> val   Override any YAML config value at runtime
  --no-cache           Force preprocessing to rerun even if cache is valid
  --mlflow-off         Disable MLflow tracking (useful for quick debug runs)

Display: rich progress bars showing nested outer (run 1/10) and inner
(epoch 1/N) loops simultaneously, with live loss and ETA. All output is
tee'd to outputs/<run_folder>/run_XX/train.log.
"""

import argparse
import sys
from pathlib import Path


ADAPTER_REGISTRY = {
    "agcrn":     "models.baselines.agcrn_adapter.AGCRNAdapter",
    "clcrn":     "models.baselines.clcrn_adapter.CLCRNAdapter",
    "mtgode":    "models.continuous_time.mtgode_adapter.MTGODEAdapter",
    "climode":   "models.continuous_time.climode_adapter.ClimODEAdapter",
    "stlgru":    "models.lightweight.stlgru_adapter.STLGRUAdapter",
    "litestgnn": "models.lightweight.litestgnn_adapter.LiteSTGNNAdapter",
    "diffstg":   "models.diffusion.diffstg_adapter.DiffSTGAdapter",
    "ustd":      "models.diffusion.ustd_adapter.USTDAdapter",
}


def parse_args() -> argparse.Namespace:
    raise NotImplementedError


def load_config(model_name: str, overrides: dict) -> dict:
    """Load experiments/configs/<model>.yaml and apply CLI overrides."""
    raise NotImplementedError


def load_adapter(model_name: str):
    """Dynamically import and return the adapter class for the given model name."""
    raise NotImplementedError


def make_output_dir(model: str, dataset: str, task: str) -> Path:
    """Create and return a timestamped output directory."""
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
