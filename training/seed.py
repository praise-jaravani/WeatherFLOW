"""
Reproducibility utility: fix all random seeds in one call.

Sets seeds for Python's random module, NumPy, PyTorch CPU, and PyTorch CUDA
simultaneously. Called at the start of every run. The seed value is logged to
MLflow and stored in the run's metrics.json.

Seeds for the 10 runs per experiment: 42 through 51 inclusive.
"""

import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch (CPU + CUDA) random seeds."""
    raise NotImplementedError
