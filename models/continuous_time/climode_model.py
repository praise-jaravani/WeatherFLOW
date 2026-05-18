"""
Model wrapper for ClimODE (Climate and Weather Forecasting with Physics-informed Neural ODEs).

Cloned repo: models/continuous_time/ClimODE/
Key entry points:
  - ClimODE/model_function.py — Optim_velocity and the core ODE dynamics
  - ClimODE/model_utils.py    — supporting utilities (emission model, etc.)
  - ClimODE/utils.py          — data and training helpers

ClimODE models atmospheric transport as an advection ODE driven by learned
velocity fields. Unlike MTGODE, ClimODE was designed for gridded ERA5 data.
The wrapper adapts its interface to accept (batch, T, N, d) station tensors.

Adds the cloned repo to sys.path and wraps the ClimODE model as a clean
torch.nn.Module.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

REPO_PATH = Path(__file__).parent / "ClimODE"


class ClimODEModel(nn.Module):
    """Thin wrapper around the cloned ClimODE nn.Module."""

    def __init__(self, config: Dict[str, Any], num_nodes: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, N, d)

        Returns:
            (batch, H, N, d)
        """
        raise NotImplementedError
