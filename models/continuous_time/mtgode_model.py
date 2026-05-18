"""
Model wrapper for MTGODE (Multi-scale Temporal Graph ODE).

Cloned repo: models/continuous_time/MTGODE/
Key entry point: MTGODE/model.py — classes ODEFunc, ODEBlock, and the top-level
MTGODE model. Imports from MTGODE/layer.py (dilated_inception, mixprop, CGP,
graph_constructor).

Adds the cloned repo to sys.path and wraps the MTGODE model as a clean
torch.nn.Module. ODE solver settings (method, step_size, rtol, atol) are read
from the experiment config and passed to the model at construction time.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

REPO_PATH = Path(__file__).parent / "MTGODE"


class MTGODEModel(nn.Module):
    """Thin wrapper around the cloned MTGODE nn.Module."""

    def __init__(self, config: Dict[str, Any], num_nodes: int, adj: torch.Tensor):
        """
        Args:
            config: model config dict (hidden_dim, layers, ode_method, etc.).
            num_nodes: number of stations N.
            adj: static adjacency matrix of shape (N, N).
        """
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
