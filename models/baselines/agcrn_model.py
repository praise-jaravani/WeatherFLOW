"""
Model wrapper for AGCRN (Adaptive Graph Convolutional Recurrent Network).

Cloned repo: models/baselines/AGCRN/
Key entry point: AGCRN/model/AGCRN.py — class AGCRN(nn.Module)

Adds the cloned repo to sys.path and instantiates the AGCRN nn.Module
according to the experiment config and graph structure. Exposes a clean
torch.nn.Module that agcrn_adapter.py can call without knowing the repo's
internal import structure.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

REPO_PATH = Path(__file__).parent / "AGCRN"


class AGCRNModel(nn.Module):
    """Thin wrapper around the cloned AGCRN nn.Module."""

    def __init__(self, config: Dict[str, Any], num_nodes: int):
        """
        Args:
            config: model config dict (hidden_dim, num_layers, etc.).
            num_nodes: number of stations N (graph size).
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, N, d)
            adj: (N, N)

        Returns:
            (batch, H, N, d)
        """
        raise NotImplementedError
