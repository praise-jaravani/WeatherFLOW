"""
Model wrapper for STLGRU (Spatio-Temporal Lightweight GRU).

Cloned repo: models/lightweight/STLGRU/
Key entry points:
  - STLGRU/model.py   — main GRU-based STGNN architecture
  - STLGRU/layer.py   — spatial graph convolution layers
  - STLGRU/modules.py — reusable GRU cell modules

STLGRU is designed for efficiency — low parameter count with competitive
accuracy. Wraps the cloned model as a clean torch.nn.Module.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

REPO_PATH = Path(__file__).parent / "STLGRU"


class STLGRUModel(nn.Module):
    """Thin wrapper around the cloned STLGRU nn.Module."""

    def __init__(self, config: Dict[str, Any], num_nodes: int, adj: torch.Tensor):
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
