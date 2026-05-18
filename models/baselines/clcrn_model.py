"""
Model wrapper for CLCRN (Chebyshev-based LSTM Convolutional Recurrent Network).

Cloned repo: models/baselines/CLCRN/
Key entry point: CLCRN/model/ — contains the CLCRN nn.Module and supporting layers.

Adds the cloned repo to sys.path and wraps the CLCRN model as a clean
torch.nn.Module with the standard (batch, T, N, d) → (batch, H, N, d) interface.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

REPO_PATH = Path(__file__).parent / "CLCRN"


class CLCRNModel(nn.Module):
    """Thin wrapper around the cloned CLCRN nn.Module."""

    def __init__(self, config: Dict[str, Any], num_nodes: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
