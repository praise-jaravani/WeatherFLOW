"""
Model wrapper for Lite-STGNN (Lightweight Spatio-Temporal Graph Neural Network).

Cloned repo: models/lightweight/Lite-STGNN/
Key entry point: Lite-STGNN/src/ — contains the main model architecture.

Lite-STGNN prioritises parameter efficiency and fast inference. Wraps the
cloned model as a clean torch.nn.Module.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

REPO_PATH = Path(__file__).parent / "Lite-STGNN"


class LiteSTGNNModel(nn.Module):
    """Thin wrapper around the cloned Lite-STGNN nn.Module."""

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
