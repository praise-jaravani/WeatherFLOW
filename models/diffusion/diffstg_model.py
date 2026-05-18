"""
Model wrapper for DiffSTG (Diffusion-based Spatio-Temporal Graph model).

Cloned repo: models/diffusion/DiffSTG/
Key entry points:
  - DiffSTG/algorithm/diffstg/ — main model architecture and diffusion logic
  - DiffSTG/algorithm/dataset.py — data utilities (not used; pipeline provides data)

DiffSTG uses a conditional diffusion process over the spatio-temporal graph.
At training time a shortened denoising chain is used; at inference the full
chain runs for higher quality. See modules/denoising.py for shared schedule logic.

Wraps the cloned model as a clean torch.nn.Module.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

REPO_PATH = Path(__file__).parent / "DiffSTG"


class DiffSTGModel(nn.Module):
    """Thin wrapper around the cloned DiffSTG nn.Module."""

    def __init__(self, config: Dict[str, Any], num_nodes: int, adj: torch.Tensor):
        super().__init__()
        raise NotImplementedError

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Denoising network forward pass: predicts noise given noisy input at step t.

        Args:
            x_noisy: noisy prediction tensor.
            t: diffusion timestep tensor.
            condition: historical context tensor of shape (batch, T, N, d).

        Returns:
            Predicted noise tensor.
        """
        raise NotImplementedError
