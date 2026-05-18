"""
Model wrapper for USTD (Uncertainty-aware Spatio-Temporal Diffusion).

Cloned repo: models/diffusion/USTD/
Key entry points:
  - USTD/models/stdiffusion/ — main STDiffusion model
  - USTD/models/base_model.py — base class for diffusion models
  - USTD/options/           — config system (not used; pipeline YAML takes over)
  - USTD/utils/             — supporting utilities

USTD extends diffusion-based STGNNs with explicit uncertainty quantification.
The full denoising chain at inference produces both a point prediction and an
uncertainty estimate. Wraps the cloned model as a clean torch.nn.Module.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple

REPO_PATH = Path(__file__).parent / "USTD"


class USTDModel(nn.Module):
    """Thin wrapper around the cloned USTD nn.Module."""

    def __init__(self, config: Dict[str, Any], num_nodes: int, adj: torch.Tensor):
        super().__init__()
        raise NotImplementedError

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Denoising network forward pass.

        Args:
            x_noisy: noisy prediction tensor.
            t: diffusion timestep tensor.
            condition: historical context tensor (batch, T, N, d).

        Returns:
            Predicted noise tensor.
        """
        raise NotImplementedError
