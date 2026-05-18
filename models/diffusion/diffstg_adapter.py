"""
Adapter (method) for DiffSTG — training and prediction logic.

This is the primary example of why forward() and predict() are separate methods:

  forward() — training: computes diffusion loss using a shortened denoising chain
              (fast, approximate). Gradients flow through the denoising network.

  predict() — evaluation: runs the full denoising chain from pure Gaussian noise
              conditioned on the input, producing the highest-quality prediction.
              Called inside torch.no_grad().

Both paths use the shared denoising utilities in modules/denoising.py.
"""

import torch
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.diffusion.diffstg_model import DiffSTGModel
from models.diffusion.modules.denoising import (
    precompute_diffusion_params,
    shortened_denoising_chain,
    full_denoising_chain,
)


class DiffSTGAdapter(BaseAdapter):

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> DiffSTGModel:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Training pass — diffusion loss via shortened denoising chain."""
        raise NotImplementedError

    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Evaluation pass — full denoising chain, no gradient."""
        raise NotImplementedError

    def get_param_count(self) -> int:
        raise NotImplementedError
