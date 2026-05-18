"""
Adapter (method) for USTD — training and prediction logic.

Like DiffSTG, forward() and predict() differ significantly:

  forward() — training: shortened denoising chain, fast diffusion loss.

  predict() — evaluation: full denoising chain producing the point prediction
              (and optionally an uncertainty estimate that can be logged
              separately). Called inside torch.no_grad().

Both paths use shared denoising utilities in modules/denoising.py.
"""

import torch
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.diffusion.ustd_model import USTDModel
from models.diffusion.modules.denoising import (
    precompute_diffusion_params,
    shortened_denoising_chain,
    full_denoising_chain,
)


class USTDAdapter(BaseAdapter):

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> USTDModel:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Training pass — diffusion loss via shortened denoising chain."""
        raise NotImplementedError

    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Evaluation pass — full denoising chain, no gradient."""
        raise NotImplementedError

    def get_param_count(self) -> int:
        raise NotImplementedError
