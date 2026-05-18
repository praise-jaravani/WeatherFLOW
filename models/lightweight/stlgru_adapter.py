"""
Adapter (method) for STLGRU — training and prediction logic.

STLGRU is a deterministic recurrent model; forward() and predict() use the same
underlying computation path. predict() runs inside torch.no_grad() as required
by the BaseAdapter contract.
"""

import torch
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.lightweight.stlgru_model import STLGRUModel


class STLGRUAdapter(BaseAdapter):

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> STLGRUModel:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_param_count(self) -> int:
        raise NotImplementedError
