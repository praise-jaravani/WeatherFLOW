"""
Adapter (method) for Lite-STGNN — training and prediction logic.

Lite-STGNN is a deterministic model; forward() and predict() use the same
computation path. predict() runs inside torch.no_grad().
"""

import torch
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.lightweight.litestgnn_model import LiteSTGNNModel


class LiteSTGNNAdapter(BaseAdapter):

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> LiteSTGNNModel:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_param_count(self) -> int:
        raise NotImplementedError
