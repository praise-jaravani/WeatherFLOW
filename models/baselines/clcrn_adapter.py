"""
Adapter (method) for CLCRN — training and prediction logic.

Drives CLCRNModel through the shared pipeline interface defined in BaseAdapter.
CLCRN is a recurrent GNN baseline; forward() and predict() use the same
underlying model path.
"""

import torch
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.baselines.clcrn_model import CLCRNModel


class CLCRNAdapter(BaseAdapter):

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> CLCRNModel:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_param_count(self) -> int:
        raise NotImplementedError
