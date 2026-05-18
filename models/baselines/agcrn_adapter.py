"""
Adapter (method) for AGCRN — training and prediction logic.

Drives AGCRNModel through the shared pipeline interface defined in BaseAdapter.
AGCRN is a recurrent GNN baseline; its training and inference forward passes
are identical, so forward() and predict() call the same underlying model path.
"""

import torch
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.baselines.agcrn_model import AGCRNModel


class AGCRNAdapter(BaseAdapter):

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> AGCRNModel:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_param_count(self) -> int:
        raise NotImplementedError
