"""
Adapter (method) for AGCRN — training and prediction logic.

AGCRN is a recurrent GNN that learns its own adaptive adjacency via node
embeddings. Training and inference are identical passes (no teacher forcing
used, no solver-tolerance difference), so forward() and predict() call the
same model path — predict() just wraps it in no_grad + eval mode.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.baselines.agcrn_model import AGCRNModel


class AGCRNAdapter(BaseAdapter):

    def __init__(self):
        self.model: AGCRNModel | None = None

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> AGCRNModel:
        """
        Args:
            config: full experiment config dict.
            graph:  (N, N) static adjacency — not used by AGCRN (learns its own).

        Returns:
            Initialised AGCRNModel.
        """
        N = graph.shape[0]
        model_cfg = config["model"]
        model_cfg["input_dim"]  = config["meta"]["n_vars"]
        model_cfg["output_dim"] = config["meta"]["n_vars"]
        model_cfg["horizon"]    = config["data"]["horizon"]

        self.model = AGCRNModel(model_cfg, num_nodes=N)
        return self.model

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            x:   (B, T, N, d)
            adj: (N, N)  — ignored by AGCRN

        Returns:
            (B, H, N, d)
        """
        return self.model(x)

    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Evaluation forward pass — eval mode, no gradient.

        Args:
            x:   (B, T, N, d)
            adj: (N, N)

        Returns:
            (B, H, N, d)
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
