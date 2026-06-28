"""
Adapter (method) for Lite-STGNN — training and prediction logic.

Lite-STGNN is a deterministic model; forward() and predict() use the same
computation path. predict() runs inside torch.no_grad().

Lite-STGNN learns its own low-rank TopK adjacency from node embeddings, so the
static adjacency from preprocessing is accepted for interface consistency but
ignored internally (only its shape determines N).
"""

import torch
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.lightweight.litestgnn_model import LiteSTGNNModel


class LiteSTGNNAdapter(BaseAdapter):

    def __init__(self):
        self.model: LiteSTGNNModel | None = None

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> LiteSTGNNModel:
        """
        Args:
            config: full experiment config dict.
            graph:  (N, N) static adjacency — not used by Lite-STGNN.

        Returns:
            Initialised LiteSTGNNModel.
        """
        N = graph.shape[0]
        model_cfg = dict(config["model"])  # copy — don't mutate caller's dict
        model_cfg["input_dim"]  = config["meta"]["n_vars"]
        model_cfg["output_dim"] = config["meta"]["n_vars"]
        model_cfg["horizon"]    = config["data"]["horizon"]
        model_cfg["input_len"]  = config["data"]["input_len"]

        self.model = LiteSTGNNModel(model_cfg, num_nodes=N, adj=graph)
        return self.model

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            x:   (B, T, N, d)
            adj: (N, N) — ignored (Lite-STGNN learns its own adjacency)

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
