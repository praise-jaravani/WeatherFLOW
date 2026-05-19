"""
Adapter (method) for MTGODE — training and prediction logic.

MTGODE uses two nested ODE integrations (both euler-based) and learns its own
adaptive adjacency — the static adj from preprocessing is ignored.

forward() and predict() call the same model path. The only distinction is:
  - forward(): model in train mode (dropout active, adjoint backprop)
  - predict(): model in eval mode, inside torch.no_grad()
"""

import torch
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.continuous_time.mtgode_model import MTGODEModel


class MTGODEAdapter(BaseAdapter):

    def __init__(self):
        self.model: MTGODEModel | None = None

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> MTGODEModel:
        """
        Args:
            config: full experiment config dict. Must include:
                      config['meta']['n_vars']   — number of input/output variables
                      config['data']['horizon']  — forecast horizon H
                      config['data']['input_len']— historical window T
            graph:  (N, N) static adjacency — not used (MTGODE learns its own).

        Returns:
            Initialised MTGODEModel.
        """
        N = graph.shape[0]

        merged = {
            **config.get("model", {}),
            "model": config.get("model", {}),
            "ode":   config.get("ode", {}),
            "input_dim":  config["meta"]["n_vars"],
            "output_dim": config["meta"]["n_vars"],
            "horizon":    config["data"]["horizon"],
            "input_len":  config["data"]["input_len"],
        }

        self.model = MTGODEModel(merged, num_nodes=N)
        return self.model

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass — train mode, gradient enabled.

        Args:
            x:   (B, T, N, d)
            adj: (N, N) — ignored

        Returns:
            (B, H, N, d)
        """
        self.model.train()
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
