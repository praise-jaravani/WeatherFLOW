"""
Adapter (method) for STLGRU — training and prediction logic.

STLGRU is a deterministic recurrent model; forward() and predict() use the same
underlying computation path. predict() runs inside torch.no_grad() as required
by the BaseAdapter contract.

Device handling
---------------
The adjacency is held as a registered buffer on `STLGRUModel` (see
stlgru_model.py), so it follows the model under `.to(device)` automatically —
no manual `_move_aux_tensors` walk is required here.
"""

import torch
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.lightweight.stlgru_model import STLGRUModel


class STLGRUAdapter(BaseAdapter):

    def __init__(self):
        self.model: STLGRUModel | None = None

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> STLGRUModel:
        """
        Args:
            config: full experiment config dict. Uses:
                config['model']['hidden_dim'], ['dropout']
                config['data']['horizon']
                config['meta']['n_vars']
            graph:  (N, N) static adjacency.

        Returns:
            Initialised STLGRUModel.
        """
        N = graph.shape[0]
        model_cfg = dict(config["model"])
        model_cfg["input_dim"] = config["meta"]["n_vars"]
        model_cfg["n_vars"]    = config["meta"]["n_vars"]
        model_cfg["horizon"]   = config["data"]["horizon"]

        self.model = STLGRUModel(model_cfg, num_nodes=N, adj=graph)
        return self.model

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            x:   (B, T, N, d)
            adj: (N, N) — already bound into the model as a buffer at build time.

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
