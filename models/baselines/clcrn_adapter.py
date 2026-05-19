"""
Adapter (method) for CLCRN — training and prediction logic.

CLCRN is a spherical-geometry recurrent GNN. It builds its own k-NN graph
from node lat/lon coordinates; the static adj from preprocessing is not used.

build_model() requires coords_lonlat in meta — the pipeline attaches this
when it calls the adapter.
"""

import numpy as np
import torch
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.baselines.clcrn_model import CLCRNModel


class CLCRNAdapter(BaseAdapter):

    def __init__(self):
        self.model: CLCRNModel | None = None

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> CLCRNModel:
        """
        Args:
            config: full experiment config dict. Must include:
                      config['meta']['coords']  — (N, 2) float array [lat, lon]
            graph:  (N, N) static adjacency — not used by CLCRN.

        Returns:
            Initialised CLCRNModel.
        """
        N = graph.shape[0]
        model_cfg = config["model"]
        model_cfg["input_dim"]  = config["meta"]["n_vars"]
        model_cfg["output_dim"] = config["meta"]["n_vars"]
        model_cfg["horizon"]    = config["data"]["horizon"]
        model_cfg["input_len"]  = config["data"]["input_len"]

        # coords from preprocessing are [lat, lon]; CLCRN KernelGenerator
        # expects [lon, lat] — swap columns
        coords_latlon = config["meta"]["coords"]          # (N, 2) [lat, lon]
        coords_lonlat = coords_latlon[:, [1, 0]]          # (N, 2) [lon, lat]

        self.model = CLCRNModel(model_cfg, num_nodes=N, coords_lonlat=coords_lonlat)
        return self.model

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            x:   (B, T, N, d)
            adj: (N, N) — ignored (CLCRN uses its own spherical k-NN graph)

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
