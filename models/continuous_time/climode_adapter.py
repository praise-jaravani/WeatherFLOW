"""
Adapter (method) for the ClimODE-inspired advection graph-ODE — training and
prediction logic.

⚠️ Not the Verma et al. (ICLR 2024) ClimODE architecture. See
climode_model.py header for the naming note. The class name is kept as
`ClimODEAdapter` for CLI backwards-compat; the label "ClimODE-inspired" is
used in all human-facing docs.

Uses physics-inspired advection ODE integrated with torchdiffeq. Unlike
MTGODE, this adapter uses the static adjacency from preprocessing as the
graph transport operator (A in dX/dt = -gate(X)·A·X + emission(X)).

forward() vs predict() distinction:
  - forward(): training mode — odeint_adjoint with loose tolerances (rtol=1e-3).
  - predict(): eval mode — odeint with tight tolerances (rtol=1e-5), no gradient.

Both paths share the same model weights and dynamics function.
"""

import numpy as np
import torch
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.continuous_time.climode_model import ClimODEModel


class ClimODEAdapter(BaseAdapter):

    def __init__(self):
        self.model: ClimODEModel | None = None

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> ClimODEModel:
        """
        Args:
            config: full experiment config dict. Must include:
                      config['meta']['n_vars']    — number of variables d
                      config['data']['horizon']   — forecast horizon H
                      config['data']['input_len'] — historical window T
                      config['model']['hidden_dim']
            graph:  (N, N) static adjacency — used as graph transport operator.

        Returns:
            Initialised ClimODEModel.
        """
        N = graph.shape[0]
        adj = graph.numpy() if isinstance(graph, torch.Tensor) else graph

        merged = {
            **config.get("model", {}),
            "model":      config.get("model", {}),
            "ode":        config.get("ode", {}),
            "input_dim":  config["meta"]["n_vars"],
            "output_dim": config["meta"]["n_vars"],
            "horizon":    config["data"]["horizon"],
            "input_len":  config["data"]["input_len"],
        }

        self.model = ClimODEModel(merged, num_nodes=N, adj=adj)
        return self.model

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass — train mode, loose ODE tolerances.

        Args:
            x:   (B, T, N, d)
            adj: (N, N) — not used at call time (baked into model as buffer)

        Returns:
            (B, H, N, d)
        """
        self.model.train()
        return self.model(x)

    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Evaluation forward pass — eval mode, tight ODE tolerances, no gradient.

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
