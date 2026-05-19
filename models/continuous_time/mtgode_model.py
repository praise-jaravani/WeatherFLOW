"""
Model wrapper for MTGODE (Multi-scale Temporal Graph ODE).

Cloned repo: models/continuous_time/MTGODE/
Key entry point: MTGODE/model.py — class MTGODE.

MTGODE uses two nested ODE integrations:
  - Outer temporal ODE (ODEBlock): integrates the spatio-temporal state forward
    using method_1 ('euler' or 'rk4' only — raises ValueError otherwise).
  - Inner graph ODE (CGP inside STBlock): graph-level ODE per STBlock step.

Input convention: MTGODE expects (B, d, N, T); our pipeline uses (B, T, N, d).
This wrapper handles the permutation.

Multi-variable output: MTGODE was designed for single-variable prediction. For
d variables and H horizon steps we set out_dim = H * d and reshape the output
from (B, H*d, N, 1) back to (B, H, N, d).

Adaptive graph: buildA_true=True — MTGODE learns its own adjacency (same
philosophy as AGCRN). The static adj from preprocessing is not used.

Device note: MTGODE stores self.idx as a plain attribute (not a registered
buffer), so it is not moved by .to(device). The wrapper fixes this in forward().
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

REPO_PATH = Path(__file__).parent / "MTGODE"


class MTGODEModel(nn.Module):

    def __init__(
        self,
        config: Dict[str, Any],
        num_nodes: int,
    ):
        """
        Args:
            config:    model sub-dict merged with ode, data, and meta keys.
            num_nodes: N — number of graph nodes.
        """
        super().__init__()

        if str(REPO_PATH) not in sys.path:
            sys.path.insert(0, str(REPO_PATH))

        from model import MTGODE  # noqa: PLC0415

        model_cfg = config.get("model", config)
        ode_cfg   = config.get("ode", {})

        d  = config["input_dim"]
        H  = config["horizon"]
        T  = config["input_len"]

        self.H = H
        self.d = d

        self.net = MTGODE(
            buildA_true=True,
            num_nodes=num_nodes,
            device=torch.device("cpu"),        # moved to GPU later via .to(device)
            predefined_A=None,
            dropout=float(model_cfg.get("dropout", 0.3)),
            subgraph_size=int(model_cfg.get("graph_constructor", {}).get("k", 10)),
            node_dim=int(model_cfg.get("hidden_dim", 40)),
            dilation_exponential=1,
            conv_channels=int(model_cfg.get("hidden_dim", 40)),
            end_channels=128,
            seq_length=T,
            in_dim=d,
            out_dim=H * d,
            tanhalpha=float(model_cfg.get("graph_constructor", {}).get("alpha", 3)),
            method_1=ode_cfg.get("method_1", "euler"),
            time_1=float(ode_cfg.get("time_1", 1.2)),
            step_size_1=float(ode_cfg.get("step_size_1", 0.4)),
            method_2=ode_cfg.get("method_2", "euler"),
            time_2=float(ode_cfg.get("time_2", 1.0)),
            step_size_2=float(ode_cfg.get("step_size_2", 0.25)),
            alpha=1.0,
            rtol=float(ode_cfg.get("rtol", 1e-4)),
            atol=float(ode_cfg.get("atol", 1e-3)),
            adjoint=bool(ode_cfg.get("adjoint", True)),
            perturb=False,
            ln_affine=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, d) — pipeline convention.

        Returns:
            (B, H, N, d)
        """
        B, T, N, d = x.shape

        # MTGODE expects (B, d, N, T)
        x_mtg = x.permute(0, 3, 2, 1)   # (B, d, N, T)

        # self.idx is a plain attribute, not a buffer — move it manually
        self.net.idx = self.net.idx.to(x.device)

        out = self.net(x_mtg)             # (B, H*d, N, 1)
        out = out.squeeze(-1)             # (B, H*d, N)
        out = out.view(B, self.H, d, N)  # (B, H, d, N)
        out = out.permute(0, 1, 3, 2)    # (B, H, N, d)
        return out
