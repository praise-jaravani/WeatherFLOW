"""
Model wrapper for ClimODE (Climate and Weather Forecasting with Physics-informed Neural ODEs).

Cloned repo: models/continuous_time/ClimODE/

ClimODE's original architecture is designed for gridded ERA5 data (2D spatial
H×W grids) using torch.gradient() for advection. That spatial structure is
incompatible with our graph-based N-node format.

This wrapper implements the core ClimODE idea — physics-inspired advection ODE
integrated with torchdiffeq — adapted for graph-structured station data:

  dX/dt = -gate(X) ⊙ (A · X) + emission(X)

where:
  gate(X)      learned velocity magnitude (sigmoid), controls transport rate
  A · X        graph diffusion (row-normalised adjacency, analogous to ∇X)
  emission(X)  learned correction/source term (analogous to ClimODE's residual)

The adjacency A is registered as a buffer and moves with the model to the
correct device automatically via .to(device).

A GRU encoder compresses the T-step historical window into an initial ODE state.
torchdiffeq.odeint (or odeint_adjoint during training) integrates the ODE
forward for H steps and returns (B, H, N, d) predictions.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any

from torchdiffeq import odeint, odeint_adjoint

REPO_PATH = Path(__file__).parent / "ClimODE"


class _ClimODEDynamics(nn.Module):
    """
    ODE dynamics function: dX/dt = -gate(X) * (A·X) + emission(X)
    """

    def __init__(self, d: int, hidden_dim: int, adj: torch.Tensor):
        super().__init__()
        self.register_buffer("adj", adj)

        self.gate = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, d),
            nn.Sigmoid(),
        )
        self.emission = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d)
        v = self.gate(x)                                            # (B, N, d)
        transported = torch.einsum("nm,bmd->bnd", self.adj, x)     # (B, N, d)
        return -v * transported + self.emission(x)


class ClimODEModel(nn.Module):

    def __init__(
        self,
        config: Dict[str, Any],
        num_nodes: int,
        adj: np.ndarray,
    ):
        """
        Args:
            config:    model sub-dict merged with ode, data, and meta keys.
            num_nodes: N — number of graph nodes.
            adj:       (N, N) static adjacency as a numpy array.
        """
        super().__init__()

        d          = config["input_dim"]
        H          = config["horizon"]
        T          = config["input_len"]
        hidden_dim = int(config.get("hidden_dim", 64))
        ode_cfg    = config.get("ode", {})

        self.H         = H
        self.d         = d
        self.use_adjoint = bool(ode_cfg.get("adjoint", True))
        self.ode_method  = ode_cfg.get("method", "dopri5")
        self.train_rtol  = float(ode_cfg.get("train_rtol", 1e-3))
        self.train_atol  = float(ode_cfg.get("train_atol", 1e-4))
        self.eval_rtol   = float(ode_cfg.get("eval_rtol", 1e-5))
        self.eval_atol   = float(ode_cfg.get("eval_atol", 1e-6))

        adj_t = torch.from_numpy(adj).float()
        self.dynamics = _ClimODEDynamics(d, hidden_dim, adj_t)

        # Node-wise GRU encoder (shared weights across nodes)
        self.encoder = nn.GRU(input_size=d, hidden_size=hidden_dim, batch_first=True)
        self.proj    = nn.Linear(hidden_dim, d)

        # Integration time points: t=0 (current) + H future steps
        t_eval = torch.linspace(0.0, 1.0, H + 1)
        self.register_buffer("t_eval", t_eval)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, d) — pipeline convention.

        Returns:
            (B, H, N, d)
        """
        B, T, N, d = x.shape

        # Encode history per node: reshape to (B*N, T, d)
        x_nodes = x.permute(0, 2, 1, 3).reshape(B * N, T, d)
        _, h = self.encoder(x_nodes)          # h: (1, B*N, hidden_dim)
        h = h.squeeze(0).reshape(B, N, -1)   # (B, N, hidden_dim)
        x0 = self.proj(h)                     # (B, N, d) — initial ODE state

        t_eval = self.t_eval.to(x0.device)

        if self.training and self.use_adjoint:
            out = odeint_adjoint(
                self.dynamics, x0, t_eval,
                method=self.ode_method,
                rtol=self.train_rtol,
                atol=self.train_atol,
                adjoint_params=list(self.dynamics.parameters()),
            )
        elif self.training:
            out = odeint(
                self.dynamics, x0, t_eval,
                method=self.ode_method,
                rtol=self.train_rtol,
                atol=self.train_atol,
            )
        else:
            out = odeint(
                self.dynamics, x0, t_eval,
                method=self.ode_method,
                rtol=self.eval_rtol,
                atol=self.eval_atol,
            )

        # out: (H+1, B, N, d) — skip t=0 (initial condition)
        out = out[1:]                # (H, B, N, d)
        out = out.permute(1, 0, 2, 3)  # (B, H, N, d)
        return out
