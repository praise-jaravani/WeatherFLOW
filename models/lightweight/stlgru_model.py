"""
Model wrapper for STLGRU (Spatio-Temporal Lightweight GRU).

Cloned repo: models/lightweight/STLGRU/
Key entry points:
  - STLGRU/model.py   — main GRU-based STGNN architecture (class `model`)
  - STLGRU/layer.py   — STGRU spatial-temporal recurrent layer
  - STLGRU/modules.py — gcn_glu, GRUcell

STLGRU is designed for efficiency — low parameter count with competitive
accuracy. Wraps the cloned model as a clean torch.nn.Module.

Multi-variable handling
-----------------------
The upstream `model.forward` returns (B, out_length, N, 1) — a single output
channel. To support our d=8 multi-variable target we set
`out_length = horizon * n_vars` and reshape the upstream output back to
(B, H, N, d).

Device handling
---------------
The upstream `model` stores the adjacency as `self.A = A` (a plain attribute,
NOT a registered buffer). `nn.Module.to(device)` would therefore skip it and
cause a device-mismatch on GPU. We fix this once and for all by registering
the adjacency on THIS wrapper as a buffer (`self.A_buf`) and pointing the
upstream model's `self.A` reference at that buffer. Buffers participate in
`.to()`/`._apply()` so the inner reference moves automatically.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
from types import SimpleNamespace

REPO_PATH = Path(__file__).parent / "STLGRU"


class STLGRUModel(nn.Module):
    """Thin wrapper around the cloned STLGRU nn.Module (`STLGRU/model.py::model`)."""

    def __init__(self, config: Dict[str, Any], num_nodes: int, adj: torch.Tensor):
        """
        Args:
            config: model sub-dict from YAML. Must include:
                hidden_dim, dropout, input_dim, horizon, n_vars.
            num_nodes: N — number of graph nodes.
            adj: (N, N) adjacency tensor used by the spatial GCN.
        """
        super().__init__()
        # Force-clear conflicting top-level module names from sys.modules so we
        # always pick up STLGRU's own model.py / layer.py / modules.py /
        # Normalize.py (Lite-STGNN ships a top-level `model` too, and earlier
        # imports may have shadowed ours).
        for _name in ("model", "layer", "modules", "Normalize"):
            sys.modules.pop(_name, None)
        if str(REPO_PATH) not in sys.path:
            sys.path.insert(0, str(REPO_PATH))

        from model import model as STLGRU_upstream  # noqa: PLC0415

        self.horizon = int(config["horizon"])
        self.n_vars = int(config["n_vars"])

        # Upstream emits (B, out_length, N, 1) — we pack horizon * n_vars into
        # that single output dimension and unpack after.
        args = SimpleNamespace(
            input_dim=int(config["input_dim"]),
            n_hid=int(config["hidden_dim"]),
            num_nodes=int(num_nodes),
            dropout=float(config["dropout"]),
            out_length=self.horizon * self.n_vars,
        )

        adj_t = adj.detach().float() if isinstance(adj, torch.Tensor) else torch.as_tensor(adj, dtype=torch.float32)
        # Register the adjacency as a buffer on the wrapper. Buffers are moved
        # by nn.Module._apply (and therefore .to(device)). We then make the
        # upstream model's plain `self.A` attribute reference this same buffer.
        self.register_buffer("A_buf", adj_t)

        self.net = STLGRU_upstream(args, self.A_buf)
        # Ensure the upstream attribute references the wrapper buffer (it
        # already does because we passed self.A_buf in, but make it explicit).
        self.net.A = self.A_buf

        # Upstream STGRU has a HARDCODED `self.linap = nn.Linear(32, n_hid)`
        # (STLGRU/layer.py line 33) that only works when n_hid == 64, because
        # the preceding AvgPool1d(kernel_size=2) halves the channel dim. To
        # honour the configured hidden_dim without editing the cloned repo we
        # rebuild that single submodule to the correct shape post-hoc.
        from layer import STGRU  # noqa: PLC0415
        n_hid = int(config["hidden_dim"])
        for module in self.net.modules():
            if isinstance(module, STGRU):
                module.linap = nn.Linear(n_hid // 2, n_hid)

    def _apply(self, fn):
        # Re-bind the inner reference after any device/dtype move so the
        # upstream `self.A` always tracks the registered buffer.
        result = super()._apply(fn)
        if hasattr(self, "net"):
            self.net.A = self.A_buf
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, d)

        Returns:
            (B, H, N, d)
        """
        B, T, N, d = x.shape
        # Upstream expects (B, T, N, d) and returns (B, H*d, N, 1)
        out = self.net(x)
        # (B, H*d, N, 1) -> (B, H*d, N) -> (B, H, d, N) -> (B, H, N, d)
        out = out.squeeze(-1).view(B, self.horizon, self.n_vars, N).permute(0, 1, 3, 2)
        return out.contiguous()
