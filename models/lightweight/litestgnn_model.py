"""
Model wrapper for Lite-STGNN (Lightweight Spatio-Temporal Graph Neural Network).

Cloned repo: models/lightweight/Lite-STGNN/
Key entry point: Lite-STGNN/src/model.py — class LiteSTGNN(nn.Module)

The upstream LiteSTGNN is single-channel: it takes x of shape [B, L_in, N] and
returns y of shape [B, L_out, N]. Our pipeline operates on multi-variable data
of shape (B, T, N, d=8). We adapt by treating each variable channel as an
independent series sharing one set of learned parameters (including the learned
adjacency): we flatten (B, d) into the leading axis, call the upstream model
once, then reshape back. This is option (b) from the brief — a multi-channel
wrapping that keeps the upstream code untouched.

The upstream model creates all device-aware tensors inside forward() (eye,
arange, etc.) and stores spatial weights as nn.Parameter on E1/E2/g_params, so
nn.Module.to(device) covers all state — no plain tensor attrs requiring a
CLCRN-style _move_aux_tensors fix.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

REPO_PATH = Path(__file__).parent / "Lite-STGNN"
SRC_PATH = REPO_PATH / "src"


class LiteSTGNNModel(nn.Module):
    """Multi-channel wrapper around the upstream single-channel LiteSTGNN."""

    def __init__(self, config: Dict[str, Any], num_nodes: int, adj: torch.Tensor):
        """
        Args:
            config: model sub-dict from YAML. Reads hidden_dim, num_layers,
                    dropout, and optionally adj_rank / adj_topk / adj_tau /
                    prop_orders / gate_mode / self_loop_alpha.
            num_nodes: N — number of graph nodes.
            adj: (N, N) static adjacency — not used by Lite-STGNN (it learns
                 its own low-rank TopK adjacency from node embeddings).
        """
        super().__init__()
        # Force-clear conflicting top-level module names from sys.modules so we
        # always pick up Lite-STGNN's own src/model.py (STLGRU ships a top-level
        # `model` too, and an earlier import in the same Python process may
        # have shadowed ours).
        for _name in ("model", "layer", "modules", "Normalize"):
            sys.modules.pop(_name, None)
        if str(SRC_PATH) not in sys.path:
            sys.path.insert(0, str(SRC_PATH))

        from model import LiteSTGNN  # noqa: PLC0415

        self.seq_len = int(config["input_len"])
        self.pred_len = int(config["horizon"])
        self.n_vars = int(config["input_dim"])

        # Map our YAML knobs to upstream args. Hidden_dim and num_layers don't
        # map 1:1 to Lite-STGNN's hyperparameters (its capacity is governed by
        # adj_rank and prop_orders), so we route hidden_dim -> adj_rank and
        # num_layers -> prop_orders, while honouring explicit overrides if
        # present in the config.
        adj_rank = int(config.get("adj_rank", config.get("hidden_dim", 16)))
        adj_topk = int(config.get("adj_topk", min(10, max(1, num_nodes - 1))))
        # adj_topk must be < n_nodes; clamp defensively.
        adj_topk = max(1, min(adj_topk, num_nodes - 1)) if num_nodes > 1 else 1
        adj_tau = float(config.get("adj_tau", 1.2))
        self_loop_alpha = float(config.get("self_loop_alpha", 0.2))
        prop_orders = int(config.get("prop_orders", config.get("num_layers", 2)))
        gate_mode = str(config.get("gate_mode", "flat"))
        dropout = float(config.get("dropout", 0.0))

        self.net = LiteSTGNN(
            n_nodes=num_nodes,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            adj_rank=adj_rank,
            adj_topk=adj_topk,
            adj_tau=adj_tau,
            self_loop_alpha=self_loop_alpha,
            gate_mode=gate_mode,
            prop_orders=prop_orders,
            use_temporal_head=bool(config.get("use_temporal_head", False)),
            temporal_head_ratio=float(config.get("temporal_head_ratio", 0.5)),
            temporal_head_dropout=dropout,
            use_input_residual=bool(config.get("use_input_residual", True)),
            input_residual_dropout=dropout,
            use_feature_norm=bool(config.get("use_feature_norm", False)),
            residual_dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, d)
        Returns:
            (B, H, N, d)
        """
        B, T, N, d = x.shape
        # (B, T, N, d) -> (B, d, T, N) -> (B*d, T, N)
        x_chan = x.permute(0, 3, 1, 2).reshape(B * d, T, N)
        y_chan = self.net(x_chan)              # (B*d, H, N)
        H = y_chan.shape[1]
        # (B*d, H, N) -> (B, d, H, N) -> (B, H, N, d)
        y = y_chan.reshape(B, d, H, N).permute(0, 2, 3, 1).contiguous()
        return y
