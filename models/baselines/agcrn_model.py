"""
Model wrapper for AGCRN (Adaptive Graph Convolutional Recurrent Network).

Cloned repo: models/baselines/AGCRN/
Key entry point: AGCRN/model/AGCRN.py — class AGCRN(nn.Module)

AGCRN learns its own adaptive graph via node embeddings (AVWGCN) and does
not use the static adjacency matrix passed by the pipeline. The static adj
is accepted in forward() for interface consistency but ignored internally.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
from types import SimpleNamespace

REPO_PATH = Path(__file__).parent / "AGCRN"


class AGCRNModel(nn.Module):

    def __init__(self, config: Dict[str, Any], num_nodes: int):
        """
        Args:
            config: model sub-dict from YAML (hidden_dim, num_layers, embed_dim, cheb_k).
            num_nodes: N — number of graph nodes.
        """
        super().__init__()
        if str(REPO_PATH) not in sys.path:
            sys.path.insert(0, str(REPO_PATH))

        from model.AGCRN import AGCRN  # noqa: PLC0415

        args = SimpleNamespace(
            num_nodes=num_nodes,
            input_dim=config["input_dim"],
            rnn_units=config["hidden_dim"],
            output_dim=config["output_dim"],
            horizon=config["horizon"],
            num_layers=config["num_layers"],
            embed_dim=config["embed_dim"],
            cheb_k=config["cheb_k"],
            default_graph=True,
        )
        self.net = AGCRN(args)
        self.horizon = args.horizon
        self.output_dim = args.output_dim

        # torch.FloatTensor(shape) in AVWGCN allocates uninitialized memory (can
        # be NaN). Re-initialize all AVWGCN weight/bias pools properly.
        for module in self.net.modules():
            if hasattr(module, "weights_pool"):
                nn.init.kaiming_uniform_(module.weights_pool, nonlinearity="relu")
            if hasattr(module, "bias_pool"):
                nn.init.zeros_(module.bias_pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, d)
        Returns:
            (B, H, N, output_dim)
        """
        # AGCRN.forward(source, targets, teacher_forcing_ratio) but targets are
        # not actually used in its implementation — only source is encoded.
        dummy = torch.zeros(
            x.shape[0], self.horizon, x.shape[2], self.output_dim,
            device=x.device, dtype=x.dtype,
        )
        return self.net(x, dummy, teacher_forcing_ratio=0.0)
