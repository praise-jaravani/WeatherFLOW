"""
Model wrapper for DiffSTG (Diffusion-based Spatio-Temporal Graph model).

Cloned repo: models/diffusion/DiffSTG/
Key entry points:
  - DiffSTG/algorithm/diffstg/ugnet.py — class UGnet(nn.Module). This is the
    denoising network epsilon_theta(x_t, t, condition). It takes a noisy tensor
    of shape (B, F, V, T_h + T_p), the diffusion timestep, and a condition
    tuple (x_masked, pos_w, pos_d) where x_masked is the same-shape masked
    history+future tensor (future channels zeroed). pos_w/pos_d are *unused* by
    UGnet itself and we pass ``None`` for them.
  - DiffSTG/algorithm/diffstg/model.py — full DDPM/DDIM wrapper. We do NOT use
    this directly; the diffusion logic is reimplemented in the adapter so that
    forward/predict return shapes match the (B, H, N, d) pipeline contract.

DiffSTG uses a conditional diffusion process over the spatio-temporal graph.
At training time a shortened denoising chain (or single-step closed-form
recovery) is used; at inference the full chain runs for higher quality. See
modules/denoising.py for shared schedule logic.

This wrapper only owns the denoising network. The diffusion schedule and chain
logic live in the adapter so that we can implement the BaseAdapter contract
(forward returning (B, H, N, d)) cleanly.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any
from types import SimpleNamespace

REPO_PATH = Path(__file__).parent / "DiffSTG"
ALGO_PATH = REPO_PATH / "algorithm"


def _ensure_repo_on_path() -> None:
    """Insert the cloned DiffSTG package roots into sys.path.

    UGnet performs relative imports inside the ``algorithm/diffstg`` package
    and also imports ``utils.common_utils`` as a top-level module, so we add
    both the repo root and the ``algorithm`` directory.
    """
    for p in (REPO_PATH, ALGO_PATH):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


class _UGnetConfig(SimpleNamespace):
    """SimpleNamespace that also supports ``.get(key, default)`` like easydict.

    UGnet's __init__ calls ``config.get('n_blocks', 2)``; the upstream code
    assumes an easydict. Subclassing SimpleNamespace and adding ``.get`` keeps
    us off the easydict dependency.
    """

    def get(self, key, default=None):
        return getattr(self, key, default)


class DiffSTGModel(nn.Module):
    """Thin wrapper around the cloned DiffSTG UGnet denoising network.

    Args:
        config: model sub-dict from YAML (must contain ``hidden_dim``;
            ``channel_multipliers`` defaults to [1, 2] and ``n_blocks`` to 2).
        num_nodes: N — number of graph nodes.
        adj: (N, N) adjacency tensor; converted to numpy for UGnet's internal
            ``asym_adj`` calls.
        n_vars: feature dimension d (F in upstream notation).
        input_len: T_h (history length).
        horizon: T_p (prediction horizon).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        num_nodes: int,
        adj: torch.Tensor,
        n_vars: int,
        input_len: int,
        horizon: int,
    ):
        super().__init__()
        _ensure_repo_on_path()

        from algorithm.diffstg.ugnet import UGnet  # noqa: PLC0415

        d_h = int(config.get("hidden_dim", 32))
        channel_multipliers = config.get("channel_multipliers", [1, 2])
        n_blocks = int(config.get("n_blocks", 2))

        # UGnet stores ``a1`` and ``a2`` (normalised adjacencies) as plain
        # attributes — they are torch tensors placed on ``config.device`` at
        # construction time. To keep the wrapper movable with ``.to(...)``
        # later, we construct on CPU and patch a1/a2 to be buffers afterwards.
        adj_np = adj.detach().cpu().numpy().astype(np.float32)

        ugnet_cfg = _UGnetConfig(
            T_p=horizon,
            T_h=input_len,
            V=num_nodes,
            F=n_vars,
            d_h=d_h,
            n_blocks=n_blocks,
            channel_multipliers=list(channel_multipliers),
            supports_len=2,
            device=torch.device("cpu"),
            A=adj_np,
        )

        self.net = UGnet(ugnet_cfg)

        # Replace the plain-tensor adjacency attributes with registered buffers
        # so they follow the module on ``.to(device)`` calls (device-mismatch
        # fix; mirrors the lesson learned from the CLCRN integration).
        a1 = self.net.a1.detach().clone()
        a2 = self.net.a2.detach().clone()
        del self.net.a1
        del self.net.a2
        self.net.register_buffer("a1", a1)
        self.net.register_buffer("a2", a2)

        self.input_len = input_len
        self.horizon = horizon
        self.n_vars = n_vars
        self.num_nodes = num_nodes

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Denoising network forward pass: predict noise at diffusion step ``t``.

        Args:
            x_noisy: noisy joint history+future tensor of shape (B, F, V, T_h+T_p).
            t: long tensor of diffusion timesteps of shape (B,).
            condition: x_masked of shape (B, F, V, T_h+T_p) — history channels
                hold the real values, future channels are zeroed.

        Returns:
            Predicted noise tensor of shape (B, F, V, T_h+T_p).
        """
        # UGnet expects ``c = (x_masked, pos_w, pos_d)`` but only consumes
        # x_masked; pos_w / pos_d are unused. Pass ``None`` for the unused.
        c = (condition, None, None)
        return self.net(x_noisy, t, c)
