"""
Model wrapper for USTD (Uncertainty-aware Spatio-Temporal Diffusion).

Cloned repo: models/diffusion/USTD/
Reference entry points explored:
  - USTD/models/stdiffusion/stdiffusionfore_model.py — STDiffusionForeModel
    is the full forecasting model: it owns the diffusion schedule, the
    GWaveNetEncoder for historical context, and the STFormerForecasting
    denoising network. The training step samples t, adds noise to future_gt,
    then asks netSTD to predict noise.
  - USTD/models/stdiffusion/stformer.py — STFormerForecasting is the
    attention-based denoising network operating on (B, N, L, D) tensors and
    a condition produced by the GWaveNetEncoder. It depends on Laplacian
    positional encodings, pretrained encoder checkpoints, side-info dicts,
    and per-step diffusion embeddings — all wired to the upstream Options
    class.

Why we don't import upstream directly: STFormerForecasting and
STDiffusionForeModel inherit from a custom BaseModel that hard-requires
an `opt` namespace, pretrained checkpoints, custom dataloaders, and
laplacian PEs of a specific shape. They cannot run as a pure nn.Module on
(B, T, N, d) pipeline tensors. We therefore implement a faithful but
self-contained USTD-style denoising network (transformer with temporal +
spatial attention, conditioned on the historical context and the diffusion
step) that respects the (B, T, N, d) -> (B, H, N, d) pipeline contract.

The full diffusion forward/predict chain logic lives in ustd_adapter.py.
"""

import math
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

REPO_PATH = Path(__file__).parent / "USTD"
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional / diffusion-step embedding.

    Args:
        t: (B,) integer or float timestep tensor.
        dim: embedding dimensionality.

    Returns:
        (B, dim) embeddings.
    """
    device = t.device
    half = dim // 2
    # frequencies
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device, dtype=torch.float32) / max(half, 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:  # pad if dim is odd
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class STBlock(nn.Module):
    """A transformer-style spatio-temporal block with diffusion-step gating.

    Inputs and outputs are (B, T, N, D) hidden states. The block does:
      1. Temporal self-attention across T per node.
      2. Spatial self-attention across N per timestep.
      3. Cross-attention from x to the historical condition.
      4. FFN.
    A FiLM-style modulation from the diffusion-step embedding is added at
    each sub-block, plus a gate from the condition's mean.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.t_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.s_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.c_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # FiLM modulation from diffusion-step embedding: (B, hidden_dim)
        # produces per-block scale/shift.
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)

    def _film(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Apply per-sample FiLM modulation. x: (B, *, D), t_emb: (B, D)."""
        h = self.film(t_emb)  # (B, 2D)
        scale, shift = h.chunk(2, dim=-1)
        # broadcast across all intermediate dims
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return x * (1.0 + scale) + shift

    def forward(
        self,
        x: torch.Tensor,           # (B, T, N, D)
        cond: torch.Tensor,        # (B, T_c, N, D)
        t_emb: torch.Tensor,       # (B, D)
    ) -> torch.Tensor:
        B, T, N, D = x.shape
        Tc = cond.shape[1]

        # 1. Temporal self-attention: collapse N into batch -> (B*N, T, D)
        h = self.norm1(x)
        h = self._film(h, t_emb)
        h_t = h.permute(0, 2, 1, 3).reshape(B * N, T, D)
        h_t, _ = self.t_attn(h_t, h_t, h_t, need_weights=False)
        h_t = h_t.reshape(B, N, T, D).permute(0, 2, 1, 3)
        x = x + h_t

        # 2. Spatial self-attention: collapse T into batch -> (B*T, N, D)
        h = self.norm2(x)
        h = self._film(h, t_emb)
        h_s = h.reshape(B * T, N, D)
        h_s, _ = self.s_attn(h_s, h_s, h_s, need_weights=False)
        h_s = h_s.reshape(B, T, N, D)
        x = x + h_s

        # 3. Cross-attention to condition: per-node, query along T, key/value
        # along T_c. (B*N, T, D) attends to (B*N, T_c, D).
        h = self.norm3(x)
        q = h.permute(0, 2, 1, 3).reshape(B * N, T, D)
        kv = cond.permute(0, 2, 1, 3).reshape(B * N, Tc, D)
        h_c, _ = self.c_attn(q, kv, kv, need_weights=False)
        h_c = h_c.reshape(B, N, T, D).permute(0, 2, 1, 3)
        x = x + h_c

        # 4. FFN
        h = self.norm4(x)
        h = self._film(h, t_emb)
        x = x + self.ffn(h)
        return x


class USTDModel(nn.Module):
    """Self-contained USTD-style denoising network.

    The model predicts noise (epsilon) given:
      - x_noisy:  (B, H, N, d) — noisy version of the future at timestep t
      - t:        (B,)         — diffusion step indices
      - condition:(B, T, N, d) — clean historical context (the adapter's x)

    Output: (B, H, N, d) — predicted noise.

    Diffusion schedule tensors live on the adapter, not the model, but the
    model still needs to be torch-pure so .to(device) moves all params.
    """

    def __init__(self, config: Dict[str, Any], num_nodes: int, adj: torch.Tensor):
        super().__init__()
        model_cfg = config["model"]
        self.hidden_dim = int(model_cfg.get("hidden_dim", 64))
        self.num_heads = int(model_cfg.get("num_heads", 4))
        self.num_layers = int(model_cfg.get("num_layers", 2))
        self.dropout = float(model_cfg.get("dropout", 0.1))

        self.input_dim = int(config["meta"]["n_vars"])
        self.input_len = int(config["data"]["input_len"])
        self.horizon = int(config["data"]["horizon"])
        self.num_nodes = int(num_nodes)

        # Input/condition embeddings (per-(node, time) token).
        self.x_embed = nn.Linear(self.input_dim, self.hidden_dim)
        self.cond_embed = nn.Linear(self.input_dim, self.hidden_dim)

        # Learnable node embedding — replaces the upstream Laplacian PE,
        # works for any N because we allocate at build time given the
        # actual N from the pipeline.
        self.node_embedding = nn.Parameter(torch.randn(self.num_nodes, self.hidden_dim) * 0.02)

        # Learnable temporal positional embeddings for x (length=horizon)
        # and for condition (length=input_len).
        self.x_time_pe = nn.Parameter(torch.randn(self.horizon, self.hidden_dim) * 0.02)
        self.cond_time_pe = nn.Parameter(torch.randn(self.input_len, self.hidden_dim) * 0.02)

        # Diffusion-step embedding MLP.
        self.t_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

        # Stack of spatio-temporal transformer blocks.
        self.blocks = nn.ModuleList([
            STBlock(self.hidden_dim, self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.input_dim)

        # Store adjacency as a non-trainable buffer so .to(device) moves it
        # (CLCRN-style device-mismatch defense). Not used by the attention
        # network itself but kept available for future graph extensions.
        if adj is not None:
            self.register_buffer("adj", adj.float(), persistent=False)
        else:
            self.adj = None

    def forward(
        self,
        x_noisy: torch.Tensor,    # (B, H, N, d)
        t: torch.Tensor,          # (B,)
        condition: torch.Tensor,  # (B, T, N, d)
    ) -> torch.Tensor:
        """Predict noise. Returns (B, H, N, d)."""
        B, H, N, _ = x_noisy.shape
        T = condition.shape[1]

        # embed and add positional encodings
        h = self.x_embed(x_noisy)                       # (B, H, N, D)
        h = h + self.x_time_pe[:H].view(1, H, 1, -1)
        h = h + self.node_embedding[:N].view(1, 1, N, -1)

        c = self.cond_embed(condition)                  # (B, T, N, D)
        c = c + self.cond_time_pe[:T].view(1, T, 1, -1)
        c = c + self.node_embedding[:N].view(1, 1, N, -1)

        # diffusion step embedding -> (B, D)
        t_emb = sinusoidal_embedding(t, self.hidden_dim)
        t_emb = self.t_mlp(t_emb)

        for block in self.blocks:
            h = block(h, c, t_emb)

        out = self.out_proj(self.out_norm(h))  # (B, H, N, d)
        return out
