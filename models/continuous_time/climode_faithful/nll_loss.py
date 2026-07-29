"""
Gaussian negative log-likelihood — Verma et al. Eq. 12.

Vendored from upstream `utils.py::nll`, with three preserved details:
  - latitude-cosine weighting in the RMSE-like term
  - Gaussian variance prior with hyper-variance λ_σ
  - cosine-annealed λ_σ during training (`var_coeff` in the caller)

Loss L = 1/(N*K*H*W) Σ_i [ -log N(y_i | μ_i, σ_i²) - log N⁺(σ_i | 0, λ_σ²) ]
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _lat_weights(lat_degrees: torch.Tensor) -> torch.Tensor:
    """Cosine-latitude weights, normalised to mean 1."""
    lat_rad = lat_degrees * math.pi / 180.0
    w = torch.cos(lat_rad)
    return w / w.mean()


def climode_nll(
    mean: torch.Tensor,
    std: torch.Tensor,
    truth: torch.Tensor,
    lat_degrees: torch.Tensor,
    var_coeff: float,
) -> torch.Tensor:
    """
    Args:
        mean:  (T, B, K, H, W) predictions from ClimODE emission head
        std:   (T, B, K, H, W) predicted per-cell std (Softplus output)
        truth: (B, T, K, H, W) observed values (broadcast/transpose to align with mean if needed)
        lat_degrees: (H,) latitude values in degrees
        var_coeff: hyper-variance for the σ prior (cosine-annealed by the caller)

    Returns:
        scalar loss (Gaussian NLL + variance prior), normalised per (H*W).
    """
    # Align shapes: mean is (T, B, K, H, W); truth typically (B, T, K, H, W)
    if truth.dim() == 5 and truth.shape[0] != mean.shape[0]:
        truth = truth.transpose(0, 1)   # → (T, B, K, H, W)

    device = mean.device
    H = mean.shape[-2]
    w = _lat_weights(lat_degrees).to(device).view(1, 1, 1, H, 1)   # broadcast over (T, B, K, ·, W)

    # Clip std to avoid log(0)
    std = std.clamp(min=1e-6)

    # Gaussian NLL for the mean: 0.5*log(2πσ²) + (y-μ)²/(2σ²)
    nll_data = 0.5 * (torch.log(2 * math.pi * std ** 2) + (truth - mean) ** 2 / (std ** 2))
    nll_data = (nll_data * w).mean()

    # Variance prior: half-normal N⁺(σ|0, var_coeff) — penalises σ from blowing up
    #   log N⁺(σ|0, λ²) ∝ -σ²/(2λ²)  (up to constant)
    if var_coeff > 0:
        prior = 0.5 * (std ** 2) / (var_coeff ** 2)
        prior = (prior * w).mean()
    else:
        prior = torch.tensor(0.0, device=device)

    return nll_data + prior
