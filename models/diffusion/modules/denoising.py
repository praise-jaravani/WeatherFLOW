"""
Shared denoising schedule utilities for DiffSTG and USTD.

Both diffusion models share a common noise schedule framework (linear or cosine
beta schedule) and the core denoising math. Centralising this here prevents
duplication and ensures both models use an identical noise schedule.

Training vs. evaluation: diffusion models typically use a shortened denoising
chain during training (e.g. 10 steps) for speed, and the full chain (e.g. 100
or 1000 steps) at inference for quality. The adapter's forward()/predict()
distinction maps directly to this — forward() calls the shortened chain,
predict() calls the full chain.
"""

import torch
import numpy as np
from typing import Tuple


def linear_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """Linear noise schedule. Returns beta tensor of shape (timesteps,)."""
    raise NotImplementedError


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal 2021). Returns beta of shape (timesteps,)."""
    raise NotImplementedError


def precompute_diffusion_params(betas: torch.Tensor) -> dict:
    """
    Precompute alphas, alpha_bars, and related constants from the beta schedule.

    Returns a dict with keys: betas, alphas, alpha_bars, sqrt_alpha_bars,
    sqrt_one_minus_alpha_bars, posterior_variance — each a tensor of shape (T,).
    """
    raise NotImplementedError


def q_sample(
    x_start: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """Forward diffusion: add noise to x_start at timestep t."""
    raise NotImplementedError


def p_sample(
    model_output: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    diffusion_params: dict,
) -> torch.Tensor:
    """One reverse diffusion step: denoise x_t to x_{t-1}."""
    raise NotImplementedError


def full_denoising_chain(
    model,
    shape: Tuple,
    device: torch.device,
    diffusion_params: dict,
    condition: torch.Tensor,
) -> torch.Tensor:
    """
    Run the full reverse denoising chain from pure noise to prediction.
    Used in predict() (evaluation mode).
    """
    raise NotImplementedError


def shortened_denoising_chain(
    model,
    shape: Tuple,
    device: torch.device,
    diffusion_params: dict,
    condition: torch.Tensor,
    n_steps: int = 10,
) -> torch.Tensor:
    """
    Run a shortened denoising chain for faster training-time loss computation.
    Used in forward() (training mode).
    """
    raise NotImplementedError
