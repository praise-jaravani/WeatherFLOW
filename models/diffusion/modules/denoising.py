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

Tensor convention here: all schedule tensors are 1-D of shape (T,). The
indexing helper :func:`_gather` reshapes them to broadcast against arbitrary
data tensors. Diffusion schedules are storage-only — adapters are expected to
move them to the same device as their model (we expose schedules as plain
tensors so the caller can ``register_buffer`` them on the wrapper module to
avoid device-mismatch bugs).
"""

import math
import torch
from typing import Tuple


def linear_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """Linear noise schedule. Returns beta tensor of shape (timesteps,)."""
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal 2021). Returns beta of shape (timesteps,)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999).to(torch.float32)


def precompute_diffusion_params(betas: torch.Tensor) -> dict:
    """
    Precompute alphas, alpha_bars, and related constants from the beta schedule.

    Returns a dict with keys: betas, alphas, alpha_bars, sqrt_alpha_bars,
    sqrt_one_minus_alpha_bars, posterior_variance — each a tensor of shape (T,).
    """
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    # posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    alpha_bars_prev = torch.cat([torch.ones(1, dtype=alpha_bars.dtype), alpha_bars[:-1]])
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars).clamp(min=1e-20)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "alpha_bars_prev": alpha_bars_prev,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
        "posterior_variance": posterior_variance,
    }


def _gather(consts: torch.Tensor, t: torch.Tensor, ndim: int = 4) -> torch.Tensor:
    """Gather schedule entries at indices ``t`` and reshape for broadcasting.

    Args:
        consts: 1-D schedule tensor of shape (T,).
        t: long tensor of indices of shape (B,).
        ndim: target rank for broadcasting (default 4 → shape (B, 1, 1, 1)).

    Returns:
        Tensor of shape (B,) + (1,) * (ndim - 1).
    """
    out = consts.to(t.device).gather(-1, t)
    view_shape = (-1,) + (1,) * (ndim - 1)
    return out.reshape(view_shape)


def q_sample(
    x_start: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """Forward diffusion: add noise to ``x_start`` at timestep ``t``.

    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
    """
    ndim = x_start.ndim
    a = _gather(sqrt_alpha_bars, t, ndim=ndim)
    b = _gather(sqrt_one_minus_alpha_bars, t, ndim=ndim)
    return a * x_start + b * noise


def p_sample(
    model_output: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    diffusion_params: dict,
) -> torch.Tensor:
    """One reverse diffusion step: denoise ``x_t`` to ``x_{t-1}`` (DDPM update).

    Args:
        model_output: predicted noise eps_theta(x_t, t).
        x_t: current noisy sample.
        t: scalar timestep index in [0, T-1].
        diffusion_params: dict from :func:`precompute_diffusion_params`.
    """
    betas = diffusion_params["betas"].to(x_t.device)
    alphas = diffusion_params["alphas"].to(x_t.device)
    alpha_bars = diffusion_params["alpha_bars"].to(x_t.device)
    posterior_variance = diffusion_params["posterior_variance"].to(x_t.device)

    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bars[t]

    coef = 1.0 / torch.sqrt(alpha_t)
    eps_coef = beta_t / torch.sqrt((1.0 - alpha_bar_t).clamp(min=1e-20))
    mean = coef * (x_t - eps_coef * model_output)

    if t == 0:
        return mean
    var = posterior_variance[t]
    noise = torch.randn_like(x_t)
    return mean + torch.sqrt(var.clamp(min=0.0)) * noise


def full_denoising_chain(
    model,
    shape: Tuple,
    device: torch.device,
    diffusion_params: dict,
    condition,
) -> torch.Tensor:
    """Run the full reverse denoising chain from pure noise to prediction.

    Used in ``predict()`` (evaluation mode). The ``model`` callable must accept
    ``(x_t, t, condition)`` and return predicted noise of the same shape as
    ``x_t``.
    """
    T = diffusion_params["betas"].shape[0]
    x_t = torch.randn(shape, device=device)
    for t in range(T - 1, -1, -1):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = model(x_t, t_batch, condition)
        x_t = p_sample(eps, x_t, t, diffusion_params)
    return x_t


def shortened_denoising_chain(
    model,
    shape: Tuple,
    device: torch.device,
    diffusion_params: dict,
    condition,
    n_steps: int = 10,
) -> torch.Tensor:
    """Run a shortened reverse denoising chain for fast training-time inference.

    Used in ``forward()`` (training mode). Starts from pure noise at the
    largest timestep and visits ``n_steps`` evenly-spaced timesteps down to 0
    (DDIM-style strided sampling on the same DDPM update rule).
    """
    T = diffusion_params["betas"].shape[0]
    n_steps = max(1, min(n_steps, T))
    # Strided indices from (T-1) down to 0.
    step_indices = torch.linspace(T - 1, 0, n_steps, dtype=torch.long).tolist()

    x_t = torch.randn(shape, device=device)
    for t in step_indices:
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = model(x_t, t_batch, condition)
        x_t = p_sample(eps, x_t, t, diffusion_params)
    return x_t
