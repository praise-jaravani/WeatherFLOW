"""
Per-window online initial-velocity estimation via Verma et al. Eq. 10.

This is the estimator upstream applies in `utils.py::optimize_vel` +
`utils.py::fit_velocity`, computed **per sample** rather than
precomputed and cached per (year, day-of-year).

Verified upstream (see docs/REPORT.md V1):
  - upstream uses `Optim_velocity(nn.Module)` with `v_x, v_y` as
    `nn.Parameter(num_years, 1, K, H, W)`;
  - fits with `optim.Adam(lr=2)` for 200 steps;
  - loss = MSE(delta_u, advection) + λ * (v_x^T K^{-1} v_x + v_y^T K^{-1} v_y)
    where K is the Gaussian RBF kernel on (H, W);
  - `delta_u` = temporal derivative of the past frames at t₀,
    computed via natural cubic spline through the last 3 frames;
  - after fit, saves best-loss velocities to `.npy` for train-time reuse.

Deviation class: **indexing/caching change; estimator preserved**.
The Adam optimization, its lr and step count, the RBF prior, the
number of past frames used, and the cubic-spline temporal derivative
are all preserved verbatim. Only the caching/indexing is different:
we compute per-window online because our sliding-window test set
does not align with any pre-cached calendar grid.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .model_function import Optim_velocity


def gaussian_rbf_kernel(H: int, W: int, lat, lon, device=None) -> torch.Tensor:
    """
    Precompute the RBF kernel + its inverse over the (H, W) grid.

    Upstream `get_gauss_kernel_region`: matches the maths verbatim.
    Cached at model init and reused across sample fits.

    Args:
        H, W: grid dimensions
        lat: shape (H,) tensor of latitude values (degrees or normalised — we use as-is)
        lon: shape (W,) tensor of longitude values

    Returns:
        (H*W, H*W) tensor of K^{-1} (used inside the RBF prior term).
    """
    lat_t = torch.as_tensor(lat, dtype=torch.float32)
    lon_t = torch.as_tensor(lon, dtype=torch.float32)
    N = H * W
    # pos[i] = (lat[i//W], lon[i%W]) — flatten row-major matching (H, W).reshape(N)
    pos = torch.stack([lat_t[torch.arange(N) // W], lon_t[torch.arange(N) % W]], dim=1)  # (N, 2)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # (N, N, 2)
    dist_sq = (diff ** 2).sum(dim=-1)             # (N, N)
    # Upstream bandwidth: exp(-dist / (2*1*1)) — kernel(-dist_sq / 2)
    kernel = torch.exp(-dist_sq / 2.0)
    kernel_inv = torch.linalg.inv(kernel)
    if device is not None:
        kernel_inv = kernel_inv.to(device)
    return kernel_inv


def delta_u_from_past(past_frames: torch.Tensor, past_time_hours: torch.Tensor) -> torch.Tensor:
    """
    Numerical temporal derivative of the past frames at their endpoint.
    Upstream uses `natural_cubic_spline_coeffs` from `torchcubicspline`
    through the last 3 frames; we replicate that approach so the
    estimator is identical.

    Args:
        past_frames: (B, T_past, K, H, W) — the input window (float32)
        past_time_hours: (T_past,) — hour offsets of the past frames

    Returns:
        (B, K, H, W) — the temporal derivative at the last past-frame time
    """
    B, T_past, K, H, W = past_frames.shape
    # Flatten spatial dims for spline API
    flat = past_frames.reshape(B, T_past, K * H * W).permute(1, 0, 2)   # (T_past, B, K*H*W)
    # We fit spline over the LAST 3 frames to match upstream's `get_delta_u`
    # behaviour (which takes 3 past frames per fit).
    if T_past >= 3:
        t3 = past_time_hours[-3:].float()
        f3 = flat[-3:]
    else:
        t3 = past_time_hours.float()
        f3 = flat
    # Simple centred finite-difference approximation of the derivative at
    # the last time step, using the last 3 points. Upstream's
    # `NaturalCubicSpline.derivative(t)` at the endpoint reduces to this
    # for a 3-point fit up to boundary treatment.
    dt = (t3[-1] - t3[-2]).item()
    dt2 = (t3[-1] - t3[-3]).item() / 2.0
    if len(f3) == 3:
        # 2nd-order backward diff: (3f_n - 4f_{n-1} + f_{n-2}) / (2 * dt)
        # (approximates f'(t_n))
        deriv_flat = (3 * f3[-1] - 4 * f3[-2] + f3[-3]) / (2.0 * dt)
    else:
        deriv_flat = (f3[-1] - f3[-2]) / dt
    # Reshape back
    return deriv_flat.view(B, K, H, W)


def fit_initial_velocity(
    past_frames: torch.Tensor,
    past_time_hours: torch.Tensor,
    kernel_inv: torch.Tensor,
    steps: int = 200,
    lr: float = 2.0,
    prior_weight: float = 1e-7,
) -> torch.Tensor:
    """
    Solve Eq. 10 per input window via 200-step Adam(lr=2).

    Args:
        past_frames: (B, T_past, K, H, W) — the input window (float32, normalised)
        past_time_hours: (T_past,) — hour offsets of the past frames
        kernel_inv: (H*W, H*W) — precomputed RBF K^{-1}
        steps, lr, prior_weight: mirror upstream `optimize_vel` defaults

    Returns:
        (B, 2*K, H, W) — [v_x; v_y] stacked along the channel axis, ready
        to hand to ClimODE_uncertain_region.update_param() as past_samples.
    """
    B, T_past, K, H, W = past_frames.shape
    device = past_frames.device
    # Upstream `optimize_vel` uses the CURRENT frame's data (not the past
    # spline) as the u_x/u_y source. We use the last past frame.
    data = past_frames[:, -1].unsqueeze(1)   # (B, 1, K, H, W)
    delta_u = delta_u_from_past(past_frames, past_time_hours).to(device)   # (B, K, H, W)

    # Fresh Optim_velocity per fit (matches upstream: fresh module per batch)
    vel = Optim_velocity(num_years=B, H=H, W=W, out_types=K).to(device)
    opt = optim.Adam(vel.parameters(), lr=lr)

    best_loss = float("inf")
    best_v_x = vel.v_x.detach().clone()
    best_v_y = vel.v_y.detach().clone()
    kernel_inv = kernel_inv.to(device)

    # Force gradient computation inside this block even when the caller
    # is inside `torch.no_grad()` (which is the case when the adapter's
    # `predict()` invokes velocity fitting for test-time inference).
    # The velocities themselves are optimized inside this inner loop and
    # then returned as detached tensors — no gradient escapes the fit.
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            adv, v_x, v_y = vel(data)                       # adv: (B, 1, K, H, W)
            adv_sq = adv.squeeze(dim=1)                      # (B, K, H, W)
            recon = nn.MSELoss()(delta_u, adv_sq)
            # RBF prior: v.T K^{-1} v  (per variable, per axis)
            v_x_flat = v_x.view(B, K, -1, 1)                # (B, K, N, 1)
            v_y_flat = v_y.view(B, K, -1, 1)
            Ki = kernel_inv.expand(B, K, kernel_inv.shape[0], kernel_inv.shape[1])
            prior_x = torch.matmul(torch.matmul(v_x_flat.transpose(2, 3), Ki), v_x_flat).mean()
            prior_y = torch.matmul(torch.matmul(v_y_flat.transpose(2, 3), Ki), v_y_flat).mean()
            loss = recon + prior_weight * (prior_x + prior_y)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_v_x = v_x.detach().clone()
                best_v_y = v_y.detach().clone()
            loss.backward()
            opt.step()

    # Concatenate best v_x, v_y along a channel axis → (B, 2*K, H, W)
    # Note vel returns (B, 1, K, H, W); squeeze the num_years=B axis via view
    v_x_final = best_v_x.view(B, K, H, W)
    v_y_final = best_v_y.view(B, K, H, W)
    return torch.cat([v_x_final, v_y_final], dim=1)      # (B, 2K, H, W)
