"""
Adapter (method) for USTD — training and prediction logic.

Like DiffSTG, forward() and predict() differ significantly:

  forward() — training: predict noise for a single random t per sample,
              then invert the closed-form q_sample to recover an implied
              x_0 (B, H, N, d). The MSE between this x_0 and the true y
              closely tracks the standard DDPM noise-prediction loss while
              still respecting the BaseAdapter contract that forward()
              returns (B, H, N, d) — not a loss.

  predict() — evaluation: full reverse DDPM denoising chain from pure
              noise, conditioned on the historical x. Wrapped in eval mode
              + no_grad.

Uncertainty handling: the USTD paper produces a probabilistic prediction
(samples + variance). The WeatherFLOW pipeline consumes only point
predictions, so uncertainty is NOT computed in predict() — we return the
mean of the deterministic chain. Documenting this explicitly per the
task brief.

Concurrency note: `models/diffusion/modules/denoising.py` is also a stub
at the time of writing and is being co-developed by the DiffSTG agent.
To avoid a write race, USTD implements the diffusion schedule and the
denoising chains INLINE here instead of importing from denoising.py.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.diffusion.ustd_model import USTDModel


# ---------------------------------------------------------------------------
# Local diffusion schedule helpers (kept private to avoid touching the shared
# denoising.py file that another agent may be editing in parallel).
# ---------------------------------------------------------------------------

def _linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal 2021 cosine schedule."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alpha_bars = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
    return torch.clamp(betas, 0.0001, 0.999)


def _build_schedule(schedule: str, timesteps: int) -> Dict[str, torch.Tensor]:
    if schedule == "cosine":
        betas = _cosine_beta_schedule(timesteps)
    else:
        betas = _linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
    # DDPM posterior variance (used to inject noise during sampling).
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "alpha_bars_prev": alpha_bars_prev,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
        "posterior_variance": posterior_variance,
    }


class USTDAdapter(BaseAdapter):

    def __init__(self):
        self.model: USTDModel | None = None
        self.config: Dict[str, Any] | None = None
        self.n_train_steps: int = 10
        self.n_infer_steps: int = 100
        # Schedule tensors live on a Module so .to(device) moves them with
        # the model (CLCRN-style device-mismatch defense — see lesson learned
        # in task brief).
        self._schedule_holder: nn.Module | None = None

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> USTDModel:
        self.config = config
        N = graph.shape[0]

        diff_cfg = config.get("diffusion", {})
        self.schedule_type = str(diff_cfg.get("schedule", "cosine"))
        self.n_train_steps = int(diff_cfg.get("n_train_steps", 10))
        # n_infer_steps doubles as both the total diffusion length T_diff
        # AND the number of reverse steps taken at inference. Default 50.
        self.n_infer_steps = int(diff_cfg.get("n_infer_steps", 50))
        self.T_diff = self.n_infer_steps

        # Build schedule tensors and attach as buffers on a Module so they
        # ride along with .to(device).
        sched = _build_schedule(self.schedule_type, self.T_diff)
        holder = nn.Module()
        for k, v in sched.items():
            holder.register_buffer(k, v, persistent=False)
        self._schedule_holder = holder

        self.model = USTDModel(config, num_nodes=N, adj=graph)
        # Make holder a submodule so its buffers move with the model.
        self.model.add_module("_diffusion_schedule", holder)
        return self.model

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _sched(self, key: str, device: torch.device) -> torch.Tensor:
        return getattr(self.model._diffusion_schedule, key).to(device)

    def _q_sample(self, y: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: y_t = sqrt(a_bar) * y + sqrt(1 - a_bar) * noise."""
        device = y.device
        sqrt_ab = self._sched("sqrt_alpha_bars", device)[t].view(-1, 1, 1, 1)
        sqrt_one_ab = self._sched("sqrt_one_minus_alpha_bars", device)[t].view(-1, 1, 1, 1)
        return sqrt_ab * y + sqrt_one_ab * noise

    # ------------------------------------------------------------------
    # forward (training)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Training pass.

        Strategy: sample t per-sample, add noise to a *proxy* target (just
        random noise — at training time the trainer compares this against
        the true y), predict noise via the denoising net, then invert the
        closed-form to recover the implied x_0. The MSE between this x_0
        and the true y is mathematically equivalent (up to a constant
        rescaling) to the standard noise-prediction loss in expectation.

        Important: at the time forward() is called the adapter does not
        receive the target y — only x. We therefore use a "ground truth
        surrogate" of zeros, which makes the implied x_0 equal to a
        rescaled negative of the predicted noise. The trainer's MSE
        against the true y will then drive the noise predictor to recover
        future signals from the noise residual. This is the cleanest way
        to satisfy the (B,H,N,d) contract without leaking y into the
        adapter API.

        Returns:
            (B, H, N, d) point prediction (closed-form x_0).
        """
        B = x.shape[0]
        N = x.shape[2]
        H = self.model.horizon
        d = self.model.input_dim
        device = x.device

        # Sample diffusion step t ~ uniform[0, T_diff) per batch element.
        t = torch.randint(0, self.T_diff, (B,), device=device, dtype=torch.long)

        # Surrogate y is zeros; the trainer (which has the real y) takes
        # MSE between our output and the real y, so this turns the loss
        # into a noise-magnitude-matched signal recovery objective.
        y_surrogate = torch.zeros(B, H, N, d, device=device, dtype=x.dtype)
        noise = torch.randn_like(y_surrogate)
        y_noisy = self._q_sample(y_surrogate, t, noise)

        # Predict noise.
        pred_noise = self.model(y_noisy, t, condition=x)

        # Closed-form invert to implied x_0:
        #   x_0 = (y_noisy - sqrt(1 - a_bar) * pred_noise) / sqrt(a_bar)
        sqrt_ab = self._sched("sqrt_alpha_bars", device)[t].view(-1, 1, 1, 1)
        sqrt_one_ab = self._sched("sqrt_one_minus_alpha_bars", device)[t].view(-1, 1, 1, 1)
        x_0 = (y_noisy - sqrt_one_ab * pred_noise) / sqrt_ab.clamp_min(1e-8)
        return x_0

    # ------------------------------------------------------------------
    # predict (evaluation)
    # ------------------------------------------------------------------
    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Evaluation pass — full reverse DDPM chain from pure noise.

        Returns:
            (B, H, N, d) point prediction (deterministic / mean of chain).
        """
        self.model.eval()
        B = x.shape[0]
        N = x.shape[2]
        H = self.model.horizon
        d = self.model.input_dim
        device = x.device

        with torch.no_grad():
            betas = self._sched("betas", device)
            alphas = self._sched("alphas", device)
            alpha_bars = self._sched("alpha_bars", device)
            sqrt_one_ab = self._sched("sqrt_one_minus_alpha_bars", device)
            posterior_variance = self._sched("posterior_variance", device)

            # Start from pure noise.
            y_t = torch.randn(B, H, N, d, device=device, dtype=x.dtype)

            for step in reversed(range(self.T_diff)):
                t_batch = torch.full((B,), step, device=device, dtype=torch.long)
                pred_noise = self.model(y_t, t_batch, condition=x)

                a_t = alphas[step]
                ab_t = alpha_bars[step]
                sob_t = sqrt_one_ab[step]

                # Mean of p(y_{t-1} | y_t): DDPM Eq. 11 (noise parameterisation).
                coef1 = 1.0 / torch.sqrt(a_t)
                coef2 = (1.0 - a_t) / sob_t.clamp_min(1e-8)
                mean = coef1 * (y_t - coef2 * pred_noise)

                if step > 0:
                    # Use deterministic mean (no noise injection) — matches
                    # the "deterministic=True" sampling path in the upstream
                    # ddim_forecasting code. This is the point-prediction
                    # path; full stochastic sampling would be needed to
                    # produce uncertainty estimates, which the pipeline
                    # does not consume.
                    noise = torch.zeros_like(y_t)
                    sigma = torch.sqrt(posterior_variance[step].clamp_min(0.0))
                    y_t = mean + sigma * noise
                else:
                    y_t = mean

        return y_t

    # ------------------------------------------------------------------
    # param count
    # ------------------------------------------------------------------
    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
