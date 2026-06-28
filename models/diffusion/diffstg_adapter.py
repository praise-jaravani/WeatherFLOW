"""
Adapter (method) for DiffSTG — training and prediction logic.

This is the primary example of why forward() and predict() are separate methods:

  forward() — training: computes a fast diffusion estimate of the prediction
              using a shortened denoising chain (DDPM-style strided sampling).
              Gradients flow through the denoising network so the trainer's
              MSE(forward_output, y_true) backprops a useful signal.

  predict() — evaluation: runs the full denoising chain from pure Gaussian
              noise conditioned on the input, producing the highest-quality
              prediction. Called inside torch.no_grad().

Both paths use the shared denoising utilities in models/diffusion/modules/denoising.py.

Pipeline I/O contract:
    forward/predict input  x: (B, T=input_len, N, d=n_vars) float32
    forward/predict output y: (B, H=horizon,   N, d=n_vars) float32

Internally the denoising network operates on the *joint* history+future tensor
in upstream layout (B, F, V, T_h + T_p). The adapter handles the layout
shuffling and slices the future portion off the denoised joint tensor before
returning.

Device-mismatch handling: the diffusion schedule tensors (betas, alphas,
alpha_bars, ...) are registered as buffers on the wrapper module so that
``model.to(device)`` carries them along. This avoids the silent CPU/GPU
mismatch bugs that have bitten other model integrations in this project.
"""

import torch
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.diffusion.diffstg_model import DiffSTGModel
from models.diffusion.modules.denoising import (
    linear_beta_schedule,
    cosine_beta_schedule,
    precompute_diffusion_params,
    q_sample,
    shortened_denoising_chain,
    full_denoising_chain,
)


# Keys we expect (or precompute) inside the diffusion params dict.
_SCHEDULE_KEYS = (
    "betas",
    "alphas",
    "alpha_bars",
    "alpha_bars_prev",
    "sqrt_alpha_bars",
    "sqrt_one_minus_alpha_bars",
    "posterior_variance",
)


class DiffSTGAdapter(BaseAdapter):

    def __init__(self):
        self.model: DiffSTGModel | None = None
        self.input_len: int = 0
        self.horizon: int = 0
        self.n_vars: int = 0
        # Total diffusion timesteps in the trained schedule (== n_infer_steps).
        self.n_diffusion_steps: int = 100
        # Number of strided denoising steps used inside forward() during training.
        self.n_train_steps: int = 10

    # ------------------------------------------------------------------ build

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> DiffSTGModel:
        N = graph.shape[0]
        model_cfg = config["model"]
        data_cfg = config["data"]
        meta_cfg = config["meta"]
        diff_cfg = config.get("diffusion", {})

        self.input_len = int(data_cfg["input_len"])
        self.horizon = int(data_cfg["horizon"])
        self.n_vars = int(meta_cfg["n_vars"])

        # Diffusion schedule: ``n_infer_steps`` is the total length of the
        # noise schedule used by the full reverse chain; ``n_train_steps`` is
        # the (smaller) number of strided steps used during training-time
        # forward().
        self.n_diffusion_steps = int(diff_cfg.get("n_infer_steps", 100))
        self.n_train_steps = int(diff_cfg.get("n_train_steps", 10))

        schedule = str(diff_cfg.get("schedule", "linear")).lower()
        if schedule == "linear":
            betas = linear_beta_schedule(
                self.n_diffusion_steps,
                float(diff_cfg.get("beta_start", 1e-4)),
                float(diff_cfg.get("beta_end", 0.02)),
            )
        elif schedule == "cosine":
            betas = cosine_beta_schedule(self.n_diffusion_steps)
        else:
            raise ValueError(f"unknown diffusion schedule: {schedule!r}")

        params = precompute_diffusion_params(betas)

        self.model = DiffSTGModel(
            config=model_cfg,
            num_nodes=N,
            adj=graph,
            n_vars=self.n_vars,
            input_len=self.input_len,
            horizon=self.horizon,
        )

        # Register all schedule tensors as buffers on the wrapper module so
        # they move with ``model.to(device)`` automatically.
        for key in _SCHEDULE_KEYS:
            self.model.register_buffer(f"_diff_{key}", params[key])

        return self.model

    # ------------------------------------------------------- helpers (layout)

    def _diffusion_params(self) -> dict:
        """Reconstruct the diffusion params dict from buffers on the model."""
        return {key: getattr(self.model, f"_diff_{key}") for key in _SCHEDULE_KEYS}

    def _build_condition(self, x: torch.Tensor) -> torch.Tensor:
        """Build the masked condition tensor.

        Args:
            x: pipeline-format history (B, T_h, N, d).

        Returns:
            x_masked of shape (B, d, N, T_h + T_p) — history channels filled
            with the real values from ``x`` (transposed to upstream layout)
            and the future T_p timesteps zeroed.
        """
        B = x.shape[0]
        # (B, T_h, N, d) -> (B, d, N, T_h)
        hist = x.permute(0, 3, 2, 1).contiguous()
        fut_zeros = torch.zeros(
            B, self.n_vars, self.model.num_nodes, self.horizon,
            device=x.device, dtype=x.dtype,
        )
        return torch.cat([hist, fut_zeros], dim=3)

    def _future_from_joint(self, joint: torch.Tensor) -> torch.Tensor:
        """Slice the future portion off a joint (B, d, N, T_h+T_p) tensor and
        return it in pipeline layout (B, T_p, N, d).
        """
        fut = joint[..., self.input_len:]  # (B, d, N, T_p)
        return fut.permute(0, 3, 2, 1).contiguous()

    # -------------------------------------------------------------- training

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Training forward pass.

        Strategy: run a shortened (n_train_steps) DDPM reverse chain from pure
        noise, conditioned on the masked history, then slice the predicted
        future. The output is in the same physical space as the target y, so
        MSE(forward_output, y_true) in the trainer is a sensible loss
        (closely tracking the true diffusion loss while preserving the
        (B, H, N, d) shape contract).

        Args:
            x:   (B, T_h, N, d)
            adj: (N, N) — already baked into the model at build time; ignored.

        Returns:
            (B, H, N, d) prediction.
        """
        _ = adj  # static adjacency is fixed at build time

        condition = self._build_condition(x)
        shape = condition.shape  # (B, d, N, T_h + T_p)

        joint = shortened_denoising_chain(
            model=self.model,
            shape=shape,
            device=x.device,
            diffusion_params=self._diffusion_params(),
            condition=condition,
            n_steps=self.n_train_steps,
        )
        return self._future_from_joint(joint)

    # ------------------------------------------------------------ evaluation

    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Evaluation forward pass — full denoising chain, eval mode, no grad.

        Args:
            x:   (B, T_h, N, d)
            adj: (N, N) — ignored.

        Returns:
            (B, H, N, d) prediction.
        """
        _ = adj
        self.model.eval()
        with torch.no_grad():
            condition = self._build_condition(x)
            shape = condition.shape

            joint = full_denoising_chain(
                model=self.model,
                shape=shape,
                device=x.device,
                diffusion_params=self._diffusion_params(),
                condition=condition,
            )
            return self._future_from_joint(joint)

    # --------------------------------------------------------------- params

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
