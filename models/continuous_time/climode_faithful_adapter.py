"""
Adapter for climode_faithful — the faithful reimplementation of Verma
et al.'s ClimODE (ICLR 2024).

⚠️  IMPORTANT: this is the FAITHFUL adapter. The archived
"ClimODE-inspired" model lives at [climode_adapter.py](climode_adapter.py)
and uses the CLI registry key "climode". This adapter uses the key
"climode_faithful" and is registered separately.

Contract with the shared trainer:
  - `build_model(config, adj) -> nn.Module`
      Constructs the vendored ClimODE_uncertain_region + initial-velocity
      infrastructure. Also precomputes the RBF kernel on the (H, W) grid.
  - `forward(x, adj) -> (μ, σ)`
      Training-mode forward. Runs per-window initial-velocity fit,
      integrates the ODE, and returns the Gaussian-emission tuple.
      Trainer must call `native_loss(...)` on this tuple.
  - `predict(x, adj) -> μ`
      Evaluation-mode forward. Same computation as `forward` but wraps
      in `torch.no_grad()` and returns only μ (the predictive mean) as a
      standard (B, H, N, d) tensor for the shared metric pipeline.
  - `native_loss(pred_tuple, target, lat_degrees, var_coeff)`
      Verma et al. Eq. 12 Gaussian NLL + variance prior. Trainer calls
      this instead of MAE when `uses_native_loss` is True.
  - `uses_native_loss = True`
      Declares that this adapter needs a per-model loss (see trainer
      hook for what this triggers).

Shape conversions:
  Input to `forward/predict`:  (B, T, N=999, d=8)  — WeatherFLOW harness
  Reshape to grid:             (B, T, d, H=27, W=37) via (i//W, i%W)
  Solver runs on:              (B, 3d, H, W) state (v_x, v_y, u)
  Solver output:               (T_leads, B, d, H, W)
  Reshape back:                (B, T_leads, N, d)

The 27×37 mapping is proven regular by tests/test_grid_regularity.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.base_adapter import BaseAdapter
from models.continuous_time.climode_faithful.initial_velocity import (
    fit_initial_velocity,
    gaussian_rbf_kernel,
)
from models.continuous_time.climode_faithful.model_function import (
    ClimODE_uncertain_region,
)
from models.continuous_time.climode_faithful.nll_loss import climode_nll
from models.continuous_time.climode_faithful.velocity_cache import (
    load_or_fit_velocity,
)


class ClimODEFaithfulAdapter(BaseAdapter):
    """Adapter for the vendored ClimODE_uncertain_region."""

    # Declared for the trainer's per-model-loss hook (see docstring above).
    uses_native_loss = True

    def __init__(self):
        self.model: ClimODE_uncertain_region | None = None
        self.H: int = 0
        self.W: int = 0
        self.d: int = 0
        self.input_len: int = 0
        self.horizon: int = 0
        self._kernel_inv: torch.Tensor | None = None
        self._lat_map: torch.Tensor | None = None
        self._lon_map: torch.Tensor | None = None
        self._const_info: torch.Tensor | None = None
        self._lat_row_order: np.ndarray | None = None
        self._lon_row_order: np.ndarray | None = None
        # Velocity-cache context — the trainer/DataLoader must set these
        # per batch so the cache key (split_tag, window_index) is
        # well-defined. Default off so unit tests + smoke work without
        # them; see set_velocity_cache_context() for the API.
        self._cache_split_tag: str | None = None
        self._cache_window_indices: list[int] | None = None
        self._cache_use: bool = True

    # ---------------------------- setup -------------------------------- #

    def _capture_grid_from_coords(self, coords: np.ndarray) -> None:
        """
        Given (N, 2) coords in the WeatherFLOW row-major-north-first
        order, capture the row-order lats + lons for reshape and build
        the (H, W) lat/lon maps.
        """
        # Row-order lats: unique in the order they appear in coords[:, 0]
        _, first_idx_lat = np.unique(coords[:, 0], return_index=True)
        row_order_lats = coords[np.sort(first_idx_lat), 0]
        _, first_idx_lon = np.unique(coords[:, 1], return_index=True)
        row_order_lons = coords[np.sort(first_idx_lon), 1]
        assert len(row_order_lats) * len(row_order_lons) == coords.shape[0], (
            "climode_faithful requires a regular H×W grid; see tests/test_grid_regularity.py"
        )
        self.H = int(len(row_order_lats))
        self.W = int(len(row_order_lons))
        self._lat_row_order = np.asarray(row_order_lats, dtype=np.float32)
        self._lon_row_order = np.asarray(row_order_lons, dtype=np.float32)

        lat_grid = np.broadcast_to(self._lat_row_order[:, None], (self.H, self.W))
        lon_grid = np.broadcast_to(self._lon_row_order[None, :], (self.H, self.W))
        self._lat_map = torch.from_numpy(np.asarray(lat_grid, dtype=np.float32))
        self._lon_map = torch.from_numpy(np.asarray(lon_grid, dtype=np.float32))

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> nn.Module:
        """
        Args:
            config: full experiment config dict. Must include:
                config['meta']['coords']    — (N, 2) [lat, lon] float array
                config['meta']['n_vars']    — number of variables (K)
                config['data']['input_len'] — T (history length in hours)
                config['data']['horizon']   — H (forecast horizon in hours)
                config['model']['solver']   — ODE solver (default 'euler')
                config['model']['atol']     — solver atol (default 0.1)
                config['model']['rtol']     — solver rtol (default 0.1)
                config['model']['use_att']  — flow-velocity attention (default True)
                config['model']['use_err']  — Gaussian emission net (default True)
                config['model']['use_pos']  — pos embedding style (default False → paper's non-pos branch)
                config['model']['vel_fit_steps'] — Adam steps for velocity fit (default 200)
                config['model']['vel_fit_lr']    — Adam lr for velocity fit (default 2.0)
                config['model']['vel_fit_prior'] — RBF-prior weight (default 1e-7)
            graph: (N, N) adjacency — ignored (ClimODE uses the grid, not the graph)

        Returns:
            The initialised ClimODE_uncertain_region model.
        """
        coords = np.asarray(config["meta"]["coords"], dtype=np.float32)
        self._capture_grid_from_coords(coords)

        self.d = int(config["meta"]["n_vars"])
        self.input_len = int(config["data"]["input_len"])
        self.horizon = int(config["data"]["horizon"])

        model_cfg = config.get("model", {})
        method = model_cfg.get("solver", "euler")
        use_att = bool(model_cfg.get("use_att", True))
        use_err = bool(model_cfg.get("use_err", True))
        use_pos = bool(model_cfg.get("use_pos", False))
        self._atol = float(model_cfg.get("atol", 0.1))
        self._rtol = float(model_cfg.get("rtol", 0.1))
        self._vel_fit_steps = int(model_cfg.get("vel_fit_steps", 200))
        self._vel_fit_lr = float(model_cfg.get("vel_fit_lr", 2.0))
        self._vel_fit_prior = float(model_cfg.get("vel_fit_prior", 1e-7))

        # Constants (§1.2 of the 2026-07-28 directive): load real ERA5
        # static fields for the SA box if data/constants/*.npy exist,
        # else fall back to zero placeholders with a loud warning. The
        # smoke uses placeholders by design; production seeds MUST run
        # on real constants.
        #
        # Upstream expects const_info of shape (1, 2, H, W) with channel
        # 0 = orography, channel 1 = land-sea mask (see
        # ClimODE_uncertain_region.forward: `self.oro, self.lsm =
        # self.const_info[0, 0], self.const_info[0, 1]`).
        constants_dir = Path("data/constants")
        lsm_path = constants_dir / "era5_sa_lsm.npy"
        oro_path = constants_dir / "era5_sa_orography.npy"
        self._constants_source = "placeholder_zeros"
        if lsm_path.exists() and oro_path.exists():
            lsm = np.load(lsm_path)
            oro = np.load(oro_path)
            if lsm.shape == (self.H, self.W) and oro.shape == (self.H, self.W):
                self._const_info = torch.from_numpy(
                    np.stack([oro, lsm], axis=0)[None].astype(np.float32)
                )
                self._constants_source = "era5_real"
            else:
                # Shape mismatch (e.g. subgrid smoke test) — fall through
                # to placeholder without an alarming warning
                self._const_info = torch.zeros((1, 2, self.H, self.W), dtype=torch.float32)
                self._constants_source = f"placeholder_zeros (real available at {lsm.shape} but this run uses {(self.H, self.W)})"

        if self._constants_source.startswith("placeholder"):
            if not (lsm_path.exists() and oro_path.exists()):
                import warnings
                warnings.warn(
                    "climode_faithful: real ERA5 constants not found at "
                    "data/constants/{era5_sa_lsm.npy,era5_sa_orography.npy}; "
                    "falling back to zero placeholders. Smoke OK, production NOT. "
                    "Run scripts/download_era5_constants.py.",
                    UserWarning, stacklevel=2,
                )
            self._const_info = torch.zeros((1, 2, self.H, self.W), dtype=torch.float32)

        # Precompute RBF kernel inverse for velocity fits
        self._kernel_inv = gaussian_rbf_kernel(
            self.H, self.W, self._lat_row_order, self._lon_row_order
        )

        self.model = ClimODE_uncertain_region(
            num_channels=self.d,
            const_channels=2,
            out_types=self.d,
            method=method,
            use_att=use_att,
            use_err=use_err,
            use_pos=use_pos,
        )
        return self.model

    # ------------------------- shape helpers --------------------------- #

    def _to_grid(self, x_bthd: torch.Tensor) -> torch.Tensor:
        """(B, T, N, d) → (B, T, d, H, W)  — row-major-north-first reshape."""
        B, T, N, d = x_bthd.shape
        assert N == self.H * self.W, f"got N={N} but H*W={self.H*self.W}"
        return x_bthd.reshape(B, T, self.H, self.W, d).permute(0, 1, 4, 2, 3).contiguous()

    def _from_grid(self, x_tbdhw: torch.Tensor) -> torch.Tensor:
        """(T, B, d, H, W) → (B, T, N, d)."""
        T, B, d, H, W = x_tbdhw.shape
        return x_tbdhw.permute(1, 0, 3, 4, 2).reshape(B, T, H * W, d).contiguous()

    # ------------------------- core forward ---------------------------- #

    def _core_forward(self, x: torch.Tensor):
        """
        Shared forward path for training and eval.

        Args:
            x: (B, T=input_len, N=999, d=8) NORMALISED — WeatherFLOW harness input

        Returns:
            (mean, std) each shaped (T_leads=horizon, B, d, H, W).
            NOTE: mean is 5-D — the trainer's `native_loss` handles it.
            `predict` reshapes to (B, horizon, N, d) before returning.
        """
        assert self.model is not None, "call build_model first"
        device = x.device
        B, T, N, d = x.shape

        # (B, T, N, d) → (B, T, d, H, W) — full input window on the grid
        x_grid = self._to_grid(x)                                # (B, T, d, H, W)

        # Fit initial velocity from the T-frame past window — cached
        # per (split, window_index) when the trainer has set the context.
        past_time_hours = torch.arange(T, dtype=torch.float32, device=device)   # 0..T-1
        if self._cache_split_tag is not None and self._cache_window_indices is not None:
            # Per-window cache lookup; stack results into (B, 2d, H, W).
            assert len(self._cache_window_indices) == B, (
                f"cache_window_indices length {len(self._cache_window_indices)} != batch size {B}"
            )
            per_window = []
            for b in range(B):
                v = load_or_fit_velocity(
                    split_tag=self._cache_split_tag,
                    window_index=int(self._cache_window_indices[b]),
                    past_frames=x_grid[b:b + 1],                 # keep B=1 shape
                    past_time_hours=past_time_hours,
                    kernel_inv=self._kernel_inv,
                    steps=self._vel_fit_steps,
                    lr=self._vel_fit_lr,
                    prior_weight=self._vel_fit_prior,
                    use_cache=self._cache_use,
                )                                                 # (2d, H, W)
                per_window.append(v)
            past_samples = torch.stack(per_window, dim=0)         # (B, 2d, H, W)
        else:
            # No cache context — legacy per-batch fit (used by unit tests + smoke)
            past_samples = fit_initial_velocity(
                past_frames=x_grid,
                past_time_hours=past_time_hours,
                kernel_inv=self._kernel_inv,
                steps=self._vel_fit_steps,
                lr=self._vel_fit_lr,
                prior_weight=self._vel_fit_prior,
            )                                                    # (B, 2d, H, W)

        # Hand the model its parameters. `data` upstream is the current
        # frame in shape (B, 1, d, H, W) — we use the LAST input frame
        # as the ODE initial state (the frame at hour T-1).
        data_now = x_grid[:, -1:].contiguous()                   # (B, 1, d, H, W)
        # upstream expects (B, 1, d, H, W) → their `view(-1, out_ch, H, W)` cats past_samples along ch axis

        # Time coordinate for the ODE: T-1 → T-1+horizon (in hours)
        T_span_end = T - 1 + self.horizon
        T_leads = torch.arange(T - 1, T_span_end + 1, dtype=torch.float32, device=device)

        # Push helpers to device
        const = self._const_info.to(device)
        lat_map = self._lat_map.to(device)
        lon_map = self._lon_map.to(device)

        self.model.update_param([past_samples, const, lat_map, lon_map])
        mean, std = self.model(T_leads, data_now, atol=self._atol, rtol=self._rtol)

        # `mean` from upstream forward comes back shaped
        # (T_leads_count, B, K, H, W) or (T_leads_count, B, 1, K, H, W)
        # depending on the noise_net_contrib path. Normalise to 5-D.
        if mean.dim() == 6:
            # (T, B, 1, K, H, W) → squeeze the singleton
            mean = mean.squeeze(2)
            if std is not None:
                std = std.squeeze(2)

        # We asked the solver for T_leads = [T-1, ..., T-1+horizon] i.e.
        # `horizon+1` points; drop the first (initial state, not a forecast)
        # so we end up with exactly `horizon` predictions.
        if mean.shape[0] == self.horizon + 1:
            mean = mean[1:]
            if std is not None:
                std = std[1:]

        return mean, std

    # -------------------- BaseAdapter interface ------------------------ #

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Training-mode forward. Returns (μ, σ) 5-D tuple for the trainer's
        native_loss path.
        """
        self.model.train()
        mean, std = self._core_forward(x)
        return (mean, std)

    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Evaluation-mode forward. Returns just μ, reshaped into the
        standard (B, horizon, N, d) tensor the shared metric pipeline
        expects.
        """
        self.model.eval()
        with torch.no_grad():
            mean, _ = self._core_forward(x)
        # (horizon, B, d, H, W) → (B, horizon, N, d)
        return self._from_grid(mean)

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    # ---------------- velocity-cache context API ---------------------- #

    def set_velocity_cache_context(
        self,
        split_tag: str | None,
        window_indices: list[int] | None = None,
        use_cache: bool = True,
    ) -> None:
        """
        Set the per-batch context that keys the velocity cache.

        The trainer (or a bespoke cache-warmer) calls this immediately
        before each forward/predict so `load_or_fit_velocity(...)` can
        look up per-window fits.

        Args:
            split_tag: "train" | "val" | "test" | "smoke" | None to disable
            window_indices: length-B list of window start indices for the
                            current batch; None to disable cache lookup
            use_cache: if False, fit fresh on each call (used by the smoke
                       to measure worst-case cost per §4 of the review)
        """
        self._cache_split_tag = split_tag
        self._cache_window_indices = window_indices
        self._cache_use = use_cache

    # -------------------- native loss ---------------------------------- #

    def native_loss(
        self,
        pred_tuple: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        var_coeff: float = 1e-3,
    ) -> torch.Tensor:
        """
        Verma et al. Eq. 12 Gaussian NLL + variance prior.

        Args:
            pred_tuple: (μ, σ) from `forward` — each (T_leads, B, d, H, W)
            target:     (B, T_leads, N, d) — the same targets the shared
                        trainer passes to any other model's loss
            var_coeff:  cosine-annealed variance prior weight (trainer sets this)

        Returns:
            scalar loss
        """
        mean, std = pred_tuple
        # target is (B, T, N, d). Reshape to (T, B, d, H, W) to align with mean.
        B, T, N, d = target.shape
        assert N == self.H * self.W, f"target N={N} but H*W={self.H*self.W}"
        target_grid = target.reshape(B, T, self.H, self.W, d).permute(1, 0, 4, 2, 3).contiguous()
        lat_1d = torch.as_tensor(self._lat_row_order, device=mean.device)
        return climode_nll(mean, std, target_grid, lat_1d, var_coeff=var_coeff)
