"""
STGODE wrapper — Fang et al. (KDD 2021), "Spatial-Temporal Graph ODE Networks
for Traffic Flow Forecasting" (arXiv 2106.12931).

Bridges between our (B, T, N, D) tensor convention and upstream STGODE's
(B, N, T, F) convention. Upstream code is vendored verbatim under
`models/continuous_time/STGODE/`.

Post-construction knob override (H3 fix, 2026-08-11): upstream `model.py:95`
hardcodes `ODEG(out_channels[-1], 12, A_hat, time=6)` — both the temporal
dimension (12) and the ODE integration horizon (6) are baked into the call.
- `temporal_dim=12` is fundamental (feeds a `nn.Parameter` of that size at
  ODEFunc init); changing it needs upstream surgery. We ASSERT input_len == 12.
- `time=6` sets the `ODEblock.t` tensor and can be overridden post-hoc by
  reassigning `m.t` on each `ODEblock`. This wrapper does that after upstream
  construction, so `ode.time` in the config is a live knob (not dead as it was
  in the 2026-08-11 audit).

torchdiffeq single-step Euler at t=[0, time] (H4 disclosure): with no
`options={"step_size": ...}` argument, torchdiffeq's fixed-grid Euler uses the
passed `t` array as the integration grid. So `odeint(f, x, [0, T], method='euler')`
executes ONE Euler step of size T: `z = x + T·f(0, x)`. This is faithful to
upstream (which uses the same call) but should be described accurately in the
methods section as "continuous-depth with single-step Euler integration",
NOT as "multi-step ODE solve". If genuine multi-step is ever wanted, we can
pass `options={"step_size": T/K}` for K sub-steps — that would be a documented
departure from upstream.

Loss (uniform-harness override): STGODE paper uses SmoothL1; our stack uses
uniform MAE across all deterministic models (cross-family alignment, see
specs/HYPERPARAMETERS.md). The trainer's `_build_loss` picks this globally;
the wrapper is loss-agnostic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

# Vendored upstream lives at models/continuous_time/STGODE/
_UPSTREAM_DIR = Path(__file__).parent / "STGODE"
if str(_UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_DIR))

# The `sys.modules['model']` guard in scripts/run_worker.py + hpo_run_optuna.py
# purges these across adapter switches (identical to MTGODE / AGCRN / CLCRN
# handling — each vendored repo exposes a bare `model` module).
from model import ODEGCN  # noqa: E402  (vendored upstream)


class STGODEModel(nn.Module):
    """Thin wrapper: applies (B,T,N,D) ↔ (B,N,T,F) permutes around upstream ODEGCN."""

    _EXPECTED_INPUT_LEN = 12  # upstream STGCNBlock hardcodes temporal_dim=12

    def __init__(
        self,
        config: Dict[str, Any],
        num_nodes: int,
        A_sp_hat: torch.Tensor,
        A_se_hat: torch.Tensor,
    ):
        super().__init__()

        input_len = int(config.get("input_len", config.get("data", {}).get("input_len", 12)))
        horizon   = int(config.get("horizon",   config.get("data", {}).get("horizon", 12)))
        input_dim = int(config.get("input_dim", config.get("meta", {}).get("n_vars", 1)))

        # Low-fix: hard assertion instead of the docstring-only "clamp"
        if input_len != self._EXPECTED_INPUT_LEN:
            raise ValueError(
                f"STGODE input_len must be {self._EXPECTED_INPUT_LEN} (upstream STGCNBlock "
                f"hardcodes temporal_dim={self._EXPECTED_INPUT_LEN} at model.py:95). "
                f"Got input_len={input_len}. To change this, patch upstream + this assertion."
            )

        self.H = horizon
        self.d = input_dim
        self.N = num_nodes

        self.net = ODEGCN(
            num_nodes=num_nodes,
            num_features=input_dim,
            num_timesteps_input=input_len,
            num_timesteps_output=horizon,
            A_sp_hat=A_sp_hat,
            A_se_hat=A_se_hat,
        )

        # ---- H3 fix: override upstream's hardcoded ODEblock.t ----
        # Upstream instantiates each ODEblock with `t=torch.tensor([0, time=6])`.
        # We walk the model's submodules and reassign `t` where a user-supplied
        # ode.time is present. Also verified: `method` and other odeint kwargs
        # are set at odeint call time inside `ODEblock.forward`, so `method`
        # can't be overridden without touching that method (documented in specs).
        ode_cfg = config.get("ode", {}) or {}
        cfg_time = ode_cfg.get("time", None)
        if cfg_time is not None:
            override_t = torch.tensor([0.0, float(cfg_time)])
            n_overridden = 0
            for m in self.net.modules():
                # Detect upstream ODEblock by its `.t` attribute + `.odefunc`
                if hasattr(m, "t") and hasattr(m, "odefunc") and isinstance(m.t, torch.Tensor):
                    m.t = override_t
                    n_overridden += 1
            self._n_odeblocks_time_overridden = n_overridden
        else:
            self._n_odeblocks_time_overridden = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, D) — our stack's convention

        Returns:
            (B, H, N, 1) — univariate output wrapped to (B, H, N, D=1) so
            the trainer's downstream code sees the same shape convention
            it expects from the MTGODE / AGCRN / CLCRN adapters (which
            emit full D=6 and get sliced at loss time). Our trainer's
            2026-08-11 fix handles the adapter-univariate case symmetrically.
        """
        x_upstream = x.permute(0, 2, 1, 3).contiguous()  # (B, N, T, D)
        y_upstream = self.net(x_upstream)                # (B, N, H)
        y = y_upstream.permute(0, 2, 1).unsqueeze(-1)    # (B, H, N, 1)
        return y

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters())
