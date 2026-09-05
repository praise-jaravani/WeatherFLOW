"""
Block-autoregressive rollout evaluation — Deliverable C of specs/VPT_Extended_Spec.md.

Given a trained direct-forecast model that consumes an (input_len, N, d) window
and emits a (horizon, N, d) block, chain the model's own predictions back as
input for K blocks. Measure per-step NRMSE against the truth continuation.

The rollout produces exposure bias: the model was trained on ground-truth
windows but sees its own outputs at inference. CNB (Moges/Skokos/Moodley) and
Pathak et al. (2018) both treat this as the measurement of interest for
long-horizon skill, not a confound.

Feedback hygiene (spec §5.3): before feeding a prediction back in, we clamp
physically-bounded channels (tp ≥ 0; rh ∈ [0, 100]; ws ≥ 0) and wrap the
circular wind-direction channel to [0, 360°). For ERA5 the spec also
re-derives ws10 from u10/v10 and rh from t2m/d2m via Magnus-Tetens — those
derivations are NOT applicable to SAWS where wind speed, wind direction, and
RH are all direct sensor observations.

**Closure invariant (spec §5.1):** the rollout requires `input_len == horizon`
so that block k's prediction exactly replaces block k+1's input window. All
current SAWS configs (input_len=12, horizon=12) satisfy this. **If Moodley's
demanded 24-hour retraining goes 12-in/24-out (asymmetric), this module will
raise at rollout_predict — a windowed-feedback logic would need to be added.
If it goes 24-in/24-out, closure holds and everything works unchanged.** This
constraint should be on the SAWS retrained-config blocking-decisions list.

This module is dataset-agnostic. Hygiene rules are chosen from `var_names`
by name; unknown variables pass through unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Sensible physical bounds. Applied in denormalised (physical-unit) space.
# Keys match variable names on both ERA5 and SAWS pipelines — SAWS uses "ws"
# not "ws10"; both are registered so `apply_hygiene` clamps the right channel
# regardless of which dataset produced the prediction. tp / rh names are
# identical across pipelines.
PHYSICAL_BOUNDS = {
    "tp":   (0.0, None),      # rainfall never negative (both datasets)
    "rh":   (0.0, 100.0),     # relative humidity is a percentage (both)
    "ws10": (0.0, None),      # wind speed never negative — ERA5 name
    "ws":   (0.0, None),      # wind speed never negative — SAWS name
}

# Wind direction is circular (0° == 360°) and is not clampable as a scalar.
# Wrap into [0, 360) after denormalisation so any wandering prediction
# lands back on the training manifold. Handled separately from PHYSICAL_BOUNDS.
WD_NAMES = {"wd"}   # SAWS uses "wd"; ERA5 encodes direction via u10/v10 components


def _denormalise(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """z → physical.  Broadcast mean/std from (1, N, d) → (H, N, d)."""
    return z * std[0] + mean[0]


def _normalise(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """physical → z."""
    return (x - mean[0]) / std[0]


def apply_hygiene(
    pred_z: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    var_names: Sequence[str],
) -> np.ndarray:
    """
    Denormalise → clamp physically-bounded channels + wrap circular channels
    → renormalise.

    Args:
      pred_z    : (H, N, d) z-space prediction to sanitise
      mean, std : (1, N, d) train stats
      var_names : d variable names — used to select which channels to clamp
                  and which to wrap circularly

    Returns:
      pred_z_hygiened : (H, N, d) z-space
    """
    physical = _denormalise(pred_z, mean, std)
    for i, name in enumerate(var_names):
        if name in PHYSICAL_BOUNDS:
            lo, hi = PHYSICAL_BOUNDS[name]
            physical[..., i] = np.clip(
                physical[..., i],
                lo if lo is not None else -np.inf,
                hi if hi is not None else np.inf,
            )
        elif name in WD_NAMES:
            # Circular wrap into [0, 360). Feeding back a wandering scalar
            # of 400° or -30° pushes the model off its training manifold in
            # a way no clip fixes — mod-360 puts it back on-manifold.
            # Note: this is the minimal fix. The correct long-term treatment
            # is sin/cos encoding at training-pipeline level (a training
            # decision, not a rollout patch — see specs/VPT_Extended_Spec.md
            # follow-up in specs/ADJACENCY_AUDIT.md).
            physical[..., i] = np.mod(physical[..., i], 360.0)
    return _normalise(physical, mean, std).astype(pred_z.dtype)


# ---------------------------------------------------------------------------
# Adapter-free rollout — takes a callable to isolate testing
# ---------------------------------------------------------------------------

def rollout_predict(
    predict_fn,
    initial_window: np.ndarray,
    K: int,
    horizon: int,
    mean: np.ndarray,
    std: np.ndarray,
    var_names: Sequence[str],
    apply_hygiene_between_steps: bool = True,
) -> np.ndarray:
    """
    Run K rollout blocks. Each block: predict_fn(window) → (H, N, d); feed
    output back as next window (after hygiene).

    Args:
      predict_fn      : callable window (T, N, d) → pred (H, N, d), both z-space.
                        For an adapter, wrap `lambda w: adapter.predict(...)`.
      initial_window  : (input_len, N, d) z-space ground-truth window.
      K               : number of rollout blocks.
      horizon         : H — must equal input_len for the closed-loop invariant
                        (predicted block replaces the input window).
      mean, std       : (1, N, d) train stats.
      var_names       : d variable names.
      apply_hygiene_between_steps: run the physical-bounds clamp between steps.

    Returns:
      concatenated preds : (K·H, N, d) z-space

    Raises:
      ValueError if horizon != input_len (violates closure invariant).
    """
    T, N, d = initial_window.shape
    if T != horizon:
        raise ValueError(
            f"rollout requires input_len ({T}) == horizon ({horizon}) "
            f"for closed-loop closure (spec §5.1)"
        )

    window = initial_window.copy()
    preds_all = np.empty((K * horizon, N, d), dtype=initial_window.dtype)
    for k in range(K):
        pred = predict_fn(window)   # (H, N, d) z-space
        if pred.shape != (horizon, N, d):
            raise ValueError(
                f"predict_fn returned {pred.shape}, expected ({horizon}, {N}, {d})"
            )
        preds_all[k * horizon : (k + 1) * horizon] = pred
        if apply_hygiene_between_steps:
            window = apply_hygiene(pred, mean, std, var_names)
        else:
            window = pred
    return preds_all


def rollout_nrmse_curve(
    preds_z: np.ndarray, truth_z: np.ndarray,
) -> np.ndarray:
    """
    Per-step NRMSE across nodes for a single rollout.

    Args:
      preds_z, truth_z : (K·H, N, d) z-space

    Returns:
      curve : (K·H, d) — sqrt(mean_N((pred - truth)²)) per step per variable
    """
    if preds_z.shape != truth_z.shape:
        raise ValueError(
            f"shape mismatch: preds {preds_z.shape} vs truth {truth_z.shape}"
        )
    se = (preds_z - truth_z) ** 2
    mse = se.mean(axis=1)   # pool over N → (K·H, d)
    return np.sqrt(mse).astype(np.float32)


# ---------------------------------------------------------------------------
# Guards (spec §5.4)
# ---------------------------------------------------------------------------

def detect_numerical_failure(pred_z: np.ndarray) -> bool:
    """True if predictions contain NaN or Inf — that start's rollout terminates."""
    return bool(np.isnan(pred_z).any() or np.isinf(pred_z).any())


def detect_collapse(curve: np.ndarray, band: Tuple[float, float] = (0.9, 1.1),
                    min_consecutive: int = 12) -> np.ndarray:
    """
    CNB-analog mean-collapse screen (spec §5.4).

    A start is flagged as 'collapsed' for variable v if the NRMSE curve sits
    inside [band_lo, band_hi] for at least min_consecutive steps without
    crossing either bound. These curves are climatology-hugging, not
    tracking.

    Args:
      curve           : (K·H, d) rollout NRMSE curve
      band            : (lo, hi) — the "hovering near climatology" band
      min_consecutive : minimum consecutive steps inside the band

    Returns:
      collapsed : (d,) bool — True per variable if any run of the band is
                  min_consecutive or longer.
    """
    T, d = curve.shape
    lo, hi = band
    in_band = (curve >= lo) & (curve <= hi)   # (T, d)
    collapsed = np.zeros(d, dtype=bool)
    for v in range(d):
        run = 0
        for t in range(T):
            if in_band[t, v]:
                run += 1
                if run >= min_consecutive:
                    collapsed[v] = True
                    break
            else:
                run = 0
    return collapsed


def early_stop_start(curve: np.ndarray, threshold: float = 5.0) -> bool:
    """
    Once ALL variables' NRMSE exceeds `threshold` in one step, the start is
    effectively done — extending the rollout wastes compute (crossings are
    already recorded far below this threshold).

    Correctness: we stop only when EVERY variable at the last step is above
    threshold. Using max-over-variables would stop on the first fast-diverging
    channel (rain/wind) and right-censor slow-tracking channels (pressure/
    temperature) that were still under ε — destroying exactly the RVPT
    measurements the guard exists to preserve.
    """
    if curve.shape[0] == 0:
        return False
    # min-over-variables > threshold ⇔ all variables > threshold
    min_per_step = curve.min(axis=1)
    return bool(min_per_step[-1] > threshold)


# ---------------------------------------------------------------------------
# Reference rollouts — persistence and climatology (no model)
# ---------------------------------------------------------------------------

def persistence_rollout(
    seed_window: np.ndarray, K: int, horizon: int,
) -> np.ndarray:
    """
    Persistence rollout: repeat the last hour of the seed window forward.

    Args:
      seed_window : (input_len, N, d) — last hour is the anchor
      K           : number of blocks
      horizon     : H per block

    Returns:
      (K·H, N, d) — every step is a copy of seed_window[-1]
    """
    last = seed_window[-1:]              # (1, N, d)
    return np.repeat(last, K * horizon, axis=0)


def climatology_rollout(
    train_mean_per_node: np.ndarray, K: int, horizon: int, N: int, d: int,
) -> np.ndarray:
    """
    Climatology rollout in z-space: constant zero at every step (train mean
    is the origin of z-space by construction).

    Args:
      train_mean_per_node : (N, d) — unused, kept for API symmetry
      K, horizon, N, d    : shape

    Returns:
      (K·H, N, d) of zeros in z-space
    """
    return np.zeros((K * horizon, N, d), dtype=np.float32)
