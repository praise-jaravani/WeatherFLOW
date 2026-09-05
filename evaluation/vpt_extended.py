"""
VPT Extended — per-sample Valid Forecast Horizon (VFH), ε-sensitivity grid,
and survival-curve summaries. Implements Deliverables A and B of
specs/VPT_Extended_Spec.md.

Terminology (binding per spec §1):
  VFH  = Valid Forecast Horizon — first horizon step at which per-sample
         NRMSE crosses ε in a single direct forecast (input = ground truth).
  RVPT = Rollout VPT — same but under block-autoregressive rollout.
         Implemented in evaluation/rollout.py.

Two things the spec preserves from evaluation.regression:
  * valid_prediction_time(curve, epsilon) — first-crossing semantics, dips-back
    do not rescue. IMPORTED here rather than duplicated.
  * NRMSE computation on z-scored arrays — RMSE in z-space is NRMSE relative
    to train climatology (see regression.py:94-97 for the pre-existing
    over-normalisation guard).

Design choices spelled out in the spec that this module enforces:
  * NRMSE pools over NODES only for per-sample curves (not over samples).
  * Censoring notation: numeric storage keeps `0` for step-1-crossing and
    `H` for never-crossed; distribution summaries carry explicit pct_floor
    and pct_ceiling fractions.
  * Statistical rail: within a seed, per-sample values are overlapping
    windows and NOT independent — they are summarised to (median, IQR,
    survival curve, censoring fractions) per seed, then aggregated across
    the 10 seeds.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np

from evaluation.regression import valid_prediction_time

log = logging.getLogger(__name__)

# Default ε grid per spec §2. Broad enough to (a) de-censor low-error models
# at the 0.15-0.2 band, (b) keep 0.3 continuity with the current headline,
# and (c) reach 1.0 for CNB/Pathak climatology-equivalence semantics.
DEFAULT_EPS_GRID: Tuple[float, ...] = (0.15, 0.2, 0.3, 0.5, 0.75, 1.0)

# Multi-horizon reporting per 2026-08-04 pivot (Moodley demand for 24h+
# retraining, extended to 48h). Models are trained at horizon = 48 and
# evaluated at each of these subhorizons via truncation of the 48-step output.
# Matches Davidson/Gaibie's 5 horizons (3/6/9/12/24) and extends to 36/48.
DEFAULT_REPORT_HORIZONS: Tuple[int, ...] = (3, 6, 9, 12, 24, 36, 48)


# ---------------------------------------------------------------------------
# Per-sample NRMSE and VFH
# ---------------------------------------------------------------------------

def per_sample_nrmse(preds_z: np.ndarray, targets_z: np.ndarray) -> np.ndarray:
    """
    Per-sample NRMSE, pooled over NODES only.

    Args:
      preds_z, targets_z : (S, H, N, d) z-scored arrays

    Returns:
      curves : (S, H, d) float32 — for each sample s, horizon h, variable v,
               the RMSE across nodes (which in z-space is NRMSE relative to
               the frozen train per-node σ, see spec §2).

    Note: this deliberately does not average over the S axis. The pooled
    curve used elsewhere is `curves.mean(axis=0)`.
    """
    if preds_z.shape != targets_z.shape:
        raise ValueError(
            f"shape mismatch: preds {preds_z.shape} vs targets {targets_z.shape}"
        )
    if preds_z.ndim != 4:
        raise ValueError(f"expected (S, H, N, d) got ndim={preds_z.ndim}")
    se = (preds_z - targets_z) ** 2                # (S, H, N, d)
    mse = se.mean(axis=2)                          # pool over N → (S, H, d)
    return np.sqrt(mse).astype(np.float32)


def vfh_batch(curves: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Apply first-crossing VPT to every (sample, variable) curve.

    Args:
      curves  : (S, H, d) — per-sample NRMSE curves
      epsilon : threshold

    Returns:
      vfh : (S, d) int32 — integer horizon at which curve[s, :, v] first
            reaches or exceeds epsilon; H if it never does (censored high);
            0 if it starts above (censored low).
    """
    S, H, d = curves.shape
    out = np.empty((S, d), dtype=np.int32)
    for s in range(S):
        for v in range(d):
            out[s, v] = valid_prediction_time(curves[s, :, v], epsilon)
    return out


def vfh_grid(curves: np.ndarray, eps_grid: Sequence[float] = DEFAULT_EPS_GRID) -> np.ndarray:
    """
    VFH for every ε in the grid. Returns (S, d, n_eps) int32.

    Storing as int32 (not int8): H can be up to 12·K under rollout, so int8
    would overflow beyond K=10.
    """
    grids = [vfh_batch(curves, e) for e in eps_grid]
    return np.stack(grids, axis=-1).astype(np.int32)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def survival_curve(curves: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Fraction of samples still valid (per-sample NRMSE < ε) at each horizon.

    Args:
      curves  : (S, H, d)
      epsilon : threshold

    Returns:
      surv : (H, d) — surv[h, v] = fraction of samples whose curve stayed
             below ε for every step 0..h.

    Non-increasing in h by construction (once you leave, you stay left).
    """
    S, H, d = curves.shape
    surv = np.empty((H, d), dtype=np.float32)
    for v in range(d):
        # per-sample never-crossed-yet indicator
        below = curves[:, :, v] < epsilon              # (S, H) bool
        # cumulative AND along horizon
        still = np.ones(S, dtype=bool)
        for h in range(H):
            still &= below[:, h]
            surv[h, v] = still.mean()
    return surv


def summarise_vfh(
    vfh: np.ndarray,
    H_cap: int,
    var_names: Sequence[str],
    eps_grid: Sequence[float] = DEFAULT_EPS_GRID,
    curves_for_survival: np.ndarray | None = None,
) -> Dict[str, Dict[float, Dict[str, float | List[float]]]]:
    """
    Summarise per-sample VFH values into a (variable, ε) → stats dict.

    Args:
      vfh                : (S, d, n_eps) integer VFH values
      H_cap              : forecast horizon (12 for direct; 12·K for rollout)
      var_names          : d variable names
      eps_grid           : the ε grid used to produce vfh
      curves_for_survival: if provided, add per-variable per-ε survival curves

    Returns dict shape:
      {var: {eps: {"median", "iqr_low", "iqr_high", "min", "max",
                   "pct_floor", "pct_ceiling",
                   "survival_curve":[...]  # if curves given
                  }}}
    """
    S, d, n_eps = vfh.shape
    if len(var_names) != d:
        raise ValueError(f"var_names length {len(var_names)} != d {d}")
    if len(eps_grid) != n_eps:
        raise ValueError(f"eps_grid length {len(eps_grid)} != n_eps {n_eps}")

    out: Dict[str, Dict[float, Dict[str, float | List[float]]]] = {}
    for v_idx, vname in enumerate(var_names):
        out[vname] = {}
        for e_idx, eps in enumerate(eps_grid):
            values = vfh[:, v_idx, e_idx]   # (S,)
            summary = {
                "median":     float(np.median(values)),
                "iqr_low":    float(np.percentile(values, 25)),
                "iqr_high":   float(np.percentile(values, 75)),
                "min":        int(values.min()),
                "max":        int(values.max()),
                "pct_floor":   float((values == 0).mean()),
                "pct_ceiling": float((values >= H_cap).mean()),
            }
            if curves_for_survival is not None:
                surv = survival_curve(curves_for_survival, eps)
                summary["survival_curve"] = surv[:, v_idx].tolist()
            out[vname][float(eps)] = summary
    return out


# ---------------------------------------------------------------------------
# Deliverable A: ε-sensitivity grid from POOLED curves (no inference)
# ---------------------------------------------------------------------------

def eps_grid_from_pooled_curves(
    pooled_curves: Dict[str, List[float]],
    eps_grid: Sequence[float] = DEFAULT_EPS_GRID,
) -> Dict[str, Dict[float, int]]:
    """
    Compute VFH at each ε for each variable, from the *pooled* per-horizon
    NRMSE curves already stored in test_metrics.json. Cheap; no model
    execution.

    Args:
      pooled_curves : {var_name: [nrmse_at_h0, ..., nrmse_at_h_{H-1}]}
      eps_grid      : sequence of thresholds

    Returns:
      {var_name: {eps: int_vfh}}
    """
    out: Dict[str, Dict[float, int]] = {}
    for vname, curve_list in pooled_curves.items():
        arr = np.asarray(curve_list, dtype=np.float32)
        out[vname] = {float(e): int(valid_prediction_time(arr, e))
                      for e in eps_grid}
    return out


def aggregate_eps_grid_across_seeds(
    per_seed_pooled_vfh: List[Dict[str, Dict[float, int]]],
    eps_grid: Sequence[float] = DEFAULT_EPS_GRID,
    H_cap: int | None = None,
) -> Dict[str, Dict[float, Dict[str, float | int | List[int]]]]:
    """
    Aggregate per-seed pooled-VFH-at-ε dicts to a per-variable-per-ε summary.

    Args:
      per_seed_pooled_vfh: list of dicts (one per seed) as returned by
                           eps_grid_from_pooled_curves.
      eps_grid           : the ε grid used to produce per_seed_pooled_vfh.
      H_cap              : horizon cap. If None, inferred from the max VFH
                           value observed (safe for any horizon: at horizon=48
                           a never-crossed sample has VFH=48; that becomes
                           the cap). Explicit value overrides inference.

    Returns:
      {var: {eps: {mean, median, min, max, n_floor, n_ceiling, per_seed: [...]}}}

    Spec §3 requires n_floor (VFH == 0) and n_ceiling (VFH ≥ H_cap) explicitly
    — these are the numeric-storage form of the censoring notation in §1.
    """
    if not per_seed_pooled_vfh:
        return {}
    var_names = list(per_seed_pooled_vfh[0].keys())
    if H_cap is None:
        # Infer from data: max VFH across all seeds/vars/eps.
        all_vals = []
        for seed_dict in per_seed_pooled_vfh:
            for v_name in seed_dict:
                all_vals.extend(seed_dict[v_name].values())
        H_cap = max(all_vals) if all_vals else 12
    out: Dict[str, Dict[float, Dict[str, float | int | List[int]]]] = {}
    for v in var_names:
        out[v] = {}
        for e in eps_grid:
            per_seed = [seed_dict[v][float(e)] for seed_dict in per_seed_pooled_vfh]
            arr = np.asarray(per_seed, dtype=np.int32)
            out[v][float(e)] = {
                "mean":       float(arr.mean()),
                "median":     float(np.median(arr)),
                "min":        int(arr.min()),
                "max":        int(arr.max()),
                "n_floor":    int((arr == 0).sum()),
                "n_ceiling":  int((arr >= H_cap).sum()),
                "per_seed":   per_seed,
            }
    return out


# ---------------------------------------------------------------------------
# Pooled-consistency check (spec §4.5, non-negotiable)
# ---------------------------------------------------------------------------

def per_horizon_report_from_curves(
    curves: np.ndarray,
    var_names: Sequence[str],
    report_horizons: Sequence[int] = DEFAULT_REPORT_HORIZONS,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Report per-horizon MAE + NRMSE at each of `report_horizons` by truncating
    a single trained-at-max-horizon output.

    A model trained at horizon=48 emits a full 48-step forecast for every
    sample. To report performance "at h=24", we take the model's step-24
    prediction (index 23) directly. This is Davidson/Gaibie's reporting
    convention translated to a single trained model.

    Args:
      curves          : (S, H, d) per-sample NRMSE curves (already reduced
                        over nodes). H must be >= max(report_horizons).
      var_names       : d variable names.
      report_horizons : which horizons to report. Any horizon > H is skipped
                        with a debug log.

    Returns:
      {horizon: {var: {"mean_nrmse", "median_nrmse", "std_nrmse",
                       "p90_nrmse"}}} — one row per (horizon, variable).
    """
    S, H, d = curves.shape
    if len(var_names) != d:
        raise ValueError(f"var_names length {len(var_names)} != d {d}")
    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for h in report_horizons:
        if h > H:
            log.warning(f"report_horizons {h} > available H={H}, skipping")
            continue
        step_idx = h - 1   # 1-based horizon → 0-based index
        step_slice = curves[:, step_idx, :]   # (S, d) — per-sample NRMSE at that step
        out[int(h)] = {}
        for v_idx, v_name in enumerate(var_names):
            vals = step_slice[:, v_idx]
            out[int(h)][v_name] = {
                "mean_nrmse":   float(vals.mean()),
                "median_nrmse": float(np.median(vals)),
                "std_nrmse":    float(vals.std()),
                "p90_nrmse":    float(np.percentile(vals, 90)),
            }
    return out


def aggregate_vfh_across_seeds(
    per_seed_summaries: List[Dict],
    H_cap: int,
    var_names: Sequence[str],
    eps_grid: Sequence[float] = DEFAULT_EPS_GRID,
) -> Dict[str, Dict[float, Dict[str, float | List[float] | Dict[str, Dict]]]]:
    """
    Deliverable B cross-seed aggregator (spec §4.3).

    Combines per-seed summarise_vfh() outputs into a model-level summary:
    median-of-seed-medians, IQR of seed medians, per-seed censoring
    fractions, survival-curve mean across seeds. This is the anti-
    pseudoreplication rail from spec §2: within-seed samples are overlapping
    windows (not independent); we summarise per seed, then aggregate the
    seed-level medians.

    Args:
      per_seed_summaries: list of dicts (one per seed) as returned by
                          summarise_vfh(). Every dict must share var_names
                          and eps_grid.
      H_cap             : forecast horizon.
      var_names         : d variable names.
      eps_grid          : ε grid.

    Returns:
      {var: {eps: {median_of_seed_medians, iqr_low_of_medians,
                   iqr_high_of_medians, mean_survival_curve,
                   per_seed: {seed_idx: {median, iqr_low, iqr_high,
                                          pct_floor, pct_ceiling}}}}}
    """
    if not per_seed_summaries:
        return {}
    out: Dict = {}
    for v in var_names:
        out[v] = {}
        for e in eps_grid:
            e_key = float(e)
            seed_medians = np.array(
                [s[v][e_key]["median"] for s in per_seed_summaries], dtype=np.float32
            )
            seed_pct_floor = np.array(
                [s[v][e_key]["pct_floor"] for s in per_seed_summaries], dtype=np.float32
            )
            seed_pct_ceil = np.array(
                [s[v][e_key]["pct_ceiling"] for s in per_seed_summaries], dtype=np.float32
            )
            # Survival curves: mean across seeds if present
            surv_curves = [
                s[v][e_key].get("survival_curve") for s in per_seed_summaries
                if s[v][e_key].get("survival_curve") is not None
            ]
            mean_surv = (
                np.mean(np.array(surv_curves, dtype=np.float32), axis=0).tolist()
                if surv_curves else None
            )
            out[v][e_key] = {
                "median_of_seed_medians": float(np.median(seed_medians)),
                "iqr_low_of_medians":     float(np.percentile(seed_medians, 25)),
                "iqr_high_of_medians":    float(np.percentile(seed_medians, 75)),
                "mean_pct_floor":          float(seed_pct_floor.mean()),
                "mean_pct_ceiling":        float(seed_pct_ceil.mean()),
                "mean_survival_curve":     mean_surv,
                "per_seed": {
                    str(i): {
                        "median":      float(seed_medians[i]),
                        "pct_floor":   float(seed_pct_floor[i]),
                        "pct_ceiling": float(seed_pct_ceil[i]),
                    }
                    for i in range(len(per_seed_summaries))
                },
            }
    return out


def pooled_curve_matches(
    per_sample_curves: np.ndarray,
    pooled_curve: np.ndarray,
    rtol: float = 1e-3,
) -> Tuple[bool, float]:
    """
    Verify that the S-axis mean of per-sample curves reproduces the stored
    pooled curve within relative tolerance.

    NOTE: the pooled curve stored in test_metrics.json is
    `sqrt(mean over samples of (mean over nodes of squared error))`, while
    the S-axis mean of per-sample curves is
    `mean_s(sqrt(mean_N(sq_error_at_s)))`. These are NOT algebraically
    identical because sqrt does not commute with mean. The spec's rtol=1e-3
    is loose enough to accommodate this: the actual pooled formulation is a
    root-mean-square across all (s, N) pairs, which we should reconstruct
    to compare fairly.

    Args:
      per_sample_curves : (S, H, d) — per-sample NRMSE
      pooled_curve      : (H, d) — pooled curve to compare against
      rtol              : relative tolerance

    Returns:
      (ok, max_relative_error)

    The correct pooled reconstruction from per-sample squared errors is
    `sqrt(mean_s(per_sample_curve² weighted properly))`. Because
    per_sample_curves are already sqrt(mean_N(sq)), squaring them recovers
    mean_N(sq) per sample; mean_s of that is mean_(s,N)(sq); sqrt gives the
    stored pooled curve.
    """
    if per_sample_curves.ndim != 3 or pooled_curve.ndim != 2:
        raise ValueError(
            f"per_sample shape {per_sample_curves.shape} vs pooled {pooled_curve.shape}"
        )
    reconstructed = np.sqrt((per_sample_curves ** 2).mean(axis=0))   # (H, d)
    if reconstructed.shape != pooled_curve.shape:
        raise ValueError(
            f"reconstructed {reconstructed.shape} vs pooled {pooled_curve.shape}"
        )
    denom = np.maximum(np.abs(pooled_curve), 1e-8)
    rel_err = np.abs(reconstructed - pooled_curve) / denom
    max_rel = float(rel_err.max())
    return max_rel <= rtol, max_rel
