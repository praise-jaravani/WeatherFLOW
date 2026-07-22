"""
Binary extreme-event labelling for Phase 2 classification.

95th-percentile thresholds are computed from the TRAINING partition only and
applied across all three splits — the standard reference-period convention
(Zhang et al. 2011 ETCCDI; Lam et al. 2023 GraphCast). This must be called
after the split step.

Thresholds are saved alongside the data tensors in the .npz cache so they can
be inspected and reused without re-running the pipeline.

Precipitation wet-hour rule (§2.5 of the Phase-2 plan)
------------------------------------------------------
ETCCDI's R95p percentile is computed over wet days only (≥ 1 mm/day) because
zero/trace intervals dominate the distribution. At hourly resolution most hours
are dry at most South African stations, so a naive 95th-percentile on the tp
column can sit at trace level, silently redefining "extreme precipitation" as
"any precipitation". Where the fraction of dry hours at a station exceeds
`TP_DRY_FRACTION_TRIGGER` (default 90%) in the training set, we override the
tp threshold with the 95th percentile of wet hours only (tp ≥ 0.1 mm/h =
`TP_WET_THRESH_M`). This mirrors the ETCCDI wet-day convention translated to
hourly resolution.

If the tp variable is absent, the wet-hour rule is skipped.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# ETCCDI wet-day convention adapted to hourly resolution.
TP_WET_MM_PER_H: float = 0.1
TP_WET_THRESH_M: float = TP_WET_MM_PER_H / 1000.0    # ERA5 tp is in metres per hour
TP_DRY_FRACTION_TRIGGER: float = 0.90                # >90% dry → override
TP_MIN_WET_HOURS: int = 20                           # min wet hours for a valid percentile


def compute_thresholds(
    train_data: np.ndarray,
    percentile: float = 95.0,
) -> np.ndarray:
    """
    Compute per-variable, per-node extreme-event threshold from training data.

    Args:
        train_data: (T_train, N, d) array in physical units.
        percentile: percentile defining "extreme" (default 95).

    Returns:
        thresholds: (N, d) array — one threshold per node per variable.
    """
    return np.percentile(train_data, percentile, axis=0).astype(np.float32)


def apply_tp_wet_hour_rule(
    train_data: np.ndarray,
    thresholds: np.ndarray,
    var_names: List[str],
    percentile: float = 95.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For every station where the dry-hour fraction in the training tp series
    exceeds TP_DRY_FRACTION_TRIGGER (and at least TP_MIN_WET_HOURS wet hours
    exist), replace the tp column of `thresholds` with the 95th percentile of
    wet hours only. Otherwise the naive threshold is kept.

    Args:
        train_data: (T_train, N, d) physical-units training array.
        thresholds: (N, d) thresholds as returned by `compute_thresholds`.
        var_names:  list of length d matching the variable axis of train_data.
        percentile: percentile to use on the wet-hour subset.

    Returns:
        thresholds_out: (N, d), tp column optionally overridden.
        trigger_mask:   (N,) bool, True at stations where the override fired.
                        All False if tp is absent from var_names.
    """
    if "tp" not in var_names:
        return thresholds.copy(), np.zeros(thresholds.shape[0], dtype=bool)

    tp_idx = var_names.index("tp")
    tp = train_data[..., tp_idx]                         # (T_train, N)
    is_dry = tp < TP_WET_THRESH_M
    dry_frac = is_dry.mean(axis=0)                       # (N,)

    N = tp.shape[1]
    wet_thresh = np.full(N, np.nan, dtype=np.float32)
    for n in range(N):
        wet_hours = tp[~is_dry[:, n], n]
        if wet_hours.size >= TP_MIN_WET_HOURS:
            wet_thresh[n] = np.percentile(wet_hours, percentile)

    out = thresholds.copy()
    trigger = (dry_frac > TP_DRY_FRACTION_TRIGGER) & np.isfinite(wet_thresh)
    if trigger.any():
        out[trigger, tp_idx] = wet_thresh[trigger]
    return out, trigger


def apply_labels(
    data: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Produce binary extreme-event labels: 1 where value > threshold, else 0.

    Args:
        data:       (T, N, d) array in physical units.
        thresholds: (N, d) array computed from training data.

    Returns:
        labels: (T, N, d) int8 array.
    """
    return (data > thresholds[None, :, :]).astype(np.int8)


def label_all_splits(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    percentile: float = 95.0,
    var_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute thresholds on train, apply the tp wet-hour rule where warranted, and
    label all three splits.

    Args:
        train:      (T_train, N, d) physical-units training array.
        val:        (T_val, N, d)   physical-units val array.
        test:       (T_test, N, d)  physical-units test array.
        percentile: percentile defining "extreme". Default 95.
        var_names:  list of length d. If provided AND contains 'tp', the wet-hour
                    rule is evaluated. If omitted or 'tp' is absent, the naive
                    thresholds are used unchanged (backwards-compatible).

    Returns:
        train_labels, val_labels, test_labels, thresholds, tp_wet_trigger
        - thresholds: (N, d), tp column overridden where the rule fired
        - tp_wet_trigger: (N,) bool, per-station flag for the sign-off table
    """
    naive_thresholds = compute_thresholds(train, percentile)
    if var_names is not None:
        thresholds, tp_wet_trigger = apply_tp_wet_hour_rule(
            train, naive_thresholds, var_names, percentile
        )
    else:
        thresholds = naive_thresholds
        tp_wet_trigger = np.zeros(thresholds.shape[0], dtype=bool)

    return (
        apply_labels(train, thresholds),
        apply_labels(val,   thresholds),
        apply_labels(test,  thresholds),
        thresholds,
        tp_wet_trigger,
    )
