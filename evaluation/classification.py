"""
Classification evaluation for Phase 2 — per-variable multi-label formulation.

Per Phase-2 plan §4:

  1. **Metrics reported.** Proposal set (accuracy, precision, recall, F1,
     AUC-ROC) plus PR-AUC (Sofaer et al. 2019, Saito & Rehmsmeier 2015) and
     CSI (Wilks 2011) for the imbalanced binary regime.

  2. **Decision threshold.** Per (model, variable) τ is selected on the
     VALIDATION split by maximising F1 in [0.05, 0.95]. Reported alongside
     metrics. 0.5 is not used — pos_weight deliberately shifts the operating
     point, so raw sigmoid outputs are not calibrated probabilities.
     AUC-ROC and PR-AUC are threshold-free and are reported unchanged.

  3. **Aggregation.** For each label variable v, confusion counts pool across
     stations. Per-station F1 / recall arrays are retained for the geographic
     analysis (like the regression per_station.npz).

  4. **NaN handling.** Stations with zero test positives for a given variable
     are skipped (F1 is undefined); we track coverage (n stations with positives)
     alongside the aggregated metric.

Input shapes
  - Preds and targets carry the label-variable axis:
    ``preds, targets ∈ (samples, H, N, d_label)`` — after per-variable model
    output and label extraction.

  - Preds are probabilities in [0, 1] (i.e. post-sigmoid). Trainer applies the
    sigmoid via the adapter's ``predict()`` before calling this module.

Public entry points
  - ``select_thresholds(val_preds, val_targets, var_names)`` returns
    ``taus: (d_label,)`` — the F1-optimal decision thresholds.
  - ``compute_all(preds, targets, var_names, taus)`` returns a dict of
    per-variable metrics plus a top-level "per_station" dict.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix as sk_confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Grid used by select_thresholds. Fine enough for stable max-F1 selection,
# coarse enough to be cheap to sweep across n variables × n stations.
_TAU_GRID = np.linspace(0.05, 0.95, 91, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Shape helper
# --------------------------------------------------------------------------- #

def _flatten_time_dims(a: np.ndarray) -> np.ndarray:
    """
    Collapse leading (samples, H) dims into T while preserving the trailing
    (N, d_label) axes. Accepts arrays already in (T, N, d_label) form.
    """
    if a.ndim == 4:
        s, h, n, d = a.shape
        return a.reshape(s * h, n, d)
    if a.ndim == 3:
        return a
    raise ValueError(f"expected 3 or 4 dims, got {a.shape}")


# --------------------------------------------------------------------------- #
# Threshold selection (validation-time)
# --------------------------------------------------------------------------- #

def _best_tau_by_f1(
    probs: np.ndarray,
    targets: np.ndarray,
    tau_grid: np.ndarray = _TAU_GRID,
) -> float:
    """
    Pooled max-F1 threshold across all (T*N) samples for a single variable.

    Returns the τ that maximises F1 on ``probs`` vs ``targets``. If there are
    no positives in ``targets``, returns 0.5 (any threshold gives F1=0; picking
    0.5 is a defensible default and matches the confusion-matrix convention).
    """
    probs = probs.ravel()
    targets = targets.ravel().astype(int)
    if targets.sum() == 0:
        return 0.5
    best_tau, best_f1 = 0.5, -1.0
    for tau in tau_grid:
        pred = (probs >= tau).astype(int)
        tp = int(((pred == 1) & (targets == 1)).sum())
        fp = int(((pred == 1) & (targets == 0)).sum())
        fn = int(((pred == 0) & (targets == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else (2 * tp) / denom
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return best_tau


def select_thresholds(
    val_preds: np.ndarray,
    val_targets: np.ndarray,
    var_names: Sequence[str],
) -> np.ndarray:
    """
    Per Phase-2 plan §4.3: per-variable decision thresholds selected on
    validation by max-F1. Returns an array of length d_label.
    """
    p = _flatten_time_dims(val_preds)           # (T, N, d_label)
    t = _flatten_time_dims(val_targets)
    d_label = p.shape[-1]
    if d_label != len(var_names):
        raise ValueError(
            f"pred trailing dim {d_label} != len(var_names) {len(var_names)}"
        )
    taus = np.zeros(d_label, dtype=np.float32)
    for v in range(d_label):
        taus[v] = _best_tau_by_f1(p[..., v], t[..., v])
    return taus


# --------------------------------------------------------------------------- #
# Per-station metrics for one variable
# --------------------------------------------------------------------------- #

def _per_station_scores(
    p_var: np.ndarray, t_var: np.ndarray, tau: float
) -> Dict[str, np.ndarray]:
    """
    Per-station arrays for one variable.

    Args:
        p_var: (T, N) probabilities
        t_var: (T, N) binary targets
        tau:   decision threshold

    Returns arrays of shape (N,); NaN where the metric is undefined at that
    station (zero positives for precision/recall/F1, zero negatives for FPR,
    fewer than one positive AND one negative for AUC-ROC).
    """
    T, N = p_var.shape
    pred_bin = (p_var >= tau).astype(int)
    t_int = t_var.astype(int)

    precision = np.full(N, np.nan, dtype=np.float32)
    recall    = np.full(N, np.nan, dtype=np.float32)
    f1        = np.full(N, np.nan, dtype=np.float32)
    csi       = np.full(N, np.nan, dtype=np.float32)
    auc_roc   = np.full(N, np.nan, dtype=np.float32)
    pr_auc    = np.full(N, np.nan, dtype=np.float32)
    accuracy  = np.zeros(N, dtype=np.float32)

    for n in range(N):
        y = t_int[:, n]
        y_pred = pred_bin[:, n]
        y_prob = p_var[:, n]
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())

        accuracy[n] = float((y_pred == y).mean())

        # precision / recall / F1 / CSI need at least one positive to be defined
        if n_pos > 0:
            p_, r_, f_, _ = precision_recall_fscore_support(
                y, y_pred, average="binary", zero_division=0,
            )
            precision[n] = p_
            recall[n]    = r_
            f1[n]        = f_
            tp = int(((y_pred == 1) & (y == 1)).sum())
            fp = int(((y_pred == 1) & (y == 0)).sum())
            fn = int(((y_pred == 0) & (y == 1)).sum())
            denom = tp + fp + fn
            csi[n] = float(tp) / denom if denom > 0 else 0.0

        # AUC-ROC and PR-AUC need at least one positive AND one negative
        if n_pos > 0 and n_neg > 0:
            try:
                auc_roc[n] = float(roc_auc_score(y, y_prob))
                pr_auc[n]  = float(average_precision_score(y, y_prob))
            except ValueError:
                pass

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "csi":       csi,
        "auc_roc":   auc_roc,
        "pr_auc":    pr_auc,
        "accuracy":  accuracy,
    }


# --------------------------------------------------------------------------- #
# Pooled confusion metrics for one variable (station-aggregated headline)
# --------------------------------------------------------------------------- #

def _pooled_scores(
    p_var: np.ndarray, t_var: np.ndarray, tau: float
) -> Dict[str, float]:
    """
    Pool across stations then compute pipeline-standard confusion metrics —
    Phase-2 plan §4.4 recipe. AUC-ROC and PR-AUC are also pooled across the
    full flat (T*N,) score vector for the reported single-number per-variable
    headline.
    """
    y_true = t_var.ravel().astype(int)
    y_prob = p_var.ravel()
    y_pred = (y_prob >= tau).astype(int)

    p_, r_, f_, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0,
    )
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    denom = tp + fp + fn
    csi = 0.0 if denom == 0 else float(tp) / denom
    acc = float((y_pred == y_true).mean())

    out = {
        "accuracy":  acc,
        "precision": float(p_),
        "recall":    float(r_),
        "f1":        float(f_),
        "csi":       csi,
        "tp": tp, "fp": fp, "fn": fn,
        "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
    }

    # Threshold-free curves. Guard against degenerate single-class targets.
    pos_count = int(y_true.sum())
    if 0 < pos_count < y_true.size:
        try:
            out["auc_roc"] = float(roc_auc_score(y_true, y_prob))
            out["pr_auc"]  = float(average_precision_score(y_true, y_prob))
        except ValueError:
            out["auc_roc"] = float("nan")
            out["pr_auc"]  = float("nan")
    else:
        out["auc_roc"] = float("nan")
        out["pr_auc"]  = float("nan")

    return out


# --------------------------------------------------------------------------- #
# Top-level compute_all
# --------------------------------------------------------------------------- #

def compute_all(
    preds: np.ndarray,
    targets: np.ndarray,
    var_names: Optional[Sequence[str]] = None,
    taus: Optional[np.ndarray] = None,
) -> Dict:
    """
    Per-variable multi-label classification metrics.

    Args:
        preds:     (samples, H, N, d_label) or (T, N, d_label) probabilities.
        targets:   same shape as preds, binary.
        var_names: optional list of length d_label. If omitted, generic
                   'var0'..'varD' are used.
        taus:      per-variable decision thresholds. If None, 0.5 is used
                   everywhere — but the caller SHOULD pass the validation-
                   selected thresholds from ``select_thresholds`` (§4.3).

    Returns:
        Dict with:
          - "per_variable": {var: {precision, recall, f1, csi, auc_roc,
                                    pr_auc, accuracy, tau, station_coverage}}
          - "macro":        {f1, precision, recall, csi, auc_roc, pr_auc}
                            — mean over label variables of the pooled scores.
                            The plan's single-number model score.
          - "taus":         list of length d_label
          - "label_variables": list of names
          - "per_station":  {metric: (N, d_label) array} — for saving to
                            per_station.npz mirroring the regression path.
    """
    p = _flatten_time_dims(preds)               # (T, N, d_label)
    t = _flatten_time_dims(targets)
    if p.shape != t.shape:
        raise ValueError(f"pred {p.shape} != target {t.shape}")
    T, N, d_label = p.shape

    if var_names is None:
        var_names = [f"var{i}" for i in range(d_label)]
    elif len(var_names) != d_label:
        raise ValueError(f"var_names has {len(var_names)} entries, need {d_label}")

    if taus is None:
        taus = np.full(d_label, 0.5, dtype=np.float32)
    taus = np.asarray(taus, dtype=np.float32).reshape(-1)
    if taus.shape[0] != d_label:
        raise ValueError(f"taus has {taus.shape[0]} entries, need {d_label}")

    per_variable: Dict[str, Dict[str, float]] = {}
    per_station_metrics: Dict[str, np.ndarray] = {
        m: np.full((N, d_label), np.nan, dtype=np.float32)
        for m in ("precision", "recall", "f1", "csi", "auc_roc", "pr_auc", "accuracy")
    }

    for v, name in enumerate(var_names):
        tau = float(taus[v])
        pooled = _pooled_scores(p[..., v], t[..., v], tau)
        station = _per_station_scores(p[..., v], t[..., v], tau)

        per_variable[name] = {
            "tau":        tau,
            "accuracy":   pooled["accuracy"],
            "precision":  pooled["precision"],
            "recall":     pooled["recall"],
            "f1":         pooled["f1"],
            "csi":        pooled["csi"],
            "auc_roc":    pooled["auc_roc"],
            "pr_auc":     pooled["pr_auc"],
            "tp": pooled["tp"], "fp": pooled["fp"],
            "fn": pooled["fn"], "tn": pooled["tn"],
            # Coverage = how many stations were scorable for F1 (had ≥ 1 positive)
            "station_coverage": int(np.isfinite(station["f1"]).sum()),
            "n_stations":       int(N),
        }
        for m, arr in station.items():
            per_station_metrics[m][:, v] = arr

    # Macro-average of the pooled per-variable metrics. NaN-safe.
    macro = {}
    for m in ("precision", "recall", "f1", "csi", "auc_roc", "pr_auc", "accuracy"):
        vals = np.array([per_variable[v][m] for v in var_names], dtype=float)
        macro[m] = float(np.nanmean(vals))

    # Also a trivial "always-negative" baseline per variable for the paper's
    # sanity floor (plan §4.6). This lives in the returned dict so the
    # analyse_matrix stage can pick it up directly.
    trivial_neg = {}
    for v, name in enumerate(var_names):
        y_true = t[..., v].ravel().astype(int)
        pos_rate = float(y_true.mean())
        trivial_neg[name] = {
            "accuracy":  float((y_true == 0).mean()),
            "positive_rate": pos_rate,
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "csi": 0.0,
        }

    return {
        "per_variable":     per_variable,
        "macro":            macro,
        "taus":             taus.tolist(),
        "label_variables":  list(var_names),
        "per_station":      per_station_metrics,
        "baseline_always_negative": trivial_neg,
    }
