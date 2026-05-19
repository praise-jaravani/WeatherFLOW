"""
Classification evaluation metrics for Phase 2.

All metrics are computed per station then aggregated across stations.
Recall is the most important metric given the severe class imbalance
(~5% positive at the 95th-percentile threshold).

Input convention: preds/targets of shape (T, N), where T = samples * H flattened.

Metrics:
  Precision  — fraction of predicted extremes that are true extremes
  Recall     — fraction of true extremes that are correctly predicted
  F1-Score   — harmonic mean of precision and recall
  AUC-ROC    — area under the ROC curve (threshold-independent)

The confusion matrix is also saved per run.
"""

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix as sk_confusion_matrix,
)
from typing import Dict


def precision_recall_f1(
    preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Compute precision, recall, and F1 per station.

    Args:
        preds:     predicted probabilities or logits, shape (T, N).
        targets:   binary ground truth, shape (T, N).
        threshold: decision boundary for converting probabilities to binary.

    Returns:
        Dict with keys 'precision', 'recall', 'f1', each shape (N,).
    """
    T, N = preds.shape
    binary_preds = (preds >= threshold).astype(int)

    precision = np.zeros(N)
    recall = np.zeros(N)
    f1 = np.zeros(N)

    for n in range(N):
        p, r, f, _ = precision_recall_fscore_support(
            targets[:, n],
            binary_preds[:, n],
            average="binary",
            zero_division=0,
        )
        precision[n] = p
        recall[n] = r
        f1[n] = f

    return {"precision": precision, "recall": recall, "f1": f1}


def auc_roc(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    AUC-ROC per station. Shape: (N,).

    Stations with no positive (or no negative) samples in the test set are
    assigned 0.5 (random baseline) since AUC-ROC is undefined there.
    """
    T, N = preds.shape
    result = np.full(N, 0.5)

    for n in range(N):
        tgt_n = targets[:, n]
        n_pos = int(tgt_n.sum())
        if 0 < n_pos < T:
            result[n] = roc_auc_score(tgt_n, preds[:, n])

    return result


def confusion_matrix_per_station(
    preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """
    Confusion matrix per station. Shape: (N, 2, 2).

    Each (2, 2) slice is [[TN, FP], [FN, TP]].
    """
    T, N = preds.shape
    binary_preds = (preds >= threshold).astype(int)
    result = np.zeros((N, 2, 2), dtype=int)

    for n in range(N):
        result[n] = sk_confusion_matrix(
            targets[:, n], binary_preds[:, n], labels=[0, 1]
        )

    return result


def compute_all(
    preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute all classification metrics and return as a flat dict (mean across stations).

    Args:
        preds:     predicted probabilities, shape (T, N) or (samples, H, N) — will be flattened.
        targets:   binary ground truth, same shape as preds.
        threshold: decision boundary.
    """
    # Flatten any leading batch/horizon dimensions down to (T, N)
    if preds.ndim == 3:
        s, h, n = preds.shape
        preds = preds.reshape(s * h, n)
        targets = targets.reshape(s * h, n)

    prf = precision_recall_f1(preds, targets, threshold)
    auc = auc_roc(preds, targets)

    return {
        "precision":     float(prf["precision"].mean()),
        "precision_std": float(prf["precision"].std()),
        "recall":        float(prf["recall"].mean()),
        "recall_std":    float(prf["recall"].std()),
        "f1":            float(prf["f1"].mean()),
        "f1_std":        float(prf["f1"].std()),
        "auc_roc":       float(auc.mean()),
        "auc_roc_std":   float(auc.std()),
    }
