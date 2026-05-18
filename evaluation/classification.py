"""
Classification evaluation metrics for Phase 2.

All metrics are computed per station then aggregated across stations.
Recall is the most important metric given the severe class imbalance
(~5% positive at the 95th-percentile threshold).

Metrics:
  Precision  — fraction of predicted extremes that are true extremes
  Recall     — fraction of true extremes that are correctly predicted
  F1-Score   — harmonic mean of precision and recall
  AUC-ROC    — area under the ROC curve (threshold-independent)

The confusion matrix is also saved per run.
"""

import numpy as np
from typing import Dict


def precision_recall_f1(
    preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Compute precision, recall, and F1 per station.

    Args:
        preds: predicted probabilities or logits, shape (T, N).
        targets: binary ground truth, shape (T, N).
        threshold: decision boundary for converting probabilities to binary.

    Returns:
        Dict with keys 'precision', 'recall', 'f1', each shape (N,).
    """
    raise NotImplementedError


def auc_roc(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    AUC-ROC per station. Shape: (N,).

    Uses sklearn.metrics.roc_auc_score internally.
    """
    raise NotImplementedError


def confusion_matrix_per_station(
    preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """
    Confusion matrix per station. Shape: (N, 2, 2).

    Each (2, 2) slice is [[TN, FP], [FN, TP]].
    """
    raise NotImplementedError


def compute_all(
    preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute all classification metrics and return as a flat dict (mean across stations)."""
    raise NotImplementedError
