"""
Statistical significance testing across technique families.

Runs the Wilcoxon signed-rank test at the end of Phase 2 to compare the
distribution of F1 scores across technique families (continuous-time,
lightweight, diffusion) and between each experimental model and the baselines.

The Wilcoxon test is used rather than a t-test because the 10 runs per model
are not independent samples from a population — they are the same model on the
same data with different initialisations. The test does not assume normality
of the metric distribution.

Results are stored in the Phase 2 summary output.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def wilcoxon_test(
    scores_a: np.ndarray, scores_b: np.ndarray, alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test comparing two sets of per-run scores.

    Args:
        scores_a, scores_b: arrays of shape (n_runs,) — e.g. F1 across 10 seeds.
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        (statistic, p_value)
    """
    raise NotImplementedError


def compare_families(
    family_scores: Dict[str, np.ndarray]
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Run pairwise Wilcoxon tests across all technique family pairs.

    Args:
        family_scores: dict mapping family name → array of shape (n_runs,).

    Returns:
        Dict mapping (family_a, family_b) → (statistic, p_value).
    """
    raise NotImplementedError


def compare_to_baselines(
    model_scores: Dict[str, np.ndarray],
    baseline_scores: Dict[str, np.ndarray],
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Compare each experimental model against each baseline.

    Returns:
        Dict mapping (model_name, baseline_name) → (statistic, p_value).
    """
    raise NotImplementedError
