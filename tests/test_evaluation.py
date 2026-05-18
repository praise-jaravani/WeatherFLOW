"""
Tests for the evaluation suite.

Covers: regression metrics, classification metrics, computational metric
measurement stubs, and the Wilcoxon test on synthetic score distributions.
"""

import pytest
import numpy as np


class TestRegressionMetrics:
    def test_mae_perfect_prediction(self):
        raise NotImplementedError

    def test_rmse_greater_than_mae(self):
        raise NotImplementedError

    def test_stratified_mae_splits_correctly(self):
        raise NotImplementedError

    def test_nrmse_curve_shape(self):
        raise NotImplementedError

    def test_vpt_returns_correct_horizon(self):
        raise NotImplementedError

    def test_vpt_zero_when_always_above_epsilon(self):
        raise NotImplementedError


class TestClassificationMetrics:
    def test_perfect_precision_recall(self):
        raise NotImplementedError

    def test_auc_roc_random_is_near_half(self):
        raise NotImplementedError

    def test_confusion_matrix_shape(self):
        raise NotImplementedError


class TestComputationalMetrics:
    def test_param_count_matches_manual_count(self):
        raise NotImplementedError

    def test_inference_time_positive(self):
        raise NotImplementedError


class TestWilcoxon:
    def test_identical_distributions_high_p(self):
        raise NotImplementedError

    def test_different_distributions_low_p(self):
        raise NotImplementedError
