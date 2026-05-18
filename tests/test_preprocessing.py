"""
Tests for the preprocessing pipeline.

Covers: parsing, imputation, split ordering (leakage safety), labelling,
normalisation, graph construction, and DataLoader output shapes.
"""

import pytest
import numpy as np


class TestSAWSParser:
    def test_parse_returns_dataframe(self):
        raise NotImplementedError

    def test_parse_long_format_columns(self):
        raise NotImplementedError


class TestImpute:
    def test_no_missing_values_after_imputation(self):
        raise NotImplementedError

    def test_sparse_stations_excluded(self):
        raise NotImplementedError


class TestSplit:
    def test_chronological_order_preserved(self):
        raise NotImplementedError

    def test_split_ratios(self):
        raise NotImplementedError

    def test_no_overlap_between_splits(self):
        raise NotImplementedError


class TestLabels:
    def test_threshold_computed_on_train_only(self):
        """Verify that labels use training-partition thresholds (leakage check)."""
        raise NotImplementedError

    def test_binary_values(self):
        raise NotImplementedError

    def test_positive_rate_near_five_percent(self):
        raise NotImplementedError


class TestNormalise:
    def test_train_mean_near_zero(self):
        raise NotImplementedError

    def test_train_std_near_one(self):
        raise NotImplementedError

    def test_denormalise_recovers_original(self):
        raise NotImplementedError

    def test_stats_from_train_only(self):
        """Verify normalisation stats exclude val/test (leakage check)."""
        raise NotImplementedError


class TestGraph:
    def test_adjacency_shape(self):
        raise NotImplementedError

    def test_no_self_loops(self):
        raise NotImplementedError

    def test_threshold_produces_sparse_matrix(self):
        raise NotImplementedError


class TestDataset:
    def test_output_shapes_regression(self):
        raise NotImplementedError

    def test_output_shapes_classification(self):
        raise NotImplementedError

    def test_sliding_window_length(self):
        raise NotImplementedError
