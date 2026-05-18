"""
Tests for model adapters.

Smoke tests: verify that each adapter's build_model(), forward(), predict(),
and get_param_count() are callable and return the correct output shapes,
without requiring a full training run or GPU.
"""

import pytest
import torch


BATCH, T, N, D, H = 2, 12, 10, 3, 12
FAKE_X = torch.randn(BATCH, T, N, D)
FAKE_ADJ = torch.eye(N)


class TestAGCRNAdapter:
    def test_forward_output_shape(self):
        raise NotImplementedError

    def test_predict_output_shape(self):
        raise NotImplementedError

    def test_param_count_positive(self):
        raise NotImplementedError


class TestCLCRNAdapter:
    def test_forward_output_shape(self):
        raise NotImplementedError

    def test_predict_output_shape(self):
        raise NotImplementedError


class TestMTGODEAdapter:
    def test_forward_output_shape(self):
        raise NotImplementedError

    def test_predict_uses_tighter_tolerance(self):
        raise NotImplementedError


class TestClimODEAdapter:
    def test_forward_output_shape(self):
        raise NotImplementedError


class TestSTLGRUAdapter:
    def test_forward_output_shape(self):
        raise NotImplementedError


class TestLiteSTGNNAdapter:
    def test_forward_output_shape(self):
        raise NotImplementedError


class TestDiffSTGAdapter:
    def test_forward_output_shape(self):
        raise NotImplementedError

    def test_predict_output_shape(self):
        raise NotImplementedError


class TestUSTDAdapter:
    def test_forward_output_shape(self):
        raise NotImplementedError

    def test_predict_output_shape(self):
        raise NotImplementedError
