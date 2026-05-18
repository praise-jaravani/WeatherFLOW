"""
Computational efficiency metrics for Research Question 4.

Metrics are recorded in two stages (see data flow in ARCHITECTURE.md):

  Before the run loop (deterministic, model- and config-dependent):
    - Parameter count (M): total trainable parameters
    - FLOPs (G): floating-point operations per forward pass

  After the best checkpoint is loaded (reflects the trained model):
    - Inference time (ms/sample): wall-clock time averaged over 100 forward
      passes on the test set with the model in eval mode. Recorded post-training
      because models with adaptive computation (ODE solvers, diffusion chains)
      have data-dependent cost that reflects the converged model's behaviour.

All three values are logged to MLflow as parameters (not metrics) and included
in summary.json for every run.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict


def count_parameters(model: nn.Module) -> float:
    """Return total trainable parameter count in millions (M)."""
    raise NotImplementedError


def count_flops(model: nn.Module, sample_input: torch.Tensor) -> float:
    """
    Measure FLOPs per forward pass in billions (G) using torchinfo.

    Args:
        model: the nn.Module to profile.
        sample_input: a single batch tensor with the standard input shape.

    Returns:
        FLOPs in billions (G).
    """
    raise NotImplementedError


def measure_inference_time(
    adapter,
    test_loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
    n_passes: int = 100,
) -> float:
    """
    Measure mean inference time in milliseconds per sample.

    Runs n_passes forward passes on the test set with the model in eval mode
    and torch.no_grad(). Returns wall-clock ms per sample averaged over passes.

    Called AFTER the best checkpoint is loaded.

    Args:
        adapter: trained model adapter (adapter.predict() is called).
        test_loader: DataLoader for the test set.
        adj: adjacency matrix tensor on the correct device.
        device: torch device.
        n_passes: number of forward passes to average over.

    Returns:
        Mean inference time in ms per sample.
    """
    raise NotImplementedError


def record_static_metrics(model: nn.Module, sample_input: torch.Tensor) -> Dict[str, float]:
    """Compute and return parameter count and FLOPs. Called before the run loop."""
    raise NotImplementedError


def record_inference_time(
    adapter, test_loader: DataLoader, adj: torch.Tensor, device: torch.device
) -> Dict[str, float]:
    """Measure and return inference time. Called after best checkpoint is loaded."""
    raise NotImplementedError
