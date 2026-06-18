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
from typing import Dict, Optional


def count_parameters(model: nn.Module) -> float:
    """Return total trainable parameter count in millions (M)."""
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n / 1e6


import logging as _logging
_log = _logging.getLogger(__name__)


def count_flops(model: nn.Module, sample_input: torch.Tensor) -> Optional[float]:
    """
    Measure FLOPs per forward pass in billions (G) using torchinfo.

    Returns None if torchinfo is unavailable, the model is untraceable (e.g.
    adaptive ODE solvers), or torchinfo reports 0 — so callers always get either
    a meaningful positive number or an explicit None rather than a silent 0.

    Args:
        model:        the nn.Module to profile.
        sample_input: a single batch tensor — must already be on the correct device.

    Returns:
        FLOPs in billions (G), or None.
    """
    try:
        from torchinfo import summary as ti_summary  # noqa: PLC0415
        was_training = model.training
        info = ti_summary(model, input_data=sample_input, verbose=0)
        if was_training:
            model.train()
        flops = info.total_mult_adds
        if not flops:
            _log.warning(
                "torchinfo reported 0 FLOPs for %s — untraceable op "
                "(e.g. odeint) or bad sample input. Reporting n/a.",
                type(model).__name__,
            )
            return None
        return flops / 1e9
    except Exception as exc:
        _log.warning(
            "FLOP counting failed for %s: %s. Reporting n/a.",
            type(model).__name__, exc,
        )
        return None


def measure_inference_time(
    adapter,
    test_loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
    n_passes: int = 100,
) -> float:
    """
    Measure mean inference time in milliseconds per sample.

    Runs n_passes forward passes over the test set in eval mode. Resets the
    DataLoader iterator if it runs out before n_passes are completed.

    Args:
        adapter:     trained model adapter (adapter.predict() is called).
        test_loader: DataLoader for the test set.
        adj:         adjacency matrix tensor on the correct device.
        device:      torch device.
        n_passes:    number of forward passes to average over.

    Returns:
        Mean inference time in ms per sample.
    """
    total_ms = 0.0
    total_samples = 0
    passes = 0
    loader_iter = iter(test_loader)

    while passes < n_passes:
        try:
            x, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(test_loader)
            x, _ = next(loader_iter)

        x = x.to(device)
        batch_size = x.shape[0]

        t0 = time.perf_counter()
        _ = adapter.predict(x, adj)
        t1 = time.perf_counter()

        total_ms += (t1 - t0) * 1000.0
        total_samples += batch_size
        passes += 1

    return total_ms / total_samples


def record_static_metrics(model: nn.Module, sample_input: torch.Tensor) -> Dict:
    """Compute and return parameter count and FLOPs. Called before the run loop.
    flops_G is None when the model is untraceable (ODE solvers, etc.)."""
    return {
        "params_M": count_parameters(model),
        "flops_G":  count_flops(model, sample_input),
    }


def record_inference_time(
    adapter, test_loader: DataLoader, adj: torch.Tensor, device: torch.device
) -> Dict[str, float]:
    """Measure and return inference time. Called after best checkpoint is loaded."""
    return {
        "inference_ms_per_sample": measure_inference_time(adapter, test_loader, adj, device),
    }
