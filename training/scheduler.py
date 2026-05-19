"""
Learning rate scheduling and gradient clipping configuration.

Default schedule: CosineAnnealingWarmRestarts (T_0, T_mult configurable).
Gradient clipping: torch.nn.utils.clip_grad_norm_ with configurable max_norm.

Both defaults follow Gaibie et al. (2024) unless a model-specific YAML config
provides overrides. The scheduler is constructed fresh for each of the 10 runs
so warm-restart cycles reset properly between seeds.
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Dict, Any


def build_scheduler(
    optimiser: torch.optim.Optimizer, config: Dict[str, Any]
) -> CosineAnnealingWarmRestarts:
    """
    Construct the LR scheduler from the merged config.

    Relevant config keys (nested under training.scheduler):
      T_0, T_mult, eta_min.
    """
    sched_cfg = config.get("scheduler", {})
    return CosineAnnealingWarmRestarts(
        optimiser,
        T_0=int(sched_cfg.get("T_0", 10)),
        T_mult=int(sched_cfg.get("T_mult", 2)),
        eta_min=float(sched_cfg.get("eta_min", 1e-6)),
    )


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """
    Apply gradient clipping and return the pre-clip global grad norm.

    Args:
        model: the model whose parameters' gradients are clipped.
        max_norm: maximum allowed gradient norm (from config).

    Returns:
        Total gradient norm before clipping (useful for logging).
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return float(total_norm)
