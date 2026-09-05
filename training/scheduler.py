"""
Learning rate scheduling for the WeatherFLOW training harness.

Dispatcher on config["training"]["scheduler"]["type"]:
  "none"      — constant LR (AGCRN per Bai 2020, STGODE per Fang 2021)
  "multistep" — MultiStepLR(milestones, gamma) (CLCRN per Lin 2022 App D1,
                MTGODE per Jin 2022 run_multi_step.py)
  "cosine"    — CosineAnnealingWarmRestarts (legacy harness; kept as opt-in)

Paper provenance:
  AGCRN  (Bai et al. 2020):  no scheduler — constant Adam lr.
  STGODE (Fang et al. 2021): no scheduler — constant Adam lr=0.01.
  CLCRN  (Lin et al. 2022, App D1): MultiStepLR milestones=[10,20,30] gamma=0.05
           (confirmed from vendored experiments/config_clcrn.yaml; paper text
           says "first 50 epoch" but the code config has 3 milestones at [10,20,30]).
  MTGODE (Jin et al. 2022, run_multi_step.py:99 + run/metr-la.sh):
           MultiStepLR milestones=[100] gamma=0.1 (single step-drop at ep 100).
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR
from typing import Any, Dict, Optional


def build_scheduler(
    optimiser: torch.optim.Optimizer, config: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Construct the LR scheduler from the merged training config.

    Returns None for type="none" (constant LR). Callers must guard:
        if scheduler is not None:
            scheduler.step()
    """
    sched_cfg = config.get("scheduler", {})
    stype = str(sched_cfg.get("type", "cosine")).lower()

    if stype == "none":
        return None

    if stype == "multistep":
        milestones = list(sched_cfg.get("milestones", [100]))
        gamma = float(sched_cfg.get("gamma", 0.1))
        return MultiStepLR(optimiser, milestones=milestones, gamma=gamma)

    if stype == "cosine":
        return CosineAnnealingWarmRestarts(
            optimiser,
            T_0=int(sched_cfg.get("T_0", 10)),
            T_mult=int(sched_cfg.get("T_mult", 2)),
            eta_min=float(sched_cfg.get("eta_min", 1e-6)),
        )

    raise ValueError(
        f"Unknown scheduler type {stype!r}. "
        "Valid values: 'none', 'multistep', 'cosine'."
    )


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """
    Apply gradient clipping and return the pre-clip global grad norm.
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return float(total_norm)
