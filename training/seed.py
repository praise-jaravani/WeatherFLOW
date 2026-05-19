"""
Reproducibility utility: fix all random seeds in one call.

Seeds for the 10 runs per experiment: 42 through 51 inclusive.
"""

import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch (CPU + CUDA) random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
