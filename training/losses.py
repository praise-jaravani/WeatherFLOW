"""
Loss functions for regression and classification tasks.

MSE  — Phase 1 regression. Standard mean squared error.

WBCE — Phase 2 classification. Weighted Binary Cross-Entropy with class weights
       computed from the training set's class distribution. At the 95th-percentile
       threshold, the positive class (extreme events) is ~5% of samples, so
       w_1 ≈ 19 × w_0. Class weights are computed once at experiment start,
       logged to MLflow, and passed to the loss function for all 10 runs.
"""

import torch
import torch.nn as nn
import numpy as np


def mse_loss() -> nn.Module:
    """Return the MSE loss module used for Phase 1 regression."""
    raise NotImplementedError


def compute_class_weights(train_labels: np.ndarray) -> tuple:
    """
    Compute positive/negative class weights from training label distribution.

    Returns:
        (w_0, w_1) where w_1 = (1 - pos_rate) / pos_rate (inverse frequency).
    """
    raise NotImplementedError


def wbce_loss(w_0: float, w_1: float, device: torch.device) -> nn.Module:
    """
    Return a Weighted Binary Cross-Entropy loss module.

    Args:
        w_0: weight for the negative (non-extreme) class.
        w_1: weight for the positive (extreme) class.
        device: torch device the weight tensor should live on.
    """
    raise NotImplementedError
