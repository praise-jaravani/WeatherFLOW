"""
Graph construction utilities shared across model families.

Used both during preprocessing (building the static adjacency matrix cached in
preprocessing/cache/) and inside model forward passes that recompute the graph
dynamically (e.g. adaptive graph convolution in AGCRN).

Distinct from preprocessing/graph.py, which orchestrates the pipeline-level
static graph construction. This module provides the underlying math that both
preprocessing and model-internal code can import independently.
"""

import numpy as np
import torch
from typing import Optional


def haversine_distance_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Compute pairwise haversine distances between N stations.

    Args:
        lats, lons: arrays of shape (N,) in decimal degrees.

    Returns:
        Distance matrix of shape (N, N) in kilometres.
    """
    raise NotImplementedError


def distance_to_adjacency(
    dist_matrix: np.ndarray,
    threshold_km: float,
    power: float = 1.0,
    normalise: bool = True,
) -> np.ndarray:
    """
    Convert a distance matrix to a weighted adjacency matrix.

    Pairs farther than threshold_km receive weight 0. Remaining pairs are
    weighted by 1 / dist^power. Diagonal is set to 0.

    Args:
        dist_matrix: shape (N, N).
        threshold_km: cutoff distance.
        power: distance decay exponent.
        normalise: row-normalise the result.

    Returns:
        Adjacency matrix of shape (N, N).
    """
    raise NotImplementedError


def normalise_adjacency(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalisation: D^{-1/2} A D^{-1/2}."""
    raise NotImplementedError


def numpy_to_tensor(adj: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert adjacency matrix from numpy to float32 torch tensor on device."""
    raise NotImplementedError
