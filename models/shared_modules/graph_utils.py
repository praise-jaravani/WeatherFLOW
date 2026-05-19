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


def haversine_distance_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Compute pairwise haversine distances between N stations.

    Args:
        lats, lons: arrays of shape (N,) in decimal degrees.

    Returns:
        Distance matrix of shape (N, N) in kilometres.
    """
    R = 6371.0
    lats_r = np.radians(lats)
    lons_r = np.radians(lons)

    # Broadcasting: (N, 1) vs (1, N)
    dlat = lats_r[:, None] - lats_r[None, :]
    dlon = lons_r[:, None] - lons_r[None, :]

    a = np.sin(dlat / 2) ** 2 + np.cos(lats_r[:, None]) * np.cos(lats_r[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


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
        Adjacency matrix of shape (N, N), float32.
    """
    N = dist_matrix.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)

    mask = (dist_matrix > 0) & (dist_matrix <= threshold_km)
    adj[mask] = 1.0 / (dist_matrix[mask] ** power)
    np.fill_diagonal(adj, 0.0)

    if normalise:
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        adj = adj / row_sums

    return adj


def normalise_adjacency(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalisation: D^{-1/2} A D^{-1/2}."""
    adj = adj.astype(np.float32)
    deg = adj.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]


def numpy_to_tensor(adj: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert adjacency matrix from numpy to float32 torch tensor on device."""
    return torch.from_numpy(adj.astype(np.float32)).to(device)
