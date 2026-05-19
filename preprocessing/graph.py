"""
Adjacency matrix construction from node lat/lon coordinates.

Builds the static graph A of shape (N, N) passed to every model alongside the
feature tensor X. Default method: Gaussian kernel on haversine distance with a
hard sparsity cutoff at distance_threshold_km (pairs beyond this receive zero).

graph_utils.py in models/shared_modules/ shares the haversine formula and is
used for dynamic re-weighting inside model forward passes. This module handles
the static preprocessing-time graph only.
"""

import numpy as np
import pandas as pd


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def build_distance_matrix(coords: pd.DataFrame) -> np.ndarray:
    """
    Compute pairwise haversine distances between all nodes.

    Args:
        coords: DataFrame with columns [node_id, lat, lon], N rows.

    Returns:
        D: (N, N) distance matrix in kilometres.
    """
    lats = coords["lat"].values
    lons = coords["lon"].values
    N = len(lats)

    phi = np.radians(lats)
    lam = np.radians(lons)

    # Vectorised haversine via broadcasting
    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]
    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi[:, None]) * np.cos(phi[None, :]) * np.sin(dlam / 2) ** 2)
    D = 2 * 6371.0 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
    np.fill_diagonal(D, 0.0)
    return D.astype(np.float32)


def build_adjacency_matrix(
    coords: pd.DataFrame,
    distance_threshold_km: float = 300.0,
    sigma_km: float = 100.0,
    normalise: bool = True,
) -> np.ndarray:
    """
    Build a sparse adjacency matrix using a Gaussian kernel on haversine distance.

    Edge weight: w_ij = exp(-d_ij² / (2 * sigma²)) for d_ij < distance_threshold_km,
    zero otherwise. Diagonal is set to zero (no self-loops).

    Row normalisation (if enabled) maps each row to sum to 1, matching the
    convention used by AGCRN and most spectral GNN models.

    Args:
        coords:                DataFrame with [node_id, lat, lon].
        distance_threshold_km: pairs farther than this get weight 0.
        sigma_km:              Gaussian bandwidth in km.
        normalise:             row-normalise the adjacency matrix.

    Returns:
        A: (N, N) float32 adjacency matrix.
    """
    D = build_distance_matrix(coords)

    A = np.exp(-(D ** 2) / (2 * sigma_km ** 2))
    A[D >= distance_threshold_km] = 0.0
    np.fill_diagonal(A, 0.0)

    if normalise:
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid div-by-zero
        A = A / row_sums

    return A.astype(np.float32)
