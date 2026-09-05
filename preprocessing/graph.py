"""
Adjacency matrix construction from node lat/lon coordinates.

Builds the static graph A of shape (N, N) passed to every model alongside the
feature tensor X. Default method: Gaussian kernel on haversine distance with a
hard sparsity cutoff at distance_threshold_km (pairs beyond this receive zero).

Provenance and conventions
--------------------------
Thresholded-Gaussian-on-distance is the Davidson & Moodley (2022) ST-GNN paper
construction (§3.5, Euclidean distances), itself the DCRNN convention (Li et al.
2018). Ours upgrades Euclidean → haversine because SA lat/lon spans enough
degrees for Euclidean-in-degrees to systematically distort distances.

Two convention deltas vs the DCRNN lineage that must be disclosed if any model
consumes A: (1) we zero the diagonal (DCRNN keeps w_ii = 1); (2) we row-
normalise (DCRNN typically uses the kernel weights directly or symmetric
normalisation). Legitimate downstream choices; documented here so consumers
know.

A as a shared interface, not a shared method
--------------------------------------------
Whether any given model actually reads A is *per-adapter*. AGCRN and MTGODE
generate their own graphs from learned embeddings; CLCRN builds K-nearest-
neighbour spherical neighbourhoods from node coordinates. In those cases A is
inert plumbing — kept in the pipeline so downstream teammate models that
DO consume adjacency (STLGRU, DiffSTG-family, USTD-spirit) have something to
buffer. See specs/ADJACENCY_AUDIT.md for the empirical probe verdicts.

The above design means graph.py is INTENTIONALLY conservative: refinements
must be additive/opt-in. Do not change output semantics without checking every
downstream adapter that reads A as a buffer.
"""

import numpy as np
import pandas as pd


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points.

    Clip guard on sqrt(a): antipodal points and float rounding can push `a`
    above 1.0, making arcsin NaN. The vectorised path (build_distance_matrix)
    has this guard already; this scalar path was missing it — parity fix.
    """
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(2 * R * np.arcsin(np.clip(np.sqrt(a), 0, 1)))


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
