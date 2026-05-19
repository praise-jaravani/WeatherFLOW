"""
Missing value imputation using Inverse Distance Weighting (IDW).

ERA5 reanalysis has no missing values — this module is primarily for SAWS
station data, which has real observation gaps.

Nodes with more than `exclusion_threshold` (default 5%) missing values across
the full time series are removed before imputation. Remaining gaps are filled
by a weighted average of the k nearest neighbours (weights ∝ 1/distance²).

Operates on the UNSPLIT dataset — exclusion and imputation happen before
split.py so that exclusion decisions are not biased by the split boundaries.
"""

import numpy as np
import pandas as pd
from typing import Tuple

from preprocessing.graph import build_distance_matrix


def exclude_sparse_nodes(
    X: np.ndarray,
    coords: pd.DataFrame,
    exclusion_threshold: float = 0.05,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Remove nodes where the fraction of NaN values exceeds the threshold.

    Args:
        X:                    (T, N, d) array, possibly containing NaN.
        coords:               DataFrame with [node_id, lat, lon], N rows.
        exclusion_threshold:  max allowed NaN fraction (default 5%).

    Returns:
        X_kept:     (T, N_kept, d) array.
        coords_kept: filtered coords DataFrame, N_kept rows.
    """
    T, N, d = X.shape
    nan_frac = np.isnan(X).mean(axis=(0, 2))  # (N,) — fraction over time and vars
    keep_mask = nan_frac <= exclusion_threshold
    return X[:, keep_mask, :], coords[keep_mask].reset_index(drop=True)


def idw_impute(
    X: np.ndarray,
    coords: pd.DataFrame,
    k_neighbours: int = 5,
    power: float = 2.0,
) -> np.ndarray:
    """
    Fill NaN values at each node using IDW from the k nearest neighbours.

    For each variable independently, gaps at node i at time t are filled by
    the weighted mean of the k nearest nodes that have valid values at time t.
    If all k neighbours are also NaN, the value remains NaN (rare; caught in
    the exclusion step above).

    Args:
        X:            (T, N, d) array with NaN at missing positions.
        coords:       DataFrame with [node_id, lat, lon], N rows.
        k_neighbours: number of nearest nodes to use for interpolation.
        power:        distance decay exponent (2 → inverse-square).

    Returns:
        X_filled: (T, N, d) float32 array with NaN filled.
    """
    D = build_distance_matrix(coords)  # (N, N)
    np.fill_diagonal(D, np.inf)        # exclude self from neighbour search

    # Sort neighbour indices by distance for each node
    neighbour_idx = np.argsort(D, axis=1)[:, :k_neighbours]  # (N, k)
    neighbour_D   = np.sort(D, axis=1)[:, :k_neighbours]      # (N, k)

    X_filled = X.copy().astype(np.float32)
    T, N, d = X_filled.shape

    for var in range(d):
        for t in range(T):
            missing = np.where(np.isnan(X_filled[t, :, var]))[0]
            if len(missing) == 0:
                continue
            for i in missing:
                nbrs = neighbour_idx[i]
                vals = X_filled[t, nbrs, var]
                dists = neighbour_D[i]
                valid = ~np.isnan(vals)
                if not valid.any():
                    continue
                w = (1.0 / dists[valid] ** power)
                X_filled[t, i, var] = float(np.dot(w, vals[valid]) / w.sum())

    return X_filled
