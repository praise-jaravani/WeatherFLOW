"""
Adjacency matrix construction from station lat/lon coordinates.

Builds the static graph A of shape (N, N) that is passed to every model
alongside the feature tensor X. The default method uses inverse distance
weighting with a distance threshold: pairs of stations farther apart than
the threshold are assigned zero weight (sparse adjacency).

The construction method (distance threshold, normalisation scheme) is
configurable via the preprocessing config. This module shares geographic
distance utilities with models/shared_modules/graph_utils.py — that module
handles graph construction needed inside model forward passes (e.g. dynamic
re-weighting), while this module handles the static preprocessing-time graph.
"""

import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    raise NotImplementedError


def build_distance_matrix(station_coords: pd.DataFrame) -> np.ndarray:
    """
    Compute pairwise haversine distances between all stations.

    Args:
        station_coords: DataFrame with columns [station_id, lat, lon], N rows.

    Returns:
        Distance matrix of shape (N, N) in kilometres.
    """
    raise NotImplementedError


def build_adjacency_matrix(
    station_coords: pd.DataFrame,
    distance_threshold_km: float = 300.0,
    normalise: bool = True,
) -> np.ndarray:
    """
    Build a sparse adjacency matrix using IDW with a distance threshold.

    Pairs of stations farther than distance_threshold_km receive weight 0.
    Diagonal is set to 0 (no self-loops). If normalise=True, rows are
    normalised to sum to 1.

    Args:
        station_coords: DataFrame with columns [station_id, lat, lon].
        distance_threshold_km: maximum distance for a non-zero edge.
        normalise: whether to row-normalise the adjacency matrix.

    Returns:
        Adjacency matrix of shape (N, N).
    """
    raise NotImplementedError
