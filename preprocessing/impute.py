"""
Missing value imputation using Inverse Distance Weighting (IDW).

Stations with more than `exclusion_threshold` (default 5%) missing values
across the full time range are excluded entirely before imputation begins.
Remaining gaps at surviving stations are filled by a weighted average of
the k nearest neighbouring stations, with weights proportional to 1/distance.

This module operates on the full (unsplit) dataset — exclusion and imputation
are applied before the chronological split in split.py.
"""

import numpy as np
import pandas as pd


def exclude_sparse_stations(
    df: pd.DataFrame, exclusion_threshold: float = 0.05
) -> pd.DataFrame:
    """Remove stations where fraction of missing values exceeds the threshold."""
    raise NotImplementedError


def idw_impute(
    df: pd.DataFrame,
    station_coords: pd.DataFrame,
    k_neighbours: int = 5,
    power: float = 2.0,
) -> pd.DataFrame:
    """
    Fill missing values at each station using IDW from the k nearest neighbours.

    Args:
        df: long-format DataFrame (station_id, date, variable, value).
        station_coords: DataFrame with columns [station_id, lat, lon].
        k_neighbours: number of nearest stations to use.
        power: distance decay exponent (default 2 → inverse-square weighting).

    Returns:
        DataFrame with no missing values.
    """
    raise NotImplementedError
