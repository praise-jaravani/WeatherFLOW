"""
Parses ERA5 NetCDF (.nc) files and aligns grid values to SAWS station coordinates.

ERA5 data is on a regular lat/lon grid. This module interpolates each ERA5
variable to the exact lat/lon of each SAWS station, producing a long-format
DataFrame with the same (station_id, date, variable) structure as saws_parser.py
so the two sources can be joined directly.

Interpolation method: bilinear (configurable; nearest-neighbour as fallback).
"""

import pandas as pd
import numpy as np


def parse_era5_files(directory: str, station_coords: pd.DataFrame) -> pd.DataFrame:
    """
    Read ERA5 .nc files and interpolate to station coordinates.

    Args:
        directory: path to data/era5/raw/
        station_coords: DataFrame with columns [station_id, lat, lon]

    Returns:
        Long-format DataFrame indexed by (station_id, date, variable).
    """
    raise NotImplementedError


def bilinear_interpolate(grid_data, grid_lats, grid_lons, target_lat, target_lon):
    """Interpolate a 2-D lat/lon grid to a single target point using bilinear interpolation."""
    raise NotImplementedError
