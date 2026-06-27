"""
Parses ERA5 ZIP-wrapped NetCDF files into a (T, N, d) numpy array.

The CDS API (post-Sept 2024) delivers ERA5 data as ZIP archives containing two
inner NetCDF4 files:
  - data_stream-oper_stepType-instant.nc  → t2m, d2m, u10, v10, sp
  - data_stream-oper_stepType-accum.nc   → tp

Parsing strategy: each monthly file is opened individually, its numpy chunk
extracted, and the xarray Dataset freed before moving to the next month.
This bounds peak memory to roughly one month of xarray (~50 MB) plus the
growing numpy accumulator, rather than loading all months as a single
concatenated xarray object (~300 MB+ per 10 months, OOM at 100+ months).

Grid-native output at 0.25° resolution: 53 lat × 73 lon = 3,869 nodes.
Use lat_stride/lon_stride to subsample (stride=2 → 999 nodes at ~0.5°).
"""

import io
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

VARIABLE_RENAMES = {
    "t2m": "t2m",
    "d2m": "d2m",
    "u10": "u10",
    "v10": "v10",
    "sp":  "sp",
    "tp":  "tp",
}

# Canonical variable order — kept consistent across all datasets
VARIABLE_NAMES = ["t2m", "d2m", "tp", "u10", "v10", "sp"]
VARIABLE_LONG_NAMES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "10m_u_wind",
    "10m_v_wind",
    "surface_pressure",
]


def _open_zipped_nc(path: Path) -> xr.Dataset:
    """
    Read a ZIP-wrapped ERA5 file and return a merged xarray Dataset.
    Writes inner NetCDF4 blobs to temp files because the netcdf4 C library
    requires a real file path (cannot read from a BytesIO buffer).
    """
    tmp_paths = []
    datasets = []
    try:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                data = zf.read(name)
                fd, tmp_path = tempfile.mkstemp(suffix=".nc")
                tmp_paths.append(tmp_path)
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                ds = xr.open_dataset(tmp_path, engine="netcdf4")
                datasets.append(ds.load())
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

    merged = xr.merge(datasets, compat="override")
    # Drop scalar/expver coords that differ between inner files
    for coord in list(merged.coords):
        if merged[coord].dims == () or coord in ("number", "expver"):
            merged = merged.drop_vars(coord)
    return merged


def _parse_one_month(
    path: Path,
    lat_indices: List[int],
    lon_indices: List[int],
    var_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Open one ZIP-wrapped monthly file, extract a numpy chunk, free xarray.

    Uses integer index selection (isel) rather than coordinate-value selection
    (sel) to avoid floating-point equality issues across files.

    Returns:
        chunk:      (T_month, N, d) float32
        timestamps: (T_month,) datetime64[ns]
    """
    ds = _open_zipped_nc(path)
    ds = ds.isel(latitude=lat_indices, longitude=lon_indices)
    ds = ds.stack(node=("latitude", "longitude"))
    chunk = np.stack([ds[v].values for v in var_names], axis=-1).astype(np.float32)
    ts = ds.valid_time.values.copy()
    ds.close()
    del ds
    return chunk, ts


def parse_era5_grid(
    directory: str,
    lat_stride: int = 2,
    lon_stride: int = 2,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Parse ERA5 files using the native grid as the node set.

    Parses one month at a time to keep peak memory bounded.  For 317 monthly
    files at stride=2 (999 nodes, d=6), peak memory is ~50 MB xarray +
    accumulating numpy list → final concatenated array of ~5.6 GB.  Run on
    HPC for the full 2000–2026 dataset.

    Default stride of 2 subsamples the 0.25° grid to ~0.5°, giving
    27 × 37 = 999 nodes — a manageable size for all 8 STGNN models.
    Use stride=1 for full 3,869-node grid (slow on larger models).

    Args:
        directory: path to the folder containing era5_southafrica_*.nc files.
        lat_stride: subsampling stride along the latitude axis.
        lon_stride: subsampling stride along the longitude axis.

    Returns:
        X:          (T, N, d) float32 array of weather variables.
        timestamps: (T,) datetime64[ns] array of UTC hours.
        coords:     DataFrame with columns [node_id, lat, lon], length N.
        var_names:  list of d variable name strings (canonical order).
    """
    files = sorted(Path(directory).glob("era5_southafrica_*.nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5 files found in {directory}")

    # Determine grid structure and variable list from the first file only.
    log.info("  Determining grid from first file: %s", files[0].name)
    ds0 = _open_zipped_nc(files[0])
    lat_indices = list(range(0, len(ds0.latitude), lat_stride))
    lon_indices = list(range(0, len(ds0.longitude), lon_stride))
    ds_sub = ds0.isel(latitude=lat_indices, longitude=lon_indices)
    ds_sub = ds_sub.stack(node=("latitude", "longitude"))
    var_names = [v for v in VARIABLE_NAMES if v in ds0]
    node_index = ds_sub.indexes["node"]
    coords = pd.DataFrame({
        "node_id": np.arange(len(node_index)),
        "lat":     [pt[0] for pt in node_index],
        "lon":     [pt[1] for pt in node_index],
    })
    ds0.close()
    del ds0, ds_sub

    # Parse each month independently: extract numpy chunk, free xarray.
    chunks: List[np.ndarray] = []
    ts_list: List[np.ndarray] = []
    n = len(files)
    for i, path in enumerate(files, 1):
        log.info("  [%d/%d] Parsing %s", i, n, path.name)
        chunk, ts = _parse_one_month(path, lat_indices, lon_indices, var_names)
        chunks.append(chunk)
        ts_list.append(ts)

    X = np.concatenate(chunks, axis=0)
    timestamps = np.concatenate(ts_list, axis=0)

    # Sort by time (files should already be in order, but guard against gaps)
    order = np.argsort(timestamps)
    return X[order], timestamps[order], coords, var_names


def parse_era5_stations(
    directory: str,
    station_coords: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Parse ERA5 files with bilinear interpolation to station lat/lon points.

    Parses one month at a time (same incremental strategy as parse_era5_grid)
    to bound memory.

    Used when SAWS station coordinates are available so both datasets share
    the same N nodes, enabling direct comparison and data fusion.

    Args:
        directory:      path to ERA5 raw folder.
        station_coords: DataFrame with columns [station_id, lat, lon].

    Returns:
        X:          (T, N, d) float32 array interpolated to N station locations.
        timestamps: (T,) datetime64[ns] array.
        var_names:  list of d variable name strings.
    """
    files = sorted(Path(directory).glob("era5_southafrica_*.nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5 files found in {directory}")

    n_stations = len(station_coords)
    var_names: Optional[List[str]] = None
    chunks: List[np.ndarray] = []
    ts_list: List[np.ndarray] = []
    n = len(files)

    for i, path in enumerate(files, 1):
        log.info("  [%d/%d] Interpolating stations: %s", i, n, path.name)
        ds = _open_zipped_nc(path)

        if var_names is None:
            var_names = [v for v in VARIABLE_NAMES if v in ds]
        d = len(var_names)
        n_times = ds.dims["valid_time"]
        chunk = np.empty((n_times, n_stations, d), dtype=np.float32)

        for j, (_, row) in enumerate(station_coords.iterrows()):
            interp = ds.interp(
                latitude=float(row["lat"]),
                longitude=float(row["lon"]),
                method="linear",
            )
            for k, var in enumerate(var_names):
                chunk[:, j, k] = interp[var].values.astype(np.float32)

        ts_list.append(ds.valid_time.values.copy())
        ds.close()
        del ds
        chunks.append(chunk)

    X = np.concatenate(chunks, axis=0)
    timestamps = np.concatenate(ts_list, axis=0)
    order = np.argsort(timestamps)
    return X[order], timestamps[order], var_names or []


def parse_era5_files(
    directory: str,
    station_coords: Optional[pd.DataFrame] = None,
    lat_stride: int = 2,
    lon_stride: int = 2,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Top-level entry point called by the preprocessing pipeline.

    If station_coords is provided, uses station-interpolated mode.
    Otherwise uses grid-native mode with the given stride.

    Returns:
        X:          (T, N, d) float32 array.
        timestamps: (T,) datetime64[ns] array.
        coords:     DataFrame with [node_id, lat, lon].
        var_names:  list of variable name strings.
    """
    if station_coords is not None:
        X, timestamps, var_names = parse_era5_stations(directory, station_coords)
        coords = station_coords.rename(columns={"station_id": "node_id"})[
            ["node_id", "lat", "lon"]
        ].reset_index(drop=True)
        return X, timestamps, coords, var_names
    return parse_era5_grid(directory, lat_stride=lat_stride, lon_stride=lon_stride)


def bilinear_interpolate(
    grid_data: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> float:
    """
    Interpolate a 2-D (lat, lon) grid to a single target point.

    Falls back to nearest-neighbour if the target is outside the grid extent.
    This is the manual implementation kept for reference; parse_era5_stations
    uses xarray.interp which is equivalent and handles edge cases.
    """
    if (target_lat < grid_lats.min() or target_lat > grid_lats.max() or
            target_lon < grid_lons.min() or target_lon > grid_lons.max()):
        lat_idx = int(np.argmin(np.abs(grid_lats - target_lat)))
        lon_idx = int(np.argmin(np.abs(grid_lons - target_lon)))
        return float(grid_data[lat_idx, lon_idx])

    lat_idx = int(np.searchsorted(-grid_lats, -target_lat))
    lon_idx = int(np.searchsorted(grid_lons, target_lon))

    lat_idx = np.clip(lat_idx, 1, len(grid_lats) - 1)
    lon_idx = np.clip(lon_idx, 1, len(grid_lons) - 1)

    lat0, lat1 = grid_lats[lat_idx - 1], grid_lats[lat_idx]
    lon0, lon1 = grid_lons[lon_idx - 1], grid_lons[lon_idx]

    t = (target_lat - lat0) / (lat1 - lat0) if lat1 != lat0 else 0.0
    u = (target_lon - lon0) / (lon1 - lon0) if lon1 != lon0 else 0.0

    return float(
        (1 - t) * (1 - u) * grid_data[lat_idx - 1, lon_idx - 1]
        + t * (1 - u) * grid_data[lat_idx,     lon_idx - 1]
        + (1 - t) * u * grid_data[lat_idx - 1, lon_idx    ]
        + t * u * grid_data[lat_idx,     lon_idx    ]
    )
