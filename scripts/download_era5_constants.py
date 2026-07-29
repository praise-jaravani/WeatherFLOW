"""
Download ERA5 static fields (land–sea mask + surface geopotential a.k.a.
orography source) for the SA bounding box at 0.5° resolution.

Once — small request (two variables, one timestamp).

Output: data/constants/era5_sa_constants.nc  (or .zip containing that)
Also writes:
    data/constants/era5_sa_lsm.npy       (H, W) float32, lsm ∈ [0, 1]
    data/constants/era5_sa_orography.npy (H, W) float32, meters

Rationale (from mission directive §1.2 + PATCHES.md open TODO):
climode_faithful adapter needs real (lsm, orography) fields for the
`const_info` input rather than the current torch.zeros((1, 2, H, W))
placeholder. Placeholder is fine for the smoke; real fields required
before any production seed starts.
"""
from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "constants"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Bounding box: N=-22, W=15, S=-35, E=33  — matches preprocessing/pipeline.py
# and specs/era5_data_notes.md. CDS "area" order is [N, W, S, E].
AREA = [-22, 15, -35, 33]
GRID = [0.5, 0.5]                          # matches lat_stride=2 → 0.5° in the pipeline

REQUEST = {
    "product_type": "reanalysis",
    "variable": ["land_sea_mask", "geopotential"],
    "year": "2020",
    "month": "01",
    "day": "01",
    "time": "00:00",
    "area": AREA,
    "grid": GRID,
    "data_format": "netcdf",
}
DATASET = "reanalysis-era5-single-levels"


def _open_maybe_zipped(path: Path) -> xr.Dataset:
    """CDS delivers zipped .nc — handle both cases."""
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"PK":
        # zipped — extract and open the inner netcdf
        with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as td:
            z.extractall(td)
            nc_files = [Path(td) / n for n in z.namelist() if n.endswith(".nc")]
            print(f"  ZIP archive contains: {[n.name for n in nc_files]}")
            # merge if multiple; for constants both fields land in one file usually
            if len(nc_files) == 1:
                return xr.load_dataset(nc_files[0])
            ds = xr.merge([xr.load_dataset(n) for n in nc_files])
            return ds
    return xr.load_dataset(path)


def main():
    out_nc = OUT_DIR / "era5_sa_constants.nc"
    if out_nc.exists():
        print(f"[skip download] {out_nc} already exists")
    else:
        print(f"Downloading ERA5 constants (SA box {AREA} @ 0.5°) → {out_nc}")
        client = cdsapi.Client()
        client.retrieve(DATASET, REQUEST, str(out_nc))
        print(f"  wrote {out_nc}  ({out_nc.stat().st_size / 1024:.1f} KB)")

    print("\nOpening dataset...")
    ds = _open_maybe_zipped(out_nc)
    print(f"  variables: {list(ds.data_vars)}")
    print(f"  dims: {dict(ds.dims)}")
    print(f"  coords: {[c for c in ds.coords]}")

    # Extract lsm + orography (geopotential/g0) to numpy arrays matching
    # the (H=27, W=37) SA grid.
    # Variable names may be 'lsm' or 'land_sea_mask'; 'z' or 'geopotential'.
    lsm_var = None
    for candidate in ("lsm", "land_sea_mask"):
        if candidate in ds.data_vars:
            lsm_var = candidate
            break
    if lsm_var is None:
        raise KeyError(f"land_sea_mask not found in {list(ds.data_vars)}")

    z_var = None
    for candidate in ("z", "geopotential"):
        if candidate in ds.data_vars:
            z_var = candidate
            break
    if z_var is None:
        raise KeyError(f"geopotential not found in {list(ds.data_vars)}")

    lsm = ds[lsm_var].squeeze().values.astype(np.float32)
    geopot = ds[z_var].squeeze().values.astype(np.float32)
    # ERA5 'z' is geopotential in m²/s² — convert to orography in meters
    # by dividing by standard gravity 9.80665.
    orography = geopot / 9.80665

    print(f"\n  lsm shape: {lsm.shape}  dtype: {lsm.dtype}  range: [{lsm.min()}, {lsm.max()}]  mean: {lsm.mean():.4f}")
    print(f"  orography shape: {orography.shape}  range: [{orography.min():.1f}, {orography.max():.1f}] m")

    # Sanity: expect (27, 37) for our grid
    assert lsm.shape == (27, 37), f"lsm shape {lsm.shape} != expected (27, 37)"
    assert orography.shape == (27, 37), f"orography shape {orography.shape} != expected (27, 37)"
    assert 0.0 <= lsm.min() and lsm.max() <= 1.0, "lsm not in [0, 1]"

    # Row order: match the coords convention in preprocessing/cache/era5/coords.npy —
    # north-first (lat DESCENDING), lon ASCENDING within each row. ERA5 defaults
    # to lat DESC already; verify by inspecting the lat coord.
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    print(f"  lat range: [{lats[0]}, {lats[-1]}]  (expect north-first: {lats[0]} > {lats[-1]})")
    print(f"  lon range: [{lons[0]}, {lons[-1]}]  (expect ascending)")
    assert lats[0] > lats[-1], "ERA5 lat should be north-first (descending)"
    assert lons[0] < lons[-1], "ERA5 lon should be ascending"

    np.save(OUT_DIR / "era5_sa_lsm.npy", lsm)
    np.save(OUT_DIR / "era5_sa_orography.npy", orography)
    print(f"\nWrote {OUT_DIR / 'era5_sa_lsm.npy'} and {OUT_DIR / 'era5_sa_orography.npy'}")


if __name__ == "__main__":
    main()
