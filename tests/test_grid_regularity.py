"""
B0 GATE for climode_faithful integration
=========================================

ClimODE (Verma et al., ICLR 2024) assumes gridded H×W spatial data. Its
architecture uses `torch.gradient()` for spatial derivatives, ResNet 2D
convolutions in the flow-velocity network, and a KQV attention operating
over the grid. None of that applies to an irregular graph.

This test verifies that our 999-node ERA5 subgrid IS a regular H×W lat-lon
grid, and records the exact index permutation used to reshape linear node
tensors ↔ 2D grid tensors. If this test breaks (e.g. someone changes the
lat/lon stride, adds/removes nodes, or replaces the ERA5 subset), the
climode_faithful adapter will silently break — do NOT hand-roll a graph
substitute; STOP and reassess.

Expected: 27 lats × 37 lons = 999, 0.5° uniform spacing on both axes,
bounding box S=-35°, N=-22°, W=15°, E=33° (matches
preprocessing/pipeline.py + specs/era5_data_notes.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


CACHE = Path(__file__).resolve().parent.parent / "preprocessing" / "cache" / "era5"

EXPECTED_H = 27         # unique latitudes
EXPECTED_W = 37         # unique longitudes
EXPECTED_N = EXPECTED_H * EXPECTED_W   # 999
EXPECTED_SPACING_DEG = 0.5
LAT_BOUNDS = (-35.0, -22.0)
LON_BOUNDS = (15.0, 33.0)


def _require_cache():
    coords_path = CACHE / "coords.npy"
    if not coords_path.exists():
        pytest.skip(
            f"grid regularity check requires preprocessing cache at {coords_path} "
            "(rebuild with: python cli.py --model agcrn --dataset era5 --runs 0)"
        )
    return np.load(coords_path)


def test_shape_and_count():
    coords = _require_cache()
    assert coords.shape == (EXPECTED_N, 2), f"expected ({EXPECTED_N}, 2), got {coords.shape}"


def test_unique_lat_lon_counts_multiply_to_N():
    coords = _require_cache()
    lats = coords[:, 0]
    lons = coords[:, 1]
    unique_lats = np.unique(lats)
    unique_lons = np.unique(lons)
    assert len(unique_lats) == EXPECTED_H, f"unique lats: {len(unique_lats)}, expected {EXPECTED_H}"
    assert len(unique_lons) == EXPECTED_W, f"unique lons: {len(unique_lons)}, expected {EXPECTED_W}"
    assert len(unique_lats) * len(unique_lons) == len(coords), (
        f"H*W = {len(unique_lats) * len(unique_lons)} != N = {len(coords)}"
    )


def test_uniform_spacing():
    coords = _require_cache()
    unique_lats = np.unique(coords[:, 0])
    unique_lons = np.unique(coords[:, 1])
    lat_diffs = np.diff(unique_lats)
    lon_diffs = np.diff(unique_lons)
    assert np.allclose(lat_diffs, EXPECTED_SPACING_DEG, atol=1e-4), (
        f"lat spacing not uniform at {EXPECTED_SPACING_DEG}°: min={lat_diffs.min()}, max={lat_diffs.max()}"
    )
    assert np.allclose(lon_diffs, EXPECTED_SPACING_DEG, atol=1e-4), (
        f"lon spacing not uniform at {EXPECTED_SPACING_DEG}°: min={lon_diffs.min()}, max={lon_diffs.max()}"
    )


def test_bounding_box():
    coords = _require_cache()
    unique_lats = np.unique(coords[:, 0])
    unique_lons = np.unique(coords[:, 1])
    assert unique_lats.min() == pytest.approx(LAT_BOUNDS[0], abs=1e-4), unique_lats.min()
    assert unique_lats.max() == pytest.approx(LAT_BOUNDS[1], abs=1e-4), unique_lats.max()
    assert unique_lons.min() == pytest.approx(LON_BOUNDS[0], abs=1e-4), unique_lons.min()
    assert unique_lons.max() == pytest.approx(LON_BOUNDS[1], abs=1e-4), unique_lons.max()


def test_cartesian_product_complete():
    """Every (lat, lon) pair from the Cartesian product must appear exactly once."""
    coords = _require_cache()
    unique_lats = set(np.unique(coords[:, 0]).round(6))
    unique_lons = set(np.unique(coords[:, 1]).round(6))
    expected = {(la, lo) for la in unique_lats for lo in unique_lons}
    on_disk = set(map(tuple, coords.round(6)))
    missing = expected - on_disk
    extra = on_disk - expected
    assert not missing, f"{len(missing)} (lat, lon) pairs missing from grid"
    assert not extra, f"{len(extra)} unexpected (lat, lon) pairs present"


def test_row_order_is_lat_major_north_first():
    """Confirm the row ordering: coords is lat-major, with latitudes DESCENDING
    (north-first, so coords[0] is the north-west corner at lat=-22°, lon=15°)
    and longitudes ASCENDING within each latitude band. This is the exact
    permutation the climode_faithful adapter uses to reshape (N,) ↔ (H, W).

    Note: ERA5 files conventionally list latitude north-to-south, which is why
    the ascending np.unique(lats) needs to be reversed to match row order.
    """
    coords = _require_cache()
    # Row-order lats: unique in the order they appear in coords[:, 0]
    _, first_idx_lat = np.unique(coords[:, 0], return_index=True)
    row_order_lats = coords[np.sort(first_idx_lat), 0]
    _, first_idx_lon = np.unique(coords[:, 1], return_index=True)
    row_order_lons = coords[np.sort(first_idx_lon), 1]

    # Sanity: our northernmost lat is -22, southernmost -35, so lat should
    # decrease along the row axis; longitude should increase (west→east).
    assert row_order_lats[0] == pytest.approx(LAT_BOUNDS[1], abs=1e-4), (
        f"expected northern-most lat first (row-order lat[0]={row_order_lats[0]}, "
        f"expected {LAT_BOUNDS[1]})"
    )
    assert row_order_lats[-1] == pytest.approx(LAT_BOUNDS[0], abs=1e-4)
    assert row_order_lons[0] == pytest.approx(LON_BOUNDS[0], abs=1e-4)
    assert row_order_lons[-1] == pytest.approx(LON_BOUNDS[1], abs=1e-4)

    H, W = EXPECTED_H, EXPECTED_W
    assert len(row_order_lats) == H and len(row_order_lons) == W
    for i in range(len(coords)):
        expected_lat = row_order_lats[i // W]
        expected_lon = row_order_lons[i % W]
        assert coords[i, 0] == pytest.approx(expected_lat, abs=1e-4), (
            f"row {i}: lat {coords[i, 0]} != expected {expected_lat} "
            f"(would break the (N,) ↔ (H, W) reshape at row_order_lats[{i // W}])"
        )
        assert coords[i, 1] == pytest.approx(expected_lon, abs=1e-4), (
            f"row {i}: lon {coords[i, 1]} != expected {expected_lon}"
        )


def test_reshape_round_trip():
    """Ensure a (T, N, d) tensor reshapes losslessly to (T, H, W, d) and back."""
    coords = _require_cache()
    T, d = 3, 5
    N = coords.shape[0]
    H, W = EXPECTED_H, EXPECTED_W
    # Deterministic test tensor keyed by (t, n, d) so we can spot any permutation error
    x = np.arange(T * N * d, dtype=np.float32).reshape(T, N, d)
    x_grid = x.reshape(T, H, W, d)
    x_back = x_grid.reshape(T, N, d)
    assert np.array_equal(x, x_back), "reshape round-trip lost information"
    # Confirm the reshape aligns spatially with the coords
    for i in range(N):
        h_idx, w_idx = i // W, i % W
        assert x[0, i, 0] == x_grid[0, h_idx, w_idx, 0], (
            f"reshape misaligns coords: linear index {i} → (h={h_idx}, w={w_idx}) mismatch"
        )
