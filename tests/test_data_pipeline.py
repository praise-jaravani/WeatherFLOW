"""
Tests for the data pipeline fixes introduced in June 2026:

  - Manifest-based cache invalidation (stable across mtime changes)
  - _era5_month_list extracts (year, month) tuples from filenames
  - verify_cache_coverage detects stale and ok states
  - Meta includes manifest, date_range, n_months after a pipeline run
  - Incremental ERA5 parser (_parse_one_month, parse_era5_grid)
  - Future-month guard in download script (_is_complete_month)
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_ds(n_times: int = 24, seed: int = 0) -> xr.Dataset:
    """Synthetic xarray Dataset mimicking one month of ERA5 at 0.25° resolution."""
    rng = np.random.default_rng(seed)
    n_lat, n_lon = 5, 7   # small grid for fast tests
    lats = np.linspace(-22, -35, n_lat, dtype=np.float64)
    lons = np.linspace(15, 33, n_lon, dtype=np.float64)
    times = pd.date_range("2010-01-01", periods=n_times, freq="h")
    shape = (n_times, n_lat, n_lon)
    data = {
        v: xr.DataArray(
            rng.random(shape).astype(np.float32),
            dims=["valid_time", "latitude", "longitude"],
            coords={"valid_time": times, "latitude": lats, "longitude": lons},
        )
        for v in ["t2m", "d2m", "tp", "u10", "v10", "sp"]
    }
    return xr.Dataset(data)


def _fake_dir_with_months(tmp_path: Path, year_months: list) -> Path:
    """Create empty .nc stub files for the given (year, month) list."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for y, m in year_months:
        (tmp_path / f"era5_southafrica_{y}_{m:02d}.nc").touch()
    return tmp_path


# ---------------------------------------------------------------------------
# _manifest_fingerprint
# ---------------------------------------------------------------------------

class TestManifestFingerprint:

    def test_returns_sorted_stems(self, tmp_path):
        from preprocessing.pipeline import _manifest_fingerprint
        (tmp_path / "era5_southafrica_2010_02.nc").touch()
        (tmp_path / "era5_southafrica_2010_01.nc").touch()
        result = _manifest_fingerprint(tmp_path, "era5_southafrica_*.nc")
        assert result == ["era5_southafrica_2010_01", "era5_southafrica_2010_02"]

    def test_stable_after_mtime_touch(self, tmp_path):
        from preprocessing.pipeline import _manifest_fingerprint
        f = tmp_path / "era5_southafrica_2010_01.nc"
        f.touch()
        m1 = _manifest_fingerprint(tmp_path, "era5_southafrica_*.nc")
        time.sleep(0.02)
        f.touch()
        m2 = _manifest_fingerprint(tmp_path, "era5_southafrica_*.nc")
        assert m1 == m2

    def test_changes_when_new_file_added(self, tmp_path):
        from preprocessing.pipeline import _manifest_fingerprint
        (tmp_path / "era5_southafrica_2010_01.nc").touch()
        m1 = _manifest_fingerprint(tmp_path, "era5_southafrica_*.nc")
        (tmp_path / "era5_southafrica_2010_02.nc").touch()
        m2 = _manifest_fingerprint(tmp_path, "era5_southafrica_*.nc")
        assert m1 != m2

    def test_empty_directory(self, tmp_path):
        from preprocessing.pipeline import _manifest_fingerprint
        assert _manifest_fingerprint(tmp_path, "era5_southafrica_*.nc") == []


# ---------------------------------------------------------------------------
# _era5_month_list
# ---------------------------------------------------------------------------

class TestEra5MonthList:

    def test_extracts_year_month_tuples(self, tmp_path):
        from preprocessing.pipeline import _era5_month_list
        _fake_dir_with_months(tmp_path, [(2010, 1), (2010, 2), (2011, 12)])
        result = _era5_month_list(tmp_path)
        assert result == [(2010, 1), (2010, 2), (2011, 12)]

    def test_sorted_output(self, tmp_path):
        from preprocessing.pipeline import _era5_month_list
        _fake_dir_with_months(tmp_path, [(2011, 3), (2010, 1), (2010, 12)])
        result = _era5_month_list(tmp_path)
        assert result == sorted(result)

    def test_empty_directory(self, tmp_path):
        from preprocessing.pipeline import _era5_month_list
        assert _era5_month_list(tmp_path) == []

    def test_ignores_non_era5_files(self, tmp_path):
        from preprocessing.pipeline import _era5_month_list
        _fake_dir_with_months(tmp_path, [(2010, 1)])
        (tmp_path / "README.txt").touch()
        (tmp_path / "adj.npy").touch()
        result = _era5_month_list(tmp_path)
        assert result == [(2010, 1)]


# ---------------------------------------------------------------------------
# verify_cache_coverage
# ---------------------------------------------------------------------------

class TestVerifyCacheCoverage:

    def test_no_cache_returns_no_cache(self):
        from preprocessing.pipeline import verify_cache_coverage
        with patch("preprocessing.pipeline._cache_exists", return_value=False):
            result = verify_cache_coverage("era5")
        assert result["status"] == "no_cache"
        assert result["n_cached"] == 0
        assert result["missing_from_cache"] == []

    def test_ok_when_cache_and_disk_match(self, tmp_path):
        from preprocessing.pipeline import verify_cache_coverage
        months = [(2010, 1), (2010, 2), (2010, 3)]
        _fake_dir_with_months(tmp_path, months)
        fake_meta = {"manifest": months}

        with patch("preprocessing.pipeline._cache_exists", return_value=True), \
             patch("preprocessing.pipeline.load_cache", return_value={"meta": fake_meta}), \
             patch("preprocessing.pipeline.ERA5_RAW_DIR", tmp_path):
            result = verify_cache_coverage("era5")

        assert result["status"] == "ok"
        assert result["n_cached"] == 3
        assert result["n_on_disk"] == 3
        assert result["missing_from_cache"] == []
        assert result["extra_in_cache"] == []

    def test_stale_when_disk_has_extra_months(self, tmp_path):
        from preprocessing.pipeline import verify_cache_coverage
        cached_months = [(2010, 1), (2010, 2)]
        disk_months   = [(2010, 1), (2010, 2), (2010, 3), (2010, 4)]
        _fake_dir_with_months(tmp_path, disk_months)
        fake_meta = {"manifest": cached_months}

        with patch("preprocessing.pipeline._cache_exists", return_value=True), \
             patch("preprocessing.pipeline.load_cache", return_value={"meta": fake_meta}), \
             patch("preprocessing.pipeline.ERA5_RAW_DIR", tmp_path):
            result = verify_cache_coverage("era5")

        assert result["status"] == "stale"
        assert result["n_cached"] == 2
        assert result["n_on_disk"] == 4
        assert result["missing_from_cache"] == [(2010, 3), (2010, 4)]

    def test_extra_in_cache_detected(self, tmp_path):
        """Cache claims a month that is no longer on disk."""
        from preprocessing.pipeline import verify_cache_coverage
        cached_months = [(2010, 1), (2010, 2), (2010, 3)]
        disk_months   = [(2010, 1), (2010, 2)]
        _fake_dir_with_months(tmp_path, disk_months)
        fake_meta = {"manifest": cached_months}

        with patch("preprocessing.pipeline._cache_exists", return_value=True), \
             patch("preprocessing.pipeline.load_cache", return_value={"meta": fake_meta}), \
             patch("preprocessing.pipeline.ERA5_RAW_DIR", tmp_path):
            result = verify_cache_coverage("era5")

        assert (2010, 3) in result["extra_in_cache"]

    def test_old_cache_without_manifest_key(self, tmp_path):
        """Caches written before June 2026 have no 'manifest' key — treated as stale."""
        from preprocessing.pipeline import verify_cache_coverage
        _fake_dir_with_months(tmp_path, [(2010, 1)])
        fake_meta = {}   # no 'manifest' key

        with patch("preprocessing.pipeline._cache_exists", return_value=True), \
             patch("preprocessing.pipeline.load_cache", return_value={"meta": fake_meta}), \
             patch("preprocessing.pipeline.ERA5_RAW_DIR", tmp_path):
            result = verify_cache_coverage("era5")

        assert result["status"] == "stale"
        assert result["missing_from_cache"] == [(2010, 1)]


# ---------------------------------------------------------------------------
# run_pipeline meta — manifest, date_range, n_months written to cache
# ---------------------------------------------------------------------------

class TestPipelineMetaManifest:

    def test_meta_contains_manifest_after_run(self, tmp_path):
        """After a pipeline run, meta must include manifest, date_range, n_months."""
        from preprocessing.pipeline import run_pipeline

        months = [(2010, 1), (2010, 2)]
        era5_dir = _fake_dir_with_months(tmp_path / "raw", months)

        saved_meta = {}

        def capture_save(dataset, arrays):
            saved_meta.update(arrays.get("meta", {}))

        from tests.test_pipeline_integration import _mock_parse_return, _pipeline_config

        with patch("preprocessing.pipeline.parse_era5_files",
                   return_value=_mock_parse_return()), \
             patch("preprocessing.pipeline.save_cache", side_effect=capture_save), \
             patch("preprocessing.pipeline._write_hash"), \
             patch("preprocessing.pipeline.ERA5_RAW_DIR", era5_dir):
            run_pipeline(_pipeline_config(), force_rerun=True)

        assert "manifest" in saved_meta, "manifest missing from saved meta"
        assert "date_range" in saved_meta, "date_range missing from saved meta"
        assert "n_months" in saved_meta, "n_months missing from saved meta"

    def test_date_range_format(self, tmp_path):
        """date_range.start and .end must be 'YYYY-MM' strings."""
        from preprocessing.pipeline import run_pipeline

        months = [(2010, 1), (2010, 3)]
        era5_dir = _fake_dir_with_months(tmp_path / "raw", months)

        saved_meta = {}

        def capture_save(dataset, arrays):
            saved_meta.update(arrays.get("meta", {}))

        from tests.test_pipeline_integration import _mock_parse_return, _pipeline_config

        with patch("preprocessing.pipeline.parse_era5_files",
                   return_value=_mock_parse_return()), \
             patch("preprocessing.pipeline.save_cache", side_effect=capture_save), \
             patch("preprocessing.pipeline._write_hash"), \
             patch("preprocessing.pipeline.ERA5_RAW_DIR", era5_dir):
            run_pipeline(_pipeline_config(), force_rerun=True)

        dr = saved_meta.get("date_range", {})
        assert dr.get("start") == "2010-01"
        assert dr.get("end") == "2010-03"

    def test_n_months_matches_manifest_length(self, tmp_path):
        from preprocessing.pipeline import run_pipeline

        months = [(2010, m) for m in range(1, 7)]   # 6 months
        era5_dir = _fake_dir_with_months(tmp_path / "raw", months)

        saved_meta = {}

        def capture_save(dataset, arrays):
            saved_meta.update(arrays.get("meta", {}))

        from tests.test_pipeline_integration import _mock_parse_return, _pipeline_config

        with patch("preprocessing.pipeline.parse_era5_files",
                   return_value=_mock_parse_return()), \
             patch("preprocessing.pipeline.save_cache", side_effect=capture_save), \
             patch("preprocessing.pipeline._write_hash"), \
             patch("preprocessing.pipeline.ERA5_RAW_DIR", era5_dir):
            run_pipeline(_pipeline_config(), force_rerun=True)

        assert saved_meta.get("n_months") == len(saved_meta.get("manifest", []))


# ---------------------------------------------------------------------------
# _parse_one_month (incremental ERA5 parser)
# ---------------------------------------------------------------------------

class TestParseOneMonth:

    def test_output_shape(self, tmp_path, monkeypatch):
        from preprocessing.era5_parser import _parse_one_month

        ds = _fake_ds(n_times=24)
        monkeypatch.setattr("preprocessing.era5_parser._open_zipped_nc",
                            lambda path: ds)

        f = tmp_path / "era5_southafrica_2010_01.nc"
        f.touch()
        lat_indices = list(range(0, 5, 1))   # all 5 lats
        lon_indices = list(range(0, 7, 1))   # all 7 lons
        var_names = ["t2m", "d2m", "tp", "u10", "v10", "sp"]
        chunk, ts = _parse_one_month(f, lat_indices, lon_indices, var_names)

        assert chunk.shape == (24, 35, 6)   # 5×7=35 nodes, 6 vars
        assert chunk.dtype == np.float32
        assert len(ts) == 24

    def test_stride_subsampling(self, tmp_path, monkeypatch):
        from preprocessing.era5_parser import _parse_one_month

        ds = _fake_ds(n_times=12)
        monkeypatch.setattr("preprocessing.era5_parser._open_zipped_nc",
                            lambda path: ds)

        f = tmp_path / "era5_southafrica_2010_01.nc"
        f.touch()
        lat_indices = list(range(0, 5, 2))   # stride=2 → 3 lats
        lon_indices = list(range(0, 7, 2))   # stride=2 → 4 lons
        chunk, _ = _parse_one_month(f, lat_indices, lon_indices, ["t2m", "sp"])

        assert chunk.shape == (12, 12, 2)   # 3×4=12 nodes, 2 vars

    def test_no_nan_in_output(self, tmp_path, monkeypatch):
        from preprocessing.era5_parser import _parse_one_month

        ds = _fake_ds()
        monkeypatch.setattr("preprocessing.era5_parser._open_zipped_nc",
                            lambda path: ds)

        f = tmp_path / "era5_southafrica_2010_01.nc"
        f.touch()
        chunk, _ = _parse_one_month(f, list(range(5)), list(range(7)),
                                    ["t2m", "d2m", "tp", "u10", "v10", "sp"])
        assert not np.isnan(chunk).any()


# ---------------------------------------------------------------------------
# parse_era5_grid (full incremental path)
# ---------------------------------------------------------------------------

class TestParseEra5Grid:

    def _setup_mock(self, tmp_path, monkeypatch, n_months: int = 3):
        """Create n_months stub .nc files and mock _open_zipped_nc."""
        call_counter = {"n": 0}
        months = [(2010, m) for m in range(1, n_months + 1)]
        _fake_dir_with_months(tmp_path, months)

        def mock_open(path):
            seed = call_counter["n"]
            call_counter["n"] += 1
            # Each month starts one day after the previous (ensures distinct times)
            start = f"2010-{seed+1:02d}-01"
            times = pd.date_range(start, periods=24, freq="h")
            n_lat, n_lon = 5, 7
            lats = np.linspace(-22, -35, n_lat, dtype=np.float64)
            lons = np.linspace(15, 33, n_lon, dtype=np.float64)
            rng = np.random.default_rng(seed)
            shape = (24, n_lat, n_lon)
            data = {v: xr.DataArray(
                rng.random(shape).astype(np.float32),
                dims=["valid_time", "latitude", "longitude"],
                coords={"valid_time": times, "latitude": lats, "longitude": lons},
            ) for v in ["t2m", "d2m", "tp", "u10", "v10", "sp"]}
            return xr.Dataset(data)

        monkeypatch.setattr("preprocessing.era5_parser._open_zipped_nc", mock_open)
        return str(tmp_path)

    def test_total_timesteps(self, tmp_path, monkeypatch):
        from preprocessing.era5_parser import parse_era5_grid
        directory = self._setup_mock(tmp_path, monkeypatch, n_months=3)
        X, ts, coords, var_names = parse_era5_grid(directory, lat_stride=1, lon_stride=1)
        assert X.shape[0] == 72   # 3 months × 24 hours

    def test_node_count_with_stride(self, tmp_path, monkeypatch):
        from preprocessing.era5_parser import parse_era5_grid
        directory = self._setup_mock(tmp_path, monkeypatch, n_months=2)
        X, _, coords, _ = parse_era5_grid(directory, lat_stride=2, lon_stride=2)
        n_lats = len(range(0, 5, 2))    # 3
        n_lons = len(range(0, 7, 2))    # 4
        assert X.shape[1] == n_lats * n_lons
        assert len(coords) == n_lats * n_lons

    def test_timestamps_monotonically_increasing(self, tmp_path, monkeypatch):
        from preprocessing.era5_parser import parse_era5_grid
        directory = self._setup_mock(tmp_path, monkeypatch, n_months=3)
        _, ts, _, _ = parse_era5_grid(directory, lat_stride=1, lon_stride=1)
        assert np.all(np.diff(ts.astype(np.int64)) > 0)

    def test_var_names_canonical(self, tmp_path, monkeypatch):
        from preprocessing.era5_parser import parse_era5_grid, VARIABLE_NAMES
        directory = self._setup_mock(tmp_path, monkeypatch, n_months=1)
        _, _, _, var_names = parse_era5_grid(directory, lat_stride=1, lon_stride=1)
        assert var_names == VARIABLE_NAMES

    def test_no_nan(self, tmp_path, monkeypatch):
        from preprocessing.era5_parser import parse_era5_grid
        directory = self._setup_mock(tmp_path, monkeypatch, n_months=2)
        X, _, _, _ = parse_era5_grid(directory, lat_stride=1, lon_stride=1)
        assert not np.isnan(X).any()

    def test_coords_dataframe_structure(self, tmp_path, monkeypatch):
        from preprocessing.era5_parser import parse_era5_grid
        directory = self._setup_mock(tmp_path, monkeypatch, n_months=1)
        _, _, coords, _ = parse_era5_grid(directory, lat_stride=1, lon_stride=1)
        assert set(coords.columns) >= {"node_id", "lat", "lon"}
        assert coords["node_id"].is_unique

    def test_empty_directory_raises(self, tmp_path):
        from preprocessing.era5_parser import parse_era5_grid
        with pytest.raises(FileNotFoundError):
            parse_era5_grid(str(tmp_path))


# ---------------------------------------------------------------------------
# _is_complete_month (download script)
# ---------------------------------------------------------------------------

class TestIsCompleteMonth:

    def test_year_2000_is_complete(self):
        from scripts.download_era5 import _is_complete_month
        assert _is_complete_month(2000, 1) is True

    def test_year_2020_is_complete(self):
        from scripts.download_era5 import _is_complete_month
        assert _is_complete_month(2020, 6) is True

    def test_current_month_is_incomplete(self):
        from datetime import date
        from scripts.download_era5 import _is_complete_month
        today = date.today()
        assert _is_complete_month(today.year, today.month) is False

    def test_next_month_is_incomplete(self):
        from datetime import date
        from scripts.download_era5 import _is_complete_month
        today = date.today()
        if today.month == 12:
            y, m = today.year + 1, 1
        else:
            y, m = today.year, today.month + 1
        assert _is_complete_month(y, m) is False

    def test_previous_month_is_complete(self):
        from datetime import date
        from scripts.download_era5 import _is_complete_month
        today = date.today()
        if today.month == 1:
            y, m = today.year - 1, 12
        else:
            y, m = today.year, today.month - 1
        # Previous month always ended > 5 days ago
        assert _is_complete_month(y, m) is True

    def test_december_wraps_year_correctly(self):
        from scripts.download_era5 import _is_complete_month
        # December 2023 ended 2024-01-01 — well in the past
        assert _is_complete_month(2023, 12) is True
