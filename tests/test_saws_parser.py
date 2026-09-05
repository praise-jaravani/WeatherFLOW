"""
Regression tests for preprocessing.saws_parser.

Design gates the parser is expected to enforce:
  - Locked target: 25 stations at 2013–2020 hourly, all 6 features, <10% missing.
  - Contract with pipeline.py: returns (X, timestamps, coords, var_names) with
    X shape (T, N, D), timestamps as np.datetime64[ns], coords as a DataFrame
    with at least [lat, lon] columns, and D-length var_names list.
  - Davidson-lineage variable names: t2m, tp, rh, wd, ws, sp (so labels.py's
    tp wet-hour rule works unchanged).
  - Alphabetical node ordering (deterministic across runs).
  - Fails cleanly on missing/invalid station_coordinates.csv rather than
    silently emitting NaN or wrong-order coordinates.

These tests skip if the raw Lekuba workbooks aren't checked out (they're gitignored
because of size), keeping CI green on lightweight checkouts.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "saws" / "extracted"
COORD_FILE = Path(__file__).parent.parent / "data" / "saws" / "station_coordinates.csv"

_has_raw = (RAW_DIR / "data1.xlsx").exists() and (RAW_DIR / "data2.xls").exists()
_has_coords = COORD_FILE.exists()

requires_raw = pytest.mark.skipif(
    not _has_raw, reason="Lekuba workbooks not present in data/saws/extracted/"
)
requires_coords = pytest.mark.skipif(
    not _has_coords, reason="station_coordinates.csv not present"
)


class TestConstants:
    """Cheap tests that pin the parser's compile-time contract."""

    def test_canonical_vars_are_davidson_lineage(self):
        from preprocessing.saws_parser import CANONICAL_VARS
        assert CANONICAL_VARS == ["t2m", "tp", "rh", "wd", "ws", "sp"]

    def test_tp_in_canonical_vars_so_labels_wet_hour_rule_works(self):
        """labels.py hardcodes 'tp' for its wet-hour override. If SAWS output
        renames rain to 'Rain', that rule silently no-ops."""
        from preprocessing.saws_parser import CANONICAL_VARS
        assert "tp" in CANONICAL_VARS, (
            "rain must be exposed as 'tp' so labels.py wet-hour rule fires"
        )

    def test_rename_covers_both_wind_direction_spellings(self):
        """data1 uses 'WindDirection'; data2 uses 'WindDir'. Both must map."""
        from preprocessing.saws_parser import RENAME_RAW_TO_CANONICAL
        assert RENAME_RAW_TO_CANONICAL["WindDirection"] == "wd"
        assert RENAME_RAW_TO_CANONICAL["WindDir"] == "wd"

    def test_known_fail_stations_are_named(self):
        """These three stations were verified against raw files to fail the
        10% threshold. If someone silently removes them from KNOWN_FAIL the
        parser will start emitting them; if the raw files change and they
        now pass, KNOWN_FAIL should be pruned deliberately, not accidentally."""
        from preprocessing.saws_parser import KNOWN_FAIL
        assert "PORT ELIZABETH - WK" in KNOWN_FAIL
        assert "LADYSMITH" in KNOWN_FAIL
        assert "WARDEN - HERITAGE" in KNOWN_FAIL

    def test_window_locked_to_2013_2020(self):
        from preprocessing.saws_parser import WINDOW_START, WINDOW_END
        assert WINDOW_START.year == 2013 and WINDOW_START.month == 1 and WINDOW_START.day == 1
        assert WINDOW_END.year == 2020 and WINDOW_END.month == 12 and WINDOW_END.day == 31
        assert WINDOW_END.hour == 23

    def test_threshold_matches_davidson(self):
        from preprocessing.saws_parser import MISSINGNESS_THRESHOLD
        assert MISSINGNESS_THRESHOLD == 0.10, (
            "Davidson (2022) uses <10%; changing this needs a supervisor decision"
        )


class TestCoordinatesLoader:

    def test_missing_file_raises_clear_error(self, tmp_path):
        from preprocessing.saws_parser import _load_coordinates
        with pytest.raises(FileNotFoundError, match="Station coordinates lookup missing"):
            _load_coordinates(tmp_path / "nope.csv", ["ANY"])

    def test_missing_station_row_raises(self, tmp_path):
        csv = tmp_path / "coords.csv"
        csv.write_text("station,latitude,longitude\nA,-33.0,18.0\n")
        from preprocessing.saws_parser import _load_coordinates
        with pytest.raises(ValueError, match="absent from"):
            _load_coordinates(csv, ["A", "B"])

    def test_null_lat_lon_raises(self, tmp_path):
        csv = tmp_path / "coords.csv"
        csv.write_text("station,latitude,longitude\nA,-33.0,18.0\nB,,\n")
        from preprocessing.saws_parser import _load_coordinates
        with pytest.raises(ValueError, match="missing lat/lon"):
            _load_coordinates(csv, ["A", "B"])

    def test_valid_lookup_returns_expected_columns(self, tmp_path):
        csv = tmp_path / "coords.csv"
        csv.write_text("station,latitude,longitude\nA,-33.0,18.0\nB,-34.0,19.0\n")
        from preprocessing.saws_parser import _load_coordinates
        out = _load_coordinates(csv, ["A", "B"])
        assert set(out.columns) == {"station", "lat", "lon"}
        assert len(out) == 2


@requires_raw
@requires_coords
class TestParserContract:
    """Full parse against the actual Lekuba workbooks + coordinates lookup.
    Skipped when raw data isn't checked out."""

    @pytest.fixture(scope="class")
    def parsed(self):
        from preprocessing.saws_parser import parse_saws_files
        return parse_saws_files(str(RAW_DIR))

    def test_shape_matches_locked_target(self, parsed):
        X, ts, coords, var_names = parsed
        # 2013-01-01 00:00 → 2020-12-31 23:00 inclusive = 70,128 hours
        assert X.shape == (70128, 25, 6), (
            f"expected (70128, 25, 6) — SAWS 25-station 6-feature 8-year plan; got {X.shape}"
        )

    def test_var_names_ordering(self, parsed):
        _, _, _, var_names = parsed
        assert var_names == ["t2m", "tp", "rh", "wd", "ws", "sp"]

    def test_dtype_float32(self, parsed):
        X, _, _, _ = parsed
        assert X.dtype == np.float32, (
            "float32 is the training dtype; float64 doubles RAM for a downstream shape"
        )

    def test_timestamps_canonical_hourly(self, parsed):
        _, ts, _, _ = parsed
        assert len(ts) == 70128
        assert str(ts[0])[:10] == "2013-01-01"
        assert str(ts[-1])[:19] == "2020-12-31T23:00:00"
        diffs = np.diff(ts).astype("timedelta64[h]").astype(int)
        assert (diffs == 1).all(), "timestamps must be a strictly-1h uniform grid"

    def test_coords_alphabetical_ordering(self, parsed):
        _, _, coords, _ = parsed
        names = coords["station"].tolist()
        assert names == sorted(names), (
            "node ordering must be alphabetical for deterministic runs"
        )

    def test_coords_range_within_south_africa(self, parsed):
        _, _, coords, _ = parsed
        assert coords["lat"].between(-35, -25).all(), "lat out of SA range"
        assert coords["lon"].between(15, 32).all(), "lon out of SA range"

    def test_missingness_within_threshold(self, parsed):
        """Parser promise: every retained station's worst-feature missingness
        is ≤ 10%. If this trips, either the threshold changed or the audit
        assumptions are broken."""
        X, _, coords, var_names = parsed
        per_station_worst = 100 * np.isnan(X).mean(axis=0).max(axis=-1)
        assert (per_station_worst <= 10.0).all(), (
            f"stations exceeding 10%: "
            f"{coords['station'].iloc[np.where(per_station_worst > 10)[0]].tolist()}"
        )

    def test_known_fail_stations_are_absent(self, parsed):
        """Cross-checks KNOWN_FAIL is actually being enforced end-to-end."""
        _, _, coords, _ = parsed
        names = set(coords["station"])
        assert "PORT ELIZABETH - WK" not in names
        assert "LADYSMITH" not in names
        assert "WARDEN - HERITAGE" not in names

    def test_no_all_nan_station_variable(self, parsed):
        """A whole-column NaN would break exclude_sparse_nodes' invariant and
        trip the temporal-fill assertion in idw_impute."""
        X, _, _, _ = parsed
        per_stn_var_nan = np.isnan(X).mean(axis=0)   # (N, D)
        assert (per_stn_var_nan < 1.0).all(), (
            "at least one (station, variable) column is 100% NaN"
        )

    def test_wind_direction_in_degrees(self, parsed):
        """wd should be in [0, 360) degrees, not radians. Sanity check
        that values look like SAWS raw degrees (though NaN allowed)."""
        X, _, _, var_names = parsed
        wd = X[:, :, var_names.index("wd")]
        valid = wd[~np.isnan(wd)]
        assert valid.min() >= 0.0 and valid.max() <= 360.0, (
            f"wd range [{valid.min()}, {valid.max()}] — expected degrees in [0, 360]"
        )


class TestPipelineDefaultsForSAWS:
    """The pipeline chooses default preprocessing params based on dataset.
    ERA5 uses tight ranges (300 km / 100 km / 5%); SAWS uses looser ones
    (500 km / 200 km / 10%) because the station network is sparse — 25
    stations spread across 5 provinces, with ALEXANDERBAAI 449 km from its
    nearest neighbour. At 300 km, ALEXANDERBAAI becomes an isolated node
    (zero adjacency edges), silently corrupting the graph.

    These tests pin the default choice so a future config refactor doesn't
    revert SAWS to ERA5 defaults.
    """

    def test_saws_defaults_produce_connected_graph(self):
        """Sanity: at the SAWS defaults (500km / 200km), every one of the 25
        stations must have at least one adjacency edge."""
        import numpy as np
        import pandas as pd
        from preprocessing.graph import build_adjacency_matrix

        # Coordinates loaded from the actual station file — the specific
        # geometry (ALEXANDERBAAI at Namibia border) is what makes 500 km
        # necessary rather than 300 km.
        coord_file = Path(__file__).parent.parent / "data" / "saws" / "station_coordinates.csv"
        if not coord_file.exists():
            pytest.skip("station_coordinates.csv absent")
        coords = pd.read_csv(coord_file)
        if coords[["latitude", "longitude"]].isna().any().any():
            pytest.skip("coordinates lookup has blank rows")
        coords = coords[["station", "latitude", "longitude"]].rename(
            columns={"latitude": "lat", "longitude": "lon"}
        )
        coords["node_id"] = range(len(coords))

        # SAWS defaults per preprocessing/pipeline.py
        adj = build_adjacency_matrix(coords, distance_threshold_km=500, sigma_km=200)
        degrees = (adj > 0).sum(axis=1)
        assert (degrees >= 1).all(), (
            f"{(degrees == 0).sum()} station(s) have zero edges at 500km threshold — "
            f"raising SAWS distance_threshold_km would restore connectivity. "
            f"Isolated: {coords.iloc[np.where(degrees == 0)[0]]['station'].tolist()}"
        )

    def test_era5_defaults_leave_alexanderbaai_isolated(self):
        """Regression test: this is the specific failure mode I hit. If someone
        removes the dataset-aware defaults from pipeline.py, this test should
        fail to remind them why the branch existed."""
        import numpy as np
        import pandas as pd
        from preprocessing.graph import build_adjacency_matrix

        coord_file = Path(__file__).parent.parent / "data" / "saws" / "station_coordinates.csv"
        if not coord_file.exists():
            pytest.skip("station_coordinates.csv absent")
        coords = pd.read_csv(coord_file)
        if coords[["latitude", "longitude"]].isna().any().any():
            pytest.skip("coordinates lookup has blank rows")
        coords = coords[["station", "latitude", "longitude"]].rename(
            columns={"latitude": "lat", "longitude": "lon"}
        )
        coords["node_id"] = range(len(coords))

        # ERA5 defaults
        adj = build_adjacency_matrix(coords, distance_threshold_km=300, sigma_km=100)
        degrees = (adj > 0).sum(axis=1)
        isolated = coords.iloc[np.where(degrees == 0)[0]]["station"].tolist()
        assert "ALEXANDERBAAI" in isolated, (
            "expected ALEXANDERBAAI to be isolated at 300 km — if this passes, "
            "the SAWS station coordinates may have changed and the SAWS default "
            "of 500 km may no longer be required. Re-evaluate."
        )


@requires_raw
@requires_coords
class TestParseReport:
    """Parser writes an audit CSV as a side-effect. Downstream reproducibility
    depends on this file being present and correctly-shaped."""

    def test_parse_report_written(self):
        from preprocessing.saws_parser import parse_saws_files
        parse_saws_files(str(RAW_DIR))
        report_path = RAW_DIR.parent / "audit" / "parse_report.csv"
        assert report_path.exists()
        df = pd.read_csv(report_path)
        assert "station" in df.columns
        assert "worst_missing_pct" in df.columns
        assert "kept" in df.columns
        # Every parsed station should appear in the report
        assert len(df) >= 25   # at least the 25 kept + known-fail
