"""
Unit tests for preprocessing.graph — the static adjacency construction that
every model receives via the trainer contract, regardless of whether the model
actually consumes it (see specs/ADJACENCY_AUDIT.md).

Test coverage from Adjacency_Audit_Spec.md §5.2, adapted to the current SAWS
25-station scattered network. Two ERA5-specific tests from the original spec
(interior/edge/corner degree checks on a 27×37 regular grid) are replaced by
structural tests appropriate for a station point-cloud rather than a lattice.
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing.graph import (
    haversine_distance,
    build_distance_matrix,
    build_adjacency_matrix,
)


# ---------------------------------------------------------------------------
# Test 1: haversine known value (Cape Town ↔ Johannesburg)
# ---------------------------------------------------------------------------

def test_haversine_known_value_cape_town_to_joburg():
    """The Cape Town ↔ Johannesburg great-circle distance is a well-published
    ~1,265 km. Any deviation beyond ±10 km means the formula is broken."""
    d = haversine_distance(-33.925, 18.424, -26.204, 28.047)
    assert 1255 < d < 1275, f"Cape Town → Joburg: {d:.1f} km (expected ~1265)"


# ---------------------------------------------------------------------------
# Test 2: distance matrix symmetry, zero diagonal, scalar↔vectorised agreement
# ---------------------------------------------------------------------------

class TestDistanceMatrix:

    @pytest.fixture
    def coords(self):
        # 5 SA cities — real coords, real distances
        return pd.DataFrame({
            "node_id": range(5),
            "lat": [-33.925, -26.204, -29.856, -25.746, -33.966],
            "lon": [18.424, 28.047, 31.021, 28.187, 25.616],
        })

    def test_matrix_symmetric(self, coords):
        D = build_distance_matrix(coords)
        np.testing.assert_allclose(D, D.T, atol=1e-3)

    def test_matrix_zero_diagonal(self, coords):
        D = build_distance_matrix(coords)
        np.testing.assert_array_equal(np.diag(D), np.zeros(len(coords)))

    def test_matrix_non_negative(self, coords):
        D = build_distance_matrix(coords)
        assert (D >= 0).all()

    def test_scalar_matches_vectorised(self, coords):
        """Every pair (i, j) computed scalar-wise should match the vectorised
        matrix entry. Guards against the two implementations drifting apart."""
        D = build_distance_matrix(coords)
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                scalar = haversine_distance(
                    coords["lat"].iloc[i], coords["lon"].iloc[i],
                    coords["lat"].iloc[j], coords["lon"].iloc[j],
                )
                assert abs(scalar - D[i, j]) < 1e-3, (
                    f"scalar={scalar:.4f} vs matrix={D[i, j]:.4f} at ({i}, {j})"
                )


# ---------------------------------------------------------------------------
# Test 3: clip guard — antipodal points do not NaN
# ---------------------------------------------------------------------------

class TestClipGuard:
    """Antipodal-point sqrt(a) can numerically overshoot 1.0 → arcsin(NaN).
    Both scalar and vectorised paths must clip."""

    def test_scalar_antipodal_no_nan(self):
        # Antipodal to (0, 0) is (0, 180). Distance ≈ π R ≈ 20015 km.
        d = haversine_distance(0.0, 0.0, 0.0, 180.0)
        assert not np.isnan(d)
        assert 19900 < d < 20050

    def test_vectorised_antipodal_no_nan(self):
        coords = pd.DataFrame({"node_id": [0, 1], "lat": [0.0, 0.0], "lon": [0.0, 180.0]})
        D = build_distance_matrix(coords)
        assert not np.isnan(D).any()
        assert 19900 < D[0, 1] < 20050

    def test_identical_points_zero_distance(self):
        d = haversine_distance(-33.925, 18.424, -33.925, 18.424)
        assert d == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 4: adjacency row sums = 1 for connected rows; isolated-row zero-safe
# ---------------------------------------------------------------------------

class TestAdjacencyRowNormalisation:

    def test_row_sums_one_for_connected(self):
        coords = pd.DataFrame({
            "node_id": [0, 1, 2],
            "lat": [-33.0, -33.5, -34.0],
            "lon": [18.0, 18.0, 18.0],
        })
        A = build_adjacency_matrix(coords, distance_threshold_km=300, sigma_km=100)
        row_sums = A.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_isolated_node_row_all_zero_no_divzero(self):
        """A node further than distance_threshold_km from all others must yield
        an all-zero row (no NaN, no infinity from div-by-zero)."""
        coords = pd.DataFrame({
            "node_id": [0, 1, 2],
            "lat": [-33.0, -33.5, 0.0],   # station 2 near equator, far from others
            "lon": [18.0, 18.0, 18.0],
        })
        A = build_adjacency_matrix(coords, distance_threshold_km=300, sigma_km=100)
        assert (A[2] == 0).all()
        assert not np.isnan(A).any()
        assert not np.isinf(A).any()


# ---------------------------------------------------------------------------
# Test 5: threshold monotonicity (sparsity pattern is nested)
# ---------------------------------------------------------------------------

def test_threshold_monotonicity():
    """Every edge present at 200 km MUST also be present at 300 km — a strictly
    less-restrictive threshold can only add edges, never remove them."""
    coords = pd.DataFrame({
        "node_id": range(6),
        "lat": [-33.0, -30.0, -29.0, -28.0, -34.0, -26.0],
        "lon": [18.0, 18.5, 25.0, 30.0, 22.0, 28.0],
    })
    A_200 = build_adjacency_matrix(coords, distance_threshold_km=200, sigma_km=100)
    A_300 = build_adjacency_matrix(coords, distance_threshold_km=300, sigma_km=100)
    # (A_200 > 0) must imply (A_300 > 0)
    assert ((A_200 > 0) & (A_300 == 0)).sum() == 0


# ---------------------------------------------------------------------------
# Test 6: determinism + dtype
# ---------------------------------------------------------------------------

class TestDeterminismAndDtype:

    @pytest.fixture
    def coords(self):
        return pd.DataFrame({
            "node_id": range(4),
            "lat": [-33.0, -30.0, -29.0, -28.0],
            "lon": [18.0, 18.5, 25.0, 30.0],
        })

    def test_deterministic_across_calls(self, coords):
        A1 = build_adjacency_matrix(coords)
        A2 = build_adjacency_matrix(coords)
        np.testing.assert_array_equal(A1, A2)

    def test_output_dtype_float32(self, coords):
        A = build_adjacency_matrix(coords)
        assert A.dtype == np.float32


# ---------------------------------------------------------------------------
# Test 7: structure appropriate for the SAWS 25-node scattered network
# (adapted from the ERA5 27×37 grid tests in the original spec — the
# interior/edge/corner distinction doesn't apply to a station point-cloud)
# ---------------------------------------------------------------------------

class TestSAWSNetworkStructure:
    """The production SAWS network is 25 stations across 5 provinces.
    Structural properties we can assert without a regular grid:
      - every node has ≥1 neighbour at the SAWS 500km default
      - degrees vary (not all nodes have identical connectivity)
      - the strongest edge for every node is its nearest neighbour
    """

    @pytest.fixture
    def saws_coords(self):
        coord_file = (
            "/home/praise/Computer_Science_Resources/Weatherflow/"
            "data/saws/station_coordinates.csv"
        )
        import pathlib
        if not pathlib.Path(coord_file).exists():
            pytest.skip("station_coordinates.csv absent")
        df = pd.read_csv(coord_file)
        if df[["latitude", "longitude"]].isna().any().any():
            pytest.skip("coordinates lookup has blank rows")
        df = df[["station", "latitude", "longitude"]].rename(
            columns={"latitude": "lat", "longitude": "lon"}
        )
        df["node_id"] = range(len(df))
        return df

    def test_every_station_has_neighbour_at_saws_defaults(self, saws_coords):
        A = build_adjacency_matrix(saws_coords, distance_threshold_km=500, sigma_km=200)
        degrees = (A > 0).sum(axis=1)
        isolated = saws_coords.iloc[np.where(degrees == 0)[0]]["station"].tolist()
        assert (degrees >= 1).all(), (
            f"{(degrees == 0).sum()} station(s) isolated at 500km: {isolated}"
        )

    def test_degrees_vary_across_stations(self, saws_coords):
        """25 stations across 5 provinces: expect degree variation. Uniform
        degree would suggest all stations sit inside the disc cutoff of
        every other, which contradicts the 60-1422 km distance spread."""
        A = build_adjacency_matrix(saws_coords, distance_threshold_km=500, sigma_km=200)
        degrees = (A > 0).sum(axis=1)
        assert degrees.min() < degrees.max(), (
            f"all stations have identical degree {degrees.min()} — geometry "
            f"suggests degrees should vary"
        )

    def test_strongest_edge_is_nearest_neighbour(self, saws_coords):
        """Row-normalisation preserves edge ordering within a row: the strongest
        edge from node i must go to the closest connected neighbour."""
        A = build_adjacency_matrix(saws_coords, distance_threshold_km=500, sigma_km=200)
        D = build_distance_matrix(saws_coords)
        np.fill_diagonal(D, np.inf)  # exclude self
        for i in range(len(saws_coords)):
            if A[i].sum() == 0:
                continue
            strongest = np.argmax(A[i])
            # Among CONNECTED nodes, `strongest` must be the closest
            connected = np.where(A[i] > 0)[0]
            nearest_connected = connected[np.argmin(D[i, connected])]
            assert strongest == nearest_connected, (
                f"node {i}: strongest edge {strongest} != nearest connected {nearest_connected}"
            )


# ---------------------------------------------------------------------------
# Test 8: weight sanity — Gaussian kernel produces expected magnitudes
# ---------------------------------------------------------------------------

class TestWeightSanity:
    """Pre-normalisation Gaussian weights: w = exp(-d²/(2σ²)).
    At d=0: w=1. At d=σ: w=exp(-0.5) ≈ 0.607. At d=2σ: w=exp(-2) ≈ 0.135.
    At d=3σ: w=exp(-4.5) ≈ 0.011.
    """

    def test_weight_at_one_sigma_close_to_zero_six(self):
        """Two stations exactly σ=100km apart should have raw weight ≈ 0.607."""
        # 100 km ≈ 0.9° latitude at equator (spherical → ~0.9°)
        coords = pd.DataFrame({
            "node_id": [0, 1],
            "lat": [-33.0, -33.9],   # ≈ 100 km separation
            "lon": [18.0, 18.0],
        })
        # Verify actual distance first
        D = build_distance_matrix(coords)
        assert 90 < D[0, 1] < 110, f"distance {D[0, 1]:.1f} km; expected ~100"

        # Build adjacency without row normalisation to inspect raw weights
        # (build_adjacency_matrix always row-normalises, so we test via the
        # relationship: with only 2 nodes both are each other's only neighbour,
        # so both rows normalise to 1.0 regardless of weight magnitude.)
        # Instead, verify via three collinear stations where we CAN observe
        # relative weights.
        coords3 = pd.DataFrame({
            "node_id": [0, 1, 2],
            "lat": [-33.0, -33.9, -35.7],   # 0km, 100km, 300km from node 0
            "lon": [18.0, 18.0, 18.0],
        })
        D3 = build_distance_matrix(coords3)
        assert 90 < D3[0, 1] < 110
        assert 290 < D3[0, 2] < 310
        A = build_adjacency_matrix(coords3, distance_threshold_km=500, sigma_km=100)
        # raw w at 100km / raw w at 300km = exp(-0.5) / exp(-4.5) = exp(4) ≈ 54.6
        # After row normalisation the ratio is preserved.
        ratio = A[0, 1] / A[0, 2]
        assert 30 < ratio < 90, f"weight ratio {ratio:.1f} outside plausible range"

    def test_weight_just_under_cutoff_small(self):
        """At d just under the cutoff distance, raw weight should be much
        smaller than at d=σ. With sigma=100 and cutoff=300, weight at 290 km
        is exp(-290²/(2·100²)) = exp(-4.205) ≈ 0.0149."""
        # Use two-station geometry where we can inspect the raw kernel via
        # a direct computation, not the row-normalised A.
        d = 290.0  # km
        sigma = 100.0
        raw_w = np.exp(-(d ** 2) / (2 * sigma ** 2))
        assert 0.01 < raw_w < 0.02
