"""
Regression tests for evaluation.vpt_extended and evaluation.rollout.

Mirrors the 11 required tests from specs/VPT_Extended_Spec.md §7, adapted
for SAWS variable set [t2m, tp, rh, wd, ws, sp] (spec's ERA5 hygiene rules
for ws10 from u10/v10 and rh from Magnus-Tetens don't apply to SAWS where
those variables are directly measured).
"""

import numpy as np
import pytest

from evaluation.vpt_extended import (
    per_sample_nrmse,
    vfh_batch,
    vfh_grid,
    survival_curve,
    summarise_vfh,
    eps_grid_from_pooled_curves,
    aggregate_eps_grid_across_seeds,
    aggregate_vfh_across_seeds,
    pooled_curve_matches,
    per_horizon_report_from_curves,
    DEFAULT_EPS_GRID,
    DEFAULT_REPORT_HORIZONS,
)
from evaluation.rollout import (
    apply_hygiene,
    rollout_predict,
    rollout_nrmse_curve,
    detect_numerical_failure,
    detect_collapse,
    early_stop_start,
    persistence_rollout,
    PHYSICAL_BOUNDS,
)


# ---------------------------------------------------------------------------
# Test 1: per-sample VFH on hand-built curves → exact crossings
# ---------------------------------------------------------------------------

def test_vfh_per_sample_known_values():
    """
    Curve rises monotonically. First crossing of ε=0.3 should be at the
    hand-computed step for each sample.
    """
    # (S=3, H=6, d=1)
    curves = np.array([
        [[0.1], [0.2], [0.25], [0.29], [0.31], [0.5]],   # crosses at h=4
        [[0.4], [0.5], [0.6], [0.7], [0.8], [0.9]],      # already above at h=0 → 0
        [[0.05], [0.10], [0.15], [0.20], [0.25], [0.29]],# never crosses → 6
    ], dtype=np.float32)
    vfh = vfh_batch(curves, 0.3)
    np.testing.assert_array_equal(vfh, np.array([[4], [0], [6]]))


def test_vfh_grid_all_thresholds():
    curves = np.array([
        [[0.1], [0.2], [0.3], [0.4], [0.5]],   # crosses ε=0.3 at h=2
    ], dtype=np.float32)
    grid = vfh_grid(curves, eps_grid=[0.15, 0.3, 1.0])
    # ε=0.15: crosses at h=1; ε=0.3: at h=2; ε=1.0: never → H=5
    np.testing.assert_array_equal(grid[0, 0, :], np.array([1, 2, 5]))


# ---------------------------------------------------------------------------
# Test 2: floor + ceiling values, and censoring fractions
# ---------------------------------------------------------------------------

def test_vfh_floor_and_ceiling_summary():
    """A mix of floor (crosses at h=0), interior, and ceiling (never) samples
    must produce the correct pct_floor / pct_ceiling fractions."""
    H = 12
    # 10 samples, single variable: 3 floor, 4 interior (crossing at h=5),
    # 3 ceiling (never crossing).
    curves = np.zeros((10, H, 1), dtype=np.float32)
    curves[:3, :, 0] = 1.0            # always above ε=0.3 → floor
    curves[3:7, :, 0] = np.linspace(0.1, 0.9, H)   # crosses somewhere near h=3
    # curves[7:10] remain all-zero → never cross → ceiling
    vfh = vfh_grid(curves, eps_grid=[0.3])
    summary = summarise_vfh(vfh, H_cap=H, var_names=["x"], eps_grid=[0.3])
    stats = summary["x"][0.3]
    assert stats["pct_floor"] == pytest.approx(0.3, abs=1e-6)
    assert stats["pct_ceiling"] == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 3: survival curve is non-increasing in h
# ---------------------------------------------------------------------------

def test_survival_curve_monotone():
    rng = np.random.default_rng(seed=0)
    S, H, d = 100, 12, 2
    curves = np.cumsum(np.abs(rng.normal(0, 0.1, size=(S, H, d))), axis=1)
    surv = survival_curve(curves, 0.5)
    # Non-increasing along horizon axis
    diffs = np.diff(surv, axis=0)
    assert (diffs <= 0).all(), (
        f"survival curve must be monotone-nonincreasing; got max increase "
        f"{diffs.max():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: pooled consistency check (spec §4.5)
# ---------------------------------------------------------------------------

def test_pooled_consistency_reconstruction():
    """Root-mean-square across samples of per-sample curves should reproduce
    the pooled curve from regression.py within tight rtol."""
    rng = np.random.default_rng(seed=1)
    S, H, N, d = 50, 12, 25, 6
    preds = rng.normal(0, 1, size=(S, H, N, d)).astype(np.float32)
    targets = rng.normal(0, 1, size=(S, H, N, d)).astype(np.float32)

    # Pooled formula (as in evaluation.regression.nrmse_per_horizon):
    pooled = np.sqrt(((preds - targets) ** 2).mean(axis=(0, 2)))   # (H, d)

    per_sample = per_sample_nrmse(preds, targets)                  # (S, H, d)
    ok, max_rel = pooled_curve_matches(per_sample, pooled, rtol=1e-4)
    assert ok, f"reconstruction rel_err = {max_rel:.2e}"


# ---------------------------------------------------------------------------
# Test 5: rollout block k=0 equals a direct forecast (regression guard)
# ---------------------------------------------------------------------------

def test_rollout_first_block_equals_direct():
    """The spec's §5.4 regression guard: block k=0 of a rollout must equal
    the model's ordinary direct forecast for the same t0 (before hygiene).
    """
    rng = np.random.default_rng(seed=2)
    H, N, d = 4, 3, 2
    initial = rng.normal(0, 1, size=(H, N, d)).astype(np.float32)
    mean = np.zeros((1, N, d), dtype=np.float32)
    std = np.ones((1, N, d), dtype=np.float32)

    # Deterministic stub: outputs an affine transform of the input mean.
    calls = []
    def predict_fn(window):
        calls.append(window.copy())
        return window + 0.1

    rollout = rollout_predict(
        predict_fn, initial, K=3, horizon=H, mean=mean, std=std,
        var_names=["a", "b"], apply_hygiene_between_steps=False,
    )
    # block 0 must equal predict_fn(initial)
    direct = initial + 0.1
    np.testing.assert_array_equal(rollout[:H], direct)


# ---------------------------------------------------------------------------
# Test 6: after step k, window is (hygiene-processed) pred of step k
# ---------------------------------------------------------------------------

def test_rollout_window_replacement():
    """Every rollout iteration must pass the previous prediction (hygiened)
    as the next window, not the original truth."""
    H, N, d = 4, 3, 2
    initial = np.zeros((H, N, d), dtype=np.float32)
    mean = np.zeros((1, N, d), dtype=np.float32)
    std = np.ones((1, N, d), dtype=np.float32)

    windows_seen = []
    def predict_fn(window):
        windows_seen.append(window.copy())
        return window + 1.0

    rollout_predict(
        predict_fn, initial, K=3, horizon=H, mean=mean, std=std,
        var_names=["a", "b"], apply_hygiene_between_steps=False,
    )
    # k=0: window = initial (zeros)
    np.testing.assert_array_equal(windows_seen[0], np.zeros((H, N, d)))
    # k=1: window = initial + 1.0
    np.testing.assert_array_equal(windows_seen[1], np.ones((H, N, d)))
    # k=2: window = (initial + 1.0) + 1.0
    np.testing.assert_array_equal(windows_seen[2], 2 * np.ones((H, N, d)))


# ---------------------------------------------------------------------------
# Test 7: SAWS hygiene rules (adapted from spec's ERA5 tests)
# ---------------------------------------------------------------------------

class TestSAWSHygiene:
    """
    Spec §7 tests 7 (ws10 re-derivation) and 8 (rh from Magnus-Tetens)
    do NOT apply to SAWS — those variables are directly measured, not
    derived from other channels. Instead we verify that the SAWS-relevant
    physical bounds (tp ≥ 0, rh ∈ [0, 100]) are enforced.
    """

    def test_tp_clamp_nonnegative(self):
        var_names = ["t2m", "tp", "rh", "wd", "ws", "sp"]
        N, d = 3, 6
        mean = np.zeros((1, N, d), dtype=np.float32)
        std = np.ones((1, N, d), dtype=np.float32)
        pred_z = np.zeros((2, N, d), dtype=np.float32)
        # Force tp to be negative in physical space at some cell
        pred_z[0, 0, 1] = -0.5   # after denorm with mean=0/std=1: physical = -0.5
        cleaned = apply_hygiene(pred_z, mean, std, var_names)
        # Denormalise cleaned to check physical value
        physical = cleaned * std[0] + mean[0]
        assert physical[0, 0, 1] >= 0, (
            f"tp should be clamped to ≥ 0, got {physical[0, 0, 1]:.4f}"
        )
        # Non-tp channels untouched at cells we didn't set
        assert physical[0, 0, 0] == 0.0
        assert physical[0, 0, 5] == 0.0

    def test_rh_clamp_to_zero_hundred(self):
        var_names = ["t2m", "tp", "rh", "wd", "ws", "sp"]
        N, d = 2, 6
        mean = np.zeros((1, N, d), dtype=np.float32)
        std = np.ones((1, N, d), dtype=np.float32)
        pred_z = np.zeros((1, N, d), dtype=np.float32)
        pred_z[0, 0, 2] = 150.0   # rh far above 100
        pred_z[0, 1, 2] = -10.0   # rh far below 0
        cleaned = apply_hygiene(pred_z, mean, std, var_names)
        physical = cleaned * std[0] + mean[0]
        assert physical[0, 0, 2] == 100.0
        assert physical[0, 1, 2] == 0.0

    def test_hygiene_no_effect_when_all_within_bounds(self):
        var_names = ["t2m", "tp", "rh", "wd", "ws", "sp"]
        N, d = 2, 6
        mean = np.zeros((1, N, d), dtype=np.float32)
        std = np.ones((1, N, d), dtype=np.float32)
        pred_z = np.array([[[15.0, 0.5, 60.0, 180.0, 3.0, 1013.0]] * N],
                          dtype=np.float32)
        cleaned = apply_hygiene(pred_z, mean, std, var_names)
        np.testing.assert_allclose(cleaned, pred_z)


# ---------------------------------------------------------------------------
# Test 8: NaN in a rollout step terminates that start, not the job
# ---------------------------------------------------------------------------

def test_nan_detection_flag():
    good = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    nan_pred = good.copy(); nan_pred[0, 0] = np.nan
    inf_pred = good.copy(); inf_pred[1, 1] = np.inf
    assert detect_numerical_failure(nan_pred)
    assert detect_numerical_failure(inf_pred)
    assert not detect_numerical_failure(good)


# ---------------------------------------------------------------------------
# Test 9: ε grid output schema for Deliverable A
# ---------------------------------------------------------------------------

def test_eps_grid_shapes_and_types():
    pooled = {
        "t2m": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        "rh":  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    }
    grid = eps_grid_from_pooled_curves(pooled, eps_grid=[0.3, 0.6, 1.0])
    assert set(grid.keys()) == {"t2m", "rh"}
    # For 't2m' crosses at h=2 (0.3), h=5 (0.6), h=9 (1.0)
    assert grid["t2m"][0.3] == 2
    assert grid["t2m"][0.6] == 5
    assert grid["t2m"][1.0] == 9
    # For 'rh' constant 0.5: crosses at h=0 for ε=0.3, never for ε≥0.6
    assert grid["rh"][0.3] == 0
    assert grid["rh"][0.6] == 12
    assert grid["rh"][1.0] == 12


def test_eps_grid_aggregation_across_seeds():
    per_seed = [
        {"t2m": {0.3: 5, 1.0: 12}},
        {"t2m": {0.3: 6, 1.0: 12}},
        {"t2m": {0.3: 7, 1.0: 11}},
    ]
    out = aggregate_eps_grid_across_seeds(per_seed, eps_grid=[0.3, 1.0])
    assert out["t2m"][0.3]["mean"] == pytest.approx(6.0)
    assert out["t2m"][0.3]["min"] == 5
    assert out["t2m"][0.3]["max"] == 7
    assert out["t2m"][1.0]["median"] == 12
    assert out["t2m"][1.0]["per_seed"] == [12, 12, 11]


# ---------------------------------------------------------------------------
# Test 10: rollout NRMSE curve reproduces per-step math
# ---------------------------------------------------------------------------

def test_rollout_nrmse_curve_shape_and_values():
    preds = np.zeros((6, 4, 2), dtype=np.float32)
    truth = np.zeros((6, 4, 2), dtype=np.float32)
    truth[..., 0] = 1.0    # constant 1.0 for variable 0
    truth[..., 1] = 2.0    # constant 2.0 for variable 1
    curve = rollout_nrmse_curve(preds, truth)
    # For var 0: sqrt(mean((0-1)^2)) = 1 at every step
    # For var 1: sqrt(mean((0-2)^2)) = 2 at every step
    np.testing.assert_allclose(curve[:, 0], 1.0)
    np.testing.assert_allclose(curve[:, 1], 2.0)
    assert curve.shape == (6, 2)


# ---------------------------------------------------------------------------
# Test 11: streaming path (no full-pred accumulation)
# ---------------------------------------------------------------------------

def test_per_sample_nrmse_is_streaming_friendly():
    """The per-sample NRMSE reducer operates on ONE batch at a time — no
    dependency on later batches — so streaming callers can process arbitrary
    batch sizes and concatenate the (S, H, d) reductions without ever
    materialising the full (S, H, N, d) preds/targets. Verify by processing
    two half-batches and confirming concatenation equals the full batch.
    """
    rng = np.random.default_rng(seed=3)
    S, H, N, d = 40, 5, 6, 3
    preds = rng.normal(size=(S, H, N, d)).astype(np.float32)
    targets = rng.normal(size=(S, H, N, d)).astype(np.float32)
    full = per_sample_nrmse(preds, targets)
    half1 = per_sample_nrmse(preds[:20], targets[:20])
    half2 = per_sample_nrmse(preds[20:], targets[20:])
    np.testing.assert_array_equal(full, np.concatenate([half1, half2], axis=0))


# ---------------------------------------------------------------------------
# Bonus: collapse detection + early-stop
# ---------------------------------------------------------------------------

def test_collapse_detection():
    curve = np.array([
        [1.5, 0.5],   # var 0 outside band, var 1 inside
        [1.5, 0.6],
        [1.5, 0.7],
        [1.5, 0.8],   # var 1 stays in band for enough steps
    ] * 4, dtype=np.float32)   # (16, 2)
    collapsed = detect_collapse(curve, band=(0.5, 0.8), min_consecutive=12)
    assert not collapsed[0]   # var 0 never in band
    assert collapsed[1]       # var 1 in band for 16 consecutive steps


def test_early_stop_all_variables_diverged_stops():
    """Every variable above threshold at the last step → stop."""
    huge = np.full((10, 3), 10.0, dtype=np.float32)
    assert early_stop_start(huge)


def test_early_stop_no_variable_diverged_continues():
    """All variables well under threshold → continue."""
    small = np.full((10, 3), 0.5, dtype=np.float32)
    assert not early_stop_start(small)


def test_early_stop_one_variable_diverged_continues():
    """Regression guard for the min-vs-max bug: if one fast variable
    (rain, wind) blows past 5.0 while slow variables (pressure, temp) are
    still tracking under ε, the rollout must CONTINUE so slow-variable
    crossings are correctly measured. Prior implementation used max() and
    stopped here, systematically right-censoring slow variables."""
    curve = np.array([[0.3, 0.5, 12.0, 0.4]] * 5, dtype=np.float32)   # var 2 huge, others fine
    assert not early_stop_start(curve), (
        "one-diverged-others-fine must NOT stop the rollout — "
        "would corrupt slow-variable RVPT numbers"
    )


def test_early_stop_all_but_one_diverged_continues():
    """Even 3 of 4 variables diverged is not enough — must be all."""
    curve = np.array([[10.0, 10.0, 0.3, 10.0]] * 5, dtype=np.float32)
    assert not early_stop_start(curve)


def test_persistence_rollout_shape():
    seed = np.arange(24, dtype=np.float32).reshape(4, 3, 2)   # T=4, N=3, d=2
    out = persistence_rollout(seed, K=3, horizon=4)
    assert out.shape == (12, 3, 2)
    # Every step should equal seed[-1]
    for h in range(12):
        np.testing.assert_array_equal(out[h], seed[-1])


# ---------------------------------------------------------------------------
# Fixes from external review: wd wrap, ws clamp, n_floor/n_ceiling, aggregator
# ---------------------------------------------------------------------------

class TestSAWSHygieneExtended:
    """Coverage of the review items:
      - SAWS 'ws' name (not just ERA5 'ws10') is clamped ≥ 0
      - Wind direction 'wd' is wrapped mod 360° (circular, not clampable)
    """

    def test_saws_ws_clamped_nonnegative(self):
        """SAWS uses 'ws' rather than ERA5's 'ws10'. Both must be in
        PHYSICAL_BOUNDS or SAWS wind hygiene silently no-ops."""
        var_names = ["t2m", "tp", "rh", "wd", "ws", "sp"]
        N, d = 2, 6
        mean = np.zeros((1, N, d), dtype=np.float32)
        std = np.ones((1, N, d), dtype=np.float32)
        pred_z = np.zeros((1, N, d), dtype=np.float32)
        # Force ws to be negative in physical space
        pred_z[0, 0, 4] = -2.5   # wind speed -2.5 m/s — impossible
        cleaned = apply_hygiene(pred_z, mean, std, var_names)
        physical = cleaned * std[0] + mean[0]
        assert physical[0, 0, 4] >= 0, (
            f"SAWS 'ws' must be clamped ≥ 0, got {physical[0, 0, 4]:.4f}"
        )

    def test_wd_wraps_to_zero_360_range(self):
        """Circular wrap: wd = 400° → 40°; wd = -30° → 330°."""
        var_names = ["t2m", "tp", "rh", "wd", "ws", "sp"]
        N, d = 3, 6
        mean = np.zeros((1, N, d), dtype=np.float32)
        std = np.ones((1, N, d), dtype=np.float32)
        pred_z = np.zeros((1, N, d), dtype=np.float32)
        pred_z[0, 0, 3] = 400.0        # → 40
        pred_z[0, 1, 3] = -30.0        # → 330
        pred_z[0, 2, 3] = 720.0 + 90   # → 90 (double-wrap)
        cleaned = apply_hygiene(pred_z, mean, std, var_names)
        physical = cleaned * std[0] + mean[0]
        assert physical[0, 0, 3] == pytest.approx(40.0)
        assert physical[0, 1, 3] == pytest.approx(330.0)
        assert physical[0, 2, 3] == pytest.approx(90.0)

    def test_wd_in_range_unchanged(self):
        """A wd already in [0, 360) must round-trip identically."""
        var_names = ["t2m", "tp", "rh", "wd", "ws", "sp"]
        N, d = 1, 6
        mean = np.zeros((1, N, d), dtype=np.float32)
        std = np.ones((1, N, d), dtype=np.float32)
        pred_z = np.zeros((1, N, d), dtype=np.float32)
        pred_z[0, 0, 3] = 137.5
        cleaned = apply_hygiene(pred_z, mean, std, var_names)
        physical = cleaned * std[0] + mean[0]
        assert physical[0, 0, 3] == pytest.approx(137.5)


class TestEpsGridAggregationCensoringCounts:
    """Deliverable A output must carry n_floor and n_ceiling (spec §3)."""

    def test_n_floor_and_n_ceiling_present(self):
        per_seed = [
            {"t2m": {0.3: 0}},    # floor
            {"t2m": {0.3: 5}},    # interior
            {"t2m": {0.3: 12}},   # ceiling
            {"t2m": {0.3: 0}},    # floor
        ]
        out = aggregate_eps_grid_across_seeds(per_seed, eps_grid=[0.3], H_cap=12)
        stats = out["t2m"][0.3]
        assert stats["n_floor"] == 2
        assert stats["n_ceiling"] == 1


class TestCrossSeedVFHAggregator:
    """Deliverable B cross-seed aggregation (spec §4.3)."""

    def _fake_seed_summary(self, median: float, pct_floor: float, pct_ceiling: float,
                           survival: list[float]):
        return {"t2m": {0.3: {
            "median": median, "iqr_low": median - 1, "iqr_high": median + 1,
            "min": 0, "max": 12,
            "pct_floor": pct_floor, "pct_ceiling": pct_ceiling,
            "survival_curve": survival,
        }}}

    def test_median_of_seed_medians(self):
        per_seed = [
            self._fake_seed_summary(4.0, 0.10, 0.05, [1.0, 0.8, 0.6, 0.4]),
            self._fake_seed_summary(6.0, 0.15, 0.08, [1.0, 0.85, 0.65, 0.45]),
            self._fake_seed_summary(5.0, 0.12, 0.07, [1.0, 0.82, 0.62, 0.42]),
        ]
        out = aggregate_vfh_across_seeds(per_seed, H_cap=12,
                                          var_names=["t2m"], eps_grid=[0.3])
        stats = out["t2m"][0.3]
        assert stats["median_of_seed_medians"] == pytest.approx(5.0)
        # IQR of [4, 5, 6] = (4.5, 5.5)
        assert stats["iqr_low_of_medians"] == pytest.approx(4.5)
        assert stats["iqr_high_of_medians"] == pytest.approx(5.5)
        # Censoring fractions averaged
        assert stats["mean_pct_floor"] == pytest.approx((0.10 + 0.15 + 0.12) / 3)
        assert stats["mean_pct_ceiling"] == pytest.approx((0.05 + 0.08 + 0.07) / 3)
        # Mean survival curve
        expected_surv = np.mean([
            [1.0, 0.8, 0.6, 0.4],
            [1.0, 0.85, 0.65, 0.45],
            [1.0, 0.82, 0.62, 0.42],
        ], axis=0).tolist()
        np.testing.assert_allclose(stats["mean_survival_curve"], expected_surv)
        # Per-seed detail preserved
        assert len(stats["per_seed"]) == 3

    def test_empty_input_returns_empty(self):
        assert aggregate_vfh_across_seeds([], H_cap=12, var_names=["t2m"], eps_grid=[0.3]) == {}


class TestHorizonExtensionH48:
    """Coverage of the 2026-08-04 horizon extension to 48 with 7-horizon
    truncation reporting (spec-following of Davidson/Gaibie, extended)."""

    def test_default_report_horizons_matches_spec(self):
        """These are the reported subhorizons the truncation helper defaults
        to. Locked so a future refactor doesn't silently drop 36h or 48h."""
        assert DEFAULT_REPORT_HORIZONS == (3, 6, 9, 12, 24, 36, 48)

    def test_per_horizon_report_shape_and_values(self):
        """Truncation of a monotonically-rising per-sample curve should give
        exactly the values at the indexed step."""
        S, H, d = 5, 48, 2
        # Curve[:,h,v] = h * (v+1) / 10 — deterministic values per step and variable
        curves = np.zeros((S, H, d), dtype=np.float32)
        for h in range(H):
            for v in range(d):
                curves[:, h, v] = h * (v + 1) / 10.0
        report = per_horizon_report_from_curves(curves, var_names=["a", "b"])
        # h=3 (index 2): curve value is 2*1/10 = 0.2 for 'a', 2*2/10 = 0.4 for 'b'
        assert report[3]["a"]["mean_nrmse"] == pytest.approx(0.2)
        assert report[3]["b"]["mean_nrmse"] == pytest.approx(0.4)
        # h=48 (index 47): 47*1/10 = 4.7 for 'a', 47*2/10 = 9.4 for 'b'
        assert report[48]["a"]["mean_nrmse"] == pytest.approx(4.7)
        assert report[48]["b"]["mean_nrmse"] == pytest.approx(9.4)
        # All 7 default horizons present
        assert set(report.keys()) == {3, 6, 9, 12, 24, 36, 48}

    def test_per_horizon_report_skips_out_of_range(self):
        """If curves only go to H=12 but we ask for h=48, we skip 48 silently
        (with a log)."""
        curves = np.ones((3, 12, 2), dtype=np.float32)
        report = per_horizon_report_from_curves(
            curves, var_names=["a", "b"],
            report_horizons=[3, 12, 24, 48],
        )
        # 3 and 12 are within H=12; 24 and 48 are not
        assert set(report.keys()) == {3, 12}

    def test_h_cap_dynamic_inference_handles_horizon_48(self):
        """With H_cap=None, aggregate_eps_grid_across_seeds must infer from
        the data, correctly counting a VFH=48 sample as n_ceiling."""
        per_seed = [
            {"t2m": {0.3: 48}},   # ceiling at h=48
            {"t2m": {0.3: 47}},   # interior (one step below the max observed)
            {"t2m": {0.3: 48}},   # ceiling
        ]
        # No explicit H_cap → inferred from data as max=48
        out = aggregate_eps_grid_across_seeds(per_seed, eps_grid=[0.3])
        stats = out["t2m"][0.3]
        assert stats["n_ceiling"] == 2   # 2 seeds at VFH=48
        assert stats["n_floor"] == 0     # no seed at VFH=0
        assert stats["max"] == 48        # correctly captured
