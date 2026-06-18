"""
Unit tests for preprocessing/derive.py.

All tests use synthetic arrays — no ERA5 files required.
"""

import numpy as np
import pytest

from preprocessing.derive import derive_era5_variables

# Canonical ERA5 variable order (same as era5_parser.VARIABLE_NAMES)
ERA5_VARS = ["t2m", "d2m", "tp", "u10", "v10", "sp"]

T, N, D = 24, 10, len(ERA5_VARS)


def _fake_X(seed: int = 0) -> np.ndarray:
    """Physically plausible ERA5 array (T, N, d)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((T, N, D), dtype=np.float32)
    t_idx  = ERA5_VARS.index("t2m")
    d_idx  = ERA5_VARS.index("d2m")
    u_idx  = ERA5_VARS.index("u10")
    v_idx  = ERA5_VARS.index("v10")
    sp_idx = ERA5_VARS.index("sp")
    tp_idx = ERA5_VARS.index("tp")
    X[..., t_idx]  = rng.uniform(275.0, 310.0, (T, N)).astype(np.float32)  # K
    X[..., d_idx]  = X[..., t_idx] - rng.uniform(0.0, 20.0, (T, N)).astype(np.float32)
    X[..., u_idx]  = rng.uniform(-15.0, 15.0, (T, N)).astype(np.float32)
    X[..., v_idx]  = rng.uniform(-15.0, 15.0, (T, N)).astype(np.float32)
    X[..., sp_idx] = rng.uniform(95000.0, 102000.0, (T, N)).astype(np.float32)
    X[..., tp_idx] = rng.uniform(0.0, 0.01, (T, N)).astype(np.float32)
    return X


# ---------------------------------------------------------------------------
# Shape and naming
# ---------------------------------------------------------------------------

def test_shape():
    X_ext, _ = derive_era5_variables(_fake_X(), ERA5_VARS)
    assert X_ext.shape == (T, N, D + 2)


def test_new_varnames():
    _, names = derive_era5_variables(_fake_X(), ERA5_VARS)
    assert names[-2] == "ws10"
    assert names[-1] == "rh"
    assert len(names) == D + 2


# ---------------------------------------------------------------------------
# Physical constraints
# ---------------------------------------------------------------------------

def test_ws10_nonneg():
    X_ext, names = derive_era5_variables(_fake_X(), ERA5_VARS)
    ws = X_ext[..., names.index("ws10")]
    assert (ws >= 0).all(), "Wind speed must be non-negative everywhere"


def test_rh_range():
    X_ext, names = derive_era5_variables(_fake_X(), ERA5_VARS)
    rh = X_ext[..., names.index("rh")]
    assert (rh >= 0).all() and (rh <= 100).all(), "RH must be in [0, 100]"


# ---------------------------------------------------------------------------
# Spot-check formulas
# ---------------------------------------------------------------------------

def test_ws10_formula():
    """u=3, v=4 → ws=5 (3-4-5 Pythagorean triple)."""
    X = np.zeros((1, 1, D), dtype=np.float32)
    X[0, 0, ERA5_VARS.index("u10")] = 3.0
    X[0, 0, ERA5_VARS.index("v10")] = 4.0
    X[0, 0, ERA5_VARS.index("t2m")] = 300.0
    X[0, 0, ERA5_VARS.index("d2m")] = 290.0
    X_ext, names = derive_era5_variables(X, ERA5_VARS)
    ws = float(X_ext[0, 0, names.index("ws10")])
    assert abs(ws - 5.0) < 1e-5, f"Expected ws=5.0, got {ws}"


def test_rh_saturated():
    """t2m == d2m → RH should be 100 %."""
    X = np.zeros((1, 1, D), dtype=np.float32)
    T_k = 295.0
    X[0, 0, ERA5_VARS.index("t2m")] = T_k
    X[0, 0, ERA5_VARS.index("d2m")] = T_k   # dewpoint = temperature → saturated
    X_ext, names = derive_era5_variables(X, ERA5_VARS)
    rh = float(X_ext[0, 0, names.index("rh")])
    assert abs(rh - 100.0) < 1e-3, f"Expected RH≈100, got {rh}"


def test_rh_dry():
    """d2m much lower than t2m → RH should be near 0."""
    X = np.zeros((1, 1, D), dtype=np.float32)
    X[0, 0, ERA5_VARS.index("t2m")] = 310.0   # hot day (37 °C)
    X[0, 0, ERA5_VARS.index("d2m")] = 230.0   # very cold dewpoint (−43 °C)
    X_ext, names = derive_era5_variables(X, ERA5_VARS)
    rh = float(X_ext[0, 0, names.index("rh")])
    assert rh < 1.0, f"Expected RH near 0, got {rh}"


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

def test_dtype_float32():
    X_ext, _ = derive_era5_variables(_fake_X(), ERA5_VARS)
    assert X_ext.dtype == np.float32, f"Expected float32, got {X_ext.dtype}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_missing_variable_raises():
    bad_vars = ["t2m", "tp", "sp"]   # missing u10, v10, d2m
    X = np.zeros((T, N, 3), dtype=np.float32)
    with pytest.raises(KeyError, match="u10|v10|d2m"):
        derive_era5_variables(X, bad_vars)
