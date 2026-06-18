"""
Derives physically meaningful meteorological variables from raw ERA5 fields.

Appends two new variables to a (T, N, d) ERA5 feature array:
  ws10 — 10m wind speed [m/s]:    sqrt(u10² + v10²)
  rh   — relative humidity [%]:   100 × eₛ(d2m) / eₛ(t2m)  (Magnus formula)

Wind direction is intentionally not derived as a separate variable: u10 and v10
already encode direction in Cartesian form, and raw degrees (0–360°) are periodic
and degrade z-score normalisation.
"""

from typing import List, Tuple

import numpy as np


def _magnus_es(T_celsius: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure [hPa] via the Magnus formula."""
    return 6.112 * np.exp((17.67 * T_celsius) / (T_celsius + 243.5))


def derive_era5_variables(
    X: np.ndarray,
    var_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Append ws10 and rh to an ERA5 feature array.

    Args:
        X:         (T, N, d) float32 array in canonical ERA5 variable order.
        var_names: list of d variable name strings.

    Returns:
        X_extended:    (T, N, d+2) float32 array.
        new_var_names: var_names + ["ws10", "rh"].

    Raises:
        KeyError: if any required variable (u10, v10, t2m, d2m) is absent.
    """
    required = ("u10", "v10", "t2m", "d2m")
    missing = [v for v in required if v not in var_names]
    if missing:
        raise KeyError(
            f"derive_era5_variables: required variable(s) not found: {missing}. "
            f"Available: {var_names}"
        )

    u_idx = var_names.index("u10")
    v_idx = var_names.index("v10")
    t_idx = var_names.index("t2m")
    d_idx = var_names.index("d2m")

    # 10m wind speed
    ws10 = np.sqrt(X[..., u_idx] ** 2 + X[..., v_idx] ** 2)   # (T, N)

    # Relative humidity — ERA5 temperatures are in Kelvin
    T_c  = X[..., t_idx] - 273.15   # K → °C
    Td_c = X[..., d_idx] - 273.15   # K → °C
    rh = np.clip(100.0 * _magnus_es(Td_c) / _magnus_es(T_c), 0.0, 100.0)  # (T, N)

    X_ext = np.concatenate(
        [X, ws10[..., np.newaxis], rh[..., np.newaxis]], axis=-1
    ).astype(np.float32)

    return X_ext, var_names + ["ws10", "rh"]
