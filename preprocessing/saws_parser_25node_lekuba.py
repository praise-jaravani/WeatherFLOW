"""
SAWS station-data parser for the 25-station 2013-2020 Lekuba-only network.

Reads the two ex-student Lekuba workbooks (long-form Excel), applies the
Davidson (2022) < 10% missingness threshold on a canonical hourly grid, and
returns the pipeline-standard (X, timestamps, coords, var_names) contract.

Two source formats, both long-form:
  * data1.xlsx  — 8 columns  (StasName, T, R, H, WD, WS, P, DateT), Sheet1..21
  * data2.xls   — 12 columns (CompNo, StasName, Lat, Lon, DateT, hour, T, R, H, WD, WS, P)
                  Sheet1 has header; Sheets 2-14 do NOT (schema applied by position).

Station coordinates come from a curated lookup at data/saws/station_coordinates.csv
(pre-populated from data2 in-file, Gen's block-format extract, and the 74-row
stations.csv sample; 10 Western/Northern Cape stations require manual lookup
because they are not present in any machine-readable source we ship).

Variable naming: SAWS raw names are mapped to Davidson/ERA5-lineage lowercase
snake names so downstream code (labels.py's "tp" wet-hour rule, evaluation
scripts) continues to work unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from openpyxl import load_workbook

log = logging.getLogger(__name__)

# Canonical variable order and Davidson/ERA5-lineage naming
CANONICAL_VARS = ["t2m", "tp", "rh", "wd", "ws", "sp"]

# Raw-name → canonical-name mapping (handles the WindDirection vs WindDir
# difference between data1 and data2 without callers having to know)
RENAME_RAW_TO_CANONICAL = {
    "Temperature":   "t2m",
    "Rain":          "tp",
    "Humidity":      "rh",
    "WindDirection": "wd",   # data1
    "WindDir":       "wd",   # data2
    "WindSpeed":     "ws",
    "Pressure":      "sp",
}

# Locked network parameters (see data/saws/PLAN.md)
WINDOW_START = pd.Timestamp("2013-01-01 00:00")
WINDOW_END   = pd.Timestamp("2020-12-31 23:00")
MISSINGNESS_THRESHOLD = 0.10   # Davidson (2022) < 10%

# Stations we already know cannot pass the threshold — dropped at read time
# so they never contribute to the audit noise.
KNOWN_FAIL = {
    "PORT ELIZABETH - WK",   # data1: 1 month of data in 2013 only
    "LADYSMITH",             # data2: WindDir 13.78% inside window
    "WARDEN - HERITAGE",     # data2: extract starts mid-2014, 16.98% overall
}


# ---------------------------------------------------------------------------
# Sub-parsers
# ---------------------------------------------------------------------------

def _read_lekuba_v1(path: Path) -> pd.DataFrame:
    """data1.xlsx — 21 sheets, each with header row, 8 columns."""
    wb = load_workbook(str(path), read_only=True, data_only=True)
    frames = []
    cols = ["StasName", "Temperature", "Rain", "Humidity",
            "WindDirection", "WindSpeed", "Pressure", "DateT"]
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                continue   # every data1 sheet has a header
            rows.append(row[:8])
        if rows:
            frames.append(pd.DataFrame(rows, columns=cols))
    wb.close()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["StasName", "DateT"])
    df["DateT"] = pd.to_datetime(df["DateT"], errors="coerce")
    df = df.dropna(subset=["DateT"])
    df["StasName"] = df["StasName"].astype(str).str.strip()
    return df.rename(columns=RENAME_RAW_TO_CANONICAL)


def _read_lekuba_v2(path: Path) -> pd.DataFrame:
    """data2.xls — 14 sheets, only Sheet1 has a header, 12 columns.

    Apply Sheet1's schema to Sheets 2-14 by position.
    """
    xls = pd.ExcelFile(str(path), engine="xlrd")
    sheet1 = pd.read_excel(xls, sheet_name="Sheet1", header=0)
    schema = list(sheet1.columns)
    frames = [sheet1]
    for sn in xls.sheet_names[1:]:
        frames.append(pd.read_excel(xls, sheet_name=sn, header=None, names=schema))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["StasName", "DateT"])
    df["DateT"] = pd.to_datetime(df["DateT"], errors="coerce")
    df = df.dropna(subset=["DateT"])
    df["StasName"] = df["StasName"].astype(str).str.strip()
    return df.rename(columns=RENAME_RAW_TO_CANONICAL)


# ---------------------------------------------------------------------------
# Selection + reshape
# ---------------------------------------------------------------------------

def _select_and_audit(df: pd.DataFrame, canonical: pd.DatetimeIndex,
                      source_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter to the canonical time window, reindex per-station to the hourly grid,
    compute per-(station, feature) missingness against the canonical grid, and
    retain only stations where the worst feature is <= MISSINGNESS_THRESHOLD.

    Returns:
      kept_df: long-format DataFrame limited to retained stations, canonical grid
      audit:   per-station audit table (worst-feature missingness pct etc.)
    """
    df = df[(df["DateT"] >= canonical[0]) & (df["DateT"] <= canonical[-1])]
    df = df[~df["StasName"].isin(KNOWN_FAIL)]
    n_hours = len(canonical)

    kept_rows = []
    audit_rows = []
    for stn, g in df.groupby("StasName", sort=False):
        g = g.drop_duplicates(subset="DateT").set_index("DateT").reindex(canonical)
        misses = {v: 100 * g[v].isna().sum() / n_hours for v in CANONICAL_VARS}
        worst = max(misses.values())
        keep = worst <= MISSINGNESS_THRESHOLD * 100
        audit_rows.append({
            "station": stn, "source": source_name, "worst_missing_pct": worst,
            "kept": keep, **misses,
        })
        if keep:
            g = g.reset_index().rename(columns={"index": "DateT"})
            g["StasName"] = stn
            kept_rows.append(g)

    kept_df = pd.concat(kept_rows, ignore_index=True) if kept_rows else pd.DataFrame()
    audit = pd.DataFrame(audit_rows).sort_values("worst_missing_pct")
    return kept_df, audit


def _load_coordinates(coord_path: Path, station_names: List[str]) -> pd.DataFrame:
    """
    Load the station-coordinates lookup and validate that every station in
    `station_names` has a non-null lat/lon. Raises with a clear message if not.
    """
    if not coord_path.exists():
        raise FileNotFoundError(
            f"Station coordinates lookup missing: {coord_path}\n"
            f"See data/saws/PLAN.md §3.2 for how to populate this file."
        )
    coords = pd.read_csv(coord_path)
    missing_from_csv = set(station_names) - set(coords["station"])
    if missing_from_csv:
        raise ValueError(
            f"{len(missing_from_csv)} station(s) present in the data but absent from "
            f"{coord_path.name}: {sorted(missing_from_csv)}"
        )
    coords = coords[coords["station"].isin(station_names)].copy()
    blanks = coords[coords[["latitude", "longitude"]].isna().any(axis=1)]
    if not blanks.empty:
        raise ValueError(
            f"{len(blanks)} station(s) in {coord_path.name} have missing lat/lon; "
            f"populate them before parsing: {blanks['station'].tolist()}"
        )
    return coords[["station", "latitude", "longitude"]].rename(
        columns={"latitude": "lat", "longitude": "lon"}
    )


def _pivot_to_tensor(kept_df: pd.DataFrame, coords: pd.DataFrame,
                     canonical: pd.DatetimeIndex) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Long-form → (T, N, D) tensor + node-ordered coords DataFrame.

    Node ordering is alphabetical by station name to guarantee determinism
    across parser runs. The returned coords DataFrame has the same row order
    as the N-axis of the tensor.
    """
    stations = sorted(kept_df["StasName"].unique())
    coords = coords.set_index("station").loc[stations].reset_index()
    coords.insert(0, "node_id", range(len(coords)))

    T, N, D = len(canonical), len(stations), len(CANONICAL_VARS)
    X = np.full((T, N, D), np.nan, dtype=np.float32)

    for n, stn in enumerate(stations):
        g = kept_df[kept_df["StasName"] == stn].drop_duplicates("DateT")
        g = g.set_index("DateT").reindex(canonical)
        for d, var in enumerate(CANONICAL_VARS):
            X[:, n, d] = g[var].values.astype(np.float32)

    return X, coords


# ---------------------------------------------------------------------------
# Public entry point (matches parse_era5_files signature)
# ---------------------------------------------------------------------------

def parse_saws_files(raw_dir: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Parse the Lekuba SAWS extracts into the pipeline's standard contract.

    Args:
      raw_dir: directory containing data1.xlsx and data2.xls. In production this
               is data/saws/extracted/ (pipeline.py points here via SAWS_RAW_DIR).

    Returns:
      X          : (T, N, D) float32, may contain NaN — imputation happens
                   downstream in preprocessing.impute
      timestamps : (T,) datetime64[ns], canonical hourly grid 2013-01-01 → 2020-12-31
      coords     : DataFrame with columns [node_id, station, lat, lon], N rows,
                   ordered alphabetically by station
      var_names  : list of D canonical variable names
    """
    raw = Path(raw_dir)
    canonical = pd.date_range(WINDOW_START, WINDOW_END, freq="h")

    d1_path = raw / "data1.xlsx"
    d2_path = raw / "data2.xls"
    if not d1_path.exists() or not d2_path.exists():
        raise FileNotFoundError(
            f"Lekuba workbooks not found in {raw}\n"
            f"Expected: data1.xlsx and data2.xls"
        )

    log.info("Reading Lekuba data1.xlsx (long-form v1)…")
    d1 = _read_lekuba_v1(d1_path)
    log.info("  data1: %d rows, %d stations", len(d1), d1["StasName"].nunique())

    log.info("Reading Lekuba data2.xls (long-form v2)…")
    d2 = _read_lekuba_v2(d2_path)
    log.info("  data2: %d rows, %d stations", len(d2), d2["StasName"].nunique())

    log.info("Auditing per-(station, feature) missingness against canonical grid "
             "(%s → %s, %d hours)…", WINDOW_START.date(), WINDOW_END.date(), len(canonical))
    kept_d1, audit_d1 = _select_and_audit(d1, canonical, "data1")
    kept_d2, audit_d2 = _select_and_audit(d2, canonical, "data2")
    audit_all = pd.concat([audit_d1, audit_d2], ignore_index=True).sort_values("worst_missing_pct")

    kept = pd.concat([kept_d1, kept_d2], ignore_index=True)
    dropped = audit_all[~audit_all["kept"]]
    if not dropped.empty:
        log.info("  Dropped %d station(s) exceeding %.0f%% missingness:",
                 len(dropped), MISSINGNESS_THRESHOLD * 100)
        for _, r in dropped.iterrows():
            log.info("    %-38s worst=%.2f%% (%s)", r["station"], r["worst_missing_pct"], r["source"])

    log.info("Loading station coordinates lookup…")
    coord_path = raw.parent / "station_coordinates.csv"
    coords = _load_coordinates(coord_path, sorted(kept["StasName"].unique()))

    log.info("Pivoting to (T=%d, N=%d, D=%d) tensor…",
             len(canonical), coords.shape[0], len(CANONICAL_VARS))
    X, coords_ordered = _pivot_to_tensor(kept, coords, canonical)

    timestamps = canonical.values.astype("datetime64[ns]")

    # Persist the audit alongside outputs for provenance
    audit_out = raw.parent / "audit" / "parse_report.csv"
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_all.to_csv(audit_out, index=False)
    log.info("  Wrote per-station audit → %s", audit_out)

    log.info("SAWS parse complete: T=%d, N=%d, D=%d",
             X.shape[0], X.shape[1], X.shape[2])
    return X, timestamps, coords_ordered, list(CANONICAL_VARS)
