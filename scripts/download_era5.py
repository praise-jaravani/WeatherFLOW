"""
Download ERA5 reanalysis data via the Copernicus Climate Data Store (CDS) API.

Requires a CDS API key configured at ~/.cdsapirc. Downloads hourly reanalysis
data for the variables and geographic bounding box covering South Africa
(roughly lat: -35 to -22, lon: 16 to 33) and saves to data/era5/raw/.

Variables downloaded (matching SAWS availability):
  - 2m_temperature
  - 2m_dewpoint_temperature  (for humidity derivation)
  - total_precipitation
  - 10m_u_component_of_wind
  - 10m_v_component_of_wind
  - surface_pressure

Usage:
  python scripts/download_era5.py --year-start 2000 --year-end 2020
"""

import argparse
from pathlib import Path

ERA5_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "era5" / "raw"


def parse_args() -> argparse.Namespace:
    raise NotImplementedError


def download_variable(variable: str, years: list, output_dir: Path) -> None:
    """Download a single ERA5 variable for the specified years via cdsapi."""
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
