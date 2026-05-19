"""
Download ERA5 hourly reanalysis data for South Africa via the CDS API.

Credentials must be in ~/.cdsapirc:
    url: https://cds.climate.copernicus.eu/api
    key: <your-personal-access-token>

Downloads one year at a time so the process is resumable — if interrupted,
re-running skips files that already exist. All 6 variables are downloaded
together per year into a single NetCDF file.

South Africa bounding box: N=-22, W=15, S=-35, E=33 (0.25° resolution).

Variables:
    2m_temperature, 2m_dewpoint_temperature, total_precipitation,
    10m_u_component_of_wind, 10m_v_component_of_wind, surface_pressure

Downloads one month at a time to stay within the CDS free-tier request size limit
(a full year of 6 variables exceeds it). Each month produces one NetCDF file.
Total files for 2010–2023: 168 (14 years × 12 months).

Usage:
    python scripts/download_era5.py                        # 2010-2023, all months
    python scripts/download_era5.py --year-start 2020 --year-end 2023
    python scripts/download_era5.py --year-start 2010 --year-end 2010  # single year
"""

import argparse
import sys
import time
from pathlib import Path

import cdsapi
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from rich.table import Table
from rich import print as rprint

console = Console()

DATASET = "reanalysis-era5-single-levels"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "era5" / "raw"

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
]

# South Africa bounding box [North, West, South, East]
AREA = [-22, 15, -35, 33]

ALL_DAYS  = [f"{d:02d}" for d in range(1, 32)]
ALL_HOURS = [f"{h:02d}:00" for h in range(24)]

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download ERA5 hourly data for South Africa")
    p.add_argument("--year-start", type=int, default=2010,
                   help="First year to download (default: 2010)")
    p.add_argument("--year-end",   type=int, default=2023,
                   help="Last year to download inclusive (default: 2023)")
    return p.parse_args()


def download_month(
    client: cdsapi.Client,
    year: int,
    month: int,
    output_dir: Path,
    progress: Progress,
    task_id,
) -> bool:
    """
    Download all 6 variables for one calendar month.
    Returns True if downloaded, False if skipped (already exists).
    """
    out_path = output_dir / f"era5_southafrica_{year}_{month:02d}.nc"
    label = f"{year}-{month:02d} ({MONTH_NAMES[month]})"

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1_048_576
        progress.update(task_id, description=f"[green]✓ {label}[/green] ({size_mb:.0f} MB)", advance=1)
        return False

    progress.update(task_id, description=f"[yellow]⏳ Queued[/yellow]  {label}")

    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year":  [str(year)],
        "month": [f"{month:02d}"],
        "day":   ALL_DAYS,
        "time":  ALL_HOURS,
        "area":  AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    tmp_path = out_path.with_suffix(".nc.tmp")
    try:
        result = client.retrieve(DATASET, request)
        progress.update(task_id, description=f"[cyan]↓ Downloading[/cyan] {label}")
        result.download(str(tmp_path))
        tmp_path.rename(out_path)
        size_mb = out_path.stat().st_size / 1_048_576
        progress.update(task_id, description=f"[green]✓ Done[/green]      {label} ({size_mb:.0f} MB)", advance=1)
        return True
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        progress.update(task_id, description=f"[red]✗ Error[/red]     {label}: {e}")
        raise


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    years = list(range(args.year_start, args.year_end + 1))
    n_files = len(years) * 12

    # Header
    console.rule("[bold cyan]ERA5 Download — South Africa")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[bold]Variables[/bold]",  ", ".join(VARIABLES))
    table.add_row("[bold]Bounding box[/bold]", "N=-22  W=15  S=-35  E=33  (0.25° res)")
    table.add_row("[bold]Period[/bold]",      f"{years[0]}–{years[-1]}  ({len(years)} years, {n_files} monthly files)")
    table.add_row("[bold]Output[/bold]",      str(OUTPUT_DIR))
    table.add_row("[bold]Est. size[/bold]",   f"~{n_files * 40}–{n_files * 70} MB total")
    console.print(table)
    console.print()

    # Suppress cdsapi's own verbose logging — we show status via rich instead
    import logging
    logging.getLogger("cdsapi").setLevel(logging.WARNING)

    client = cdsapi.Client()

    with Progress(
        SpinnerColumn(),
        MofNCompleteColumn(),
        BarColumn(bar_width=35),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:
        task = progress.add_task("Starting...", total=n_files)

        downloaded = skipped = errors = 0
        for year in years:
            for month in range(1, 13):
                try:
                    got_new = download_month(client, year, month, OUTPUT_DIR, progress, task)
                    if got_new:
                        downloaded += 1
                    else:
                        skipped += 1
                except Exception:
                    errors += 1
                    # Continue with remaining months rather than aborting
                    progress.advance(task)

    console.rule("[bold green]Complete")
    console.print(f"  Downloaded : [green]{downloaded}[/green]")
    console.print(f"  Skipped    : [dim]{skipped}[/dim]  (already on disk)")
    if errors:
        console.print(f"  Errors     : [red]{errors}[/red]  (re-run to retry failed months)")
    total_mb = sum(f.stat().st_size for f in OUTPUT_DIR.glob("era5_southafrica_*.nc")) / 1_048_576
    console.print(f"  Total size : {total_mb:.0f} MB on disk")


if __name__ == "__main__":
    main()
