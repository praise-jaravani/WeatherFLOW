# ERA5 Data Notes

## Download

Script: `scripts/download_era5.py`  
Target period: 2010–2023 (168 monthly files)  
Output: `data/era5/raw/era5_southafrica_YYYY_MM.nc`

**Credentials** (`~/.cdsapirc`, chmod 600):
```
url: https://cds.climate.copernicus.eu/api
key: <personal-access-token>
```
Never commit this file to git. Requires `cdsapi>=0.7.0` (new CDS endpoint — 0.6.x will not work).

## File format

Despite the `.nc` extension, the CDS API (post-Sept 2024) delivers ERA5 as **ZIP archives**. Each file contains two inner NetCDF4 files:

| Inner file | Variables |
|---|---|
| `data_stream-oper_stepType-instant.nc` | t2m, d2m, u10, v10, sp |
| `data_stream-oper_stepType-accum.nc` | tp (total precipitation) |

`preprocessing/era5_parser.py` handles this transparently — do not try to open the `.nc` files directly with xarray/netCDF4.

## Grid

- Bounding box: N=-22, W=15, S=-35, E=33 (South Africa)
- Native resolution: 0.25° → 53 lat × 73 lon = **3,869 nodes**
- Default stride (lat=2, lon=2): ~0.5° → 27 × 37 = **999 nodes**

## Variables

| Short name | Long name | Units |
|---|---|---|
| t2m | 2m temperature | K |
| d2m | 2m dewpoint temperature | K |
| tp | Total precipitation | m |
| u10 | 10m U-component of wind | m/s |
| v10 | 10m V-component of wind | m/s |
| sp | Surface pressure | Pa |

## Download status (as of 2026-05-19)

- 2010: all 12 months ✓
- 2011: Jan–Jul ✓ (still downloading)
- 2012–2023: pending

Re-running `python scripts/download_era5.py` is safe — it skips files that already exist.
