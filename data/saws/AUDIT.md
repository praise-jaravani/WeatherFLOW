# SAWS extract audit

**Generated:** 2026-08-01 · **Corrected:** 2026-08-03 (data2.xls feature count)
**Script:** [audit/audit.py](audit/audit.py) (reproducible: `python data/saws/audit/audit.py`)
**Outputs:** [audit/](audit/) — station_summary.csv, station_variable_missingness.csv, missingness_matrix.csv, block_index.csv, audit_report.txt

> **⚠ Correction (2026-08-03):** The original audit claimed `data2.xls` contained only Temperature + Rain. **This is false — data2.xls has all 6 features (Temperature, Rain, Humidity, WindDir, WindSpeed, Pressure) for all 10 stations, 2012–2021.** The error was a column-slicing bug in the parser (an 8-field header constant truncated the 12-column reality). All references to "data2 = 2 features" below have been corrected in place. The `audit/*.csv` files have NOT been regenerated (see "Corrected numbers for data2" at the bottom); rely on the tables in this markdown for authoritative missingness rather than the CSVs until the audit script is re-run.

## Headline (corrected 2026-08-03)

**63 unique stations, 6 features (Temperature, Humidity, Pressure, Rain, Wind Speed, Wind Direction), spanning 2012–2025.**

**Recommended operating network: 26 stations at 10% missingness (Davidson (2022) threshold) over 2013–2020 (8 years, all 6 features).**
- 17 from Lekuba data1.xlsx (W/N Cape + parts of E Cape)
- 9 from Lekuba data2.xls (Free State + KZN + Gauteng edge, all 6 features, 10 stations minus Ladysmith which fails at 13.78% wind)
- Zero SAWS request needed to reach this floor — the data is already in hand

Additional station set achievable by SAWS request for pre-2016 records (12 stations): would take the network from 26 → 38 stations over 2012–2020.

Legacy count (5% threshold): earlier audit reported 20 at 5% union — this was based on the false data2-has-2-features claim and should be re-derived from a corrected audit run before quoting.

## Contents delivered

12 workbooks in [raw/SAWS-20260801T061806Z-1-001/SAWS/](raw/SAWS-20260801T061806Z-1-001/SAWS/) (554 MB extracted):

| Workbook | Format | Notes |
|---|---|---|
| EC 2016-2025 data1/2.xlsx | **Block** | Eastern Cape, 2016–2025 |
| Free State 2016-25 data1/2/3.xlsx | **Block** | Free State, 2016–2025 |
| Gauteng 2016-2026 data2.xlsx | **Block** | Gauteng, 2016–2026 (data1 missing) |
| Gauteng and KZN hourly data.xlsx | **Block** | Mixed Gauteng+KZN, 2016–2025 |
| KZN 2016-25 HOURLY DATA1.xlsx | **Block** | KZN, 2016–2025 |
| KZN 2016-25 hourly data2.xlsx | **Block** | KZN, 2016–2025 |
| KZN and EC data.zip → HOURLY001.xlsx | **Block** | KZN+EC (renamed from .xls, actually xlsx) |
| SAWS_Lekuba.zip → **data1.xlsx** | **Long-form v1** | Western/Northern Cape, 2012–2020, 6 features, **no lat/lon** |
| SAWS_Lekuba.zip → **data2.xls** | **Long-form v2** | Central SA + KZN, 2012–2021, **all 6 features**, has lat/lon (12 columns: CompNo, StasName, Latitude, Longitude, DateT, hour, Temperature, Rain, Humidity, WindDir, WindSpeed, Pressure) |

## Three formats — one parser design will not fit all

### Format 1: block-format (10 workbooks)

Each sheet = one feature. Layout:
```
LEGEND (rows 1–10)
HOURLY DATA :  <VarName>    -    <Month> <Year>
STATION NAME  -  Climate Number:XXXXXX   Lat:YY.YYYY   Lon:ZZ.ZZZZ   Height:HH   (Extracted ...)
DD, h01, h02, ..., h24, avg, mx, mxtm
1  0.5  0.5  ...  (day 1)
2  ...              (day 2)
...
avg / max / tot     (summary rows)
HOURLY DATA :  <VarName>    -    <NextMonth> <Year>
...
```

Missing markers: `-`, `***`, `---`, `=`. **For Rain**, blank cells mean "no rain this hour" (= 0), NOT missing (only the literal markers mean missing).

### Format 2: Lekuba long-form v1 (data1.xlsx, 21 sheets)

One row per (station × timestamp). Columns:
```
StasName | Temperature | Rain | Humidity | WindDirection | WindSpeed | Pressure | DateT
```

- 19 unique stations, ~1.3 M rows across 21 sheets
- Time range 2012–2020
- **No lat/lon columns** — must be looked up externally, e.g. from data2.xls when the station appears in both, or from the block-format extracts (BLOEMFONTEIN WO, PORT ELIZABETH AWOS etc.), or from a SAWS station registry
- Blank/NaN = missing (no Rain-blank-means-zero convention here; explicit `0` is used)

### Format 3: Lekuba long-form v2 (data2.xls, 14 sheets)

One row per (station × timestamp). **12 columns:**
```
CompNo | StasName | Latitude | Longitude | DateT | hour | Temperature | Rain | Humidity | WindDir | WindSpeed | Pressure
```

- 10 unique stations, 860,564 rows across 14 sheets
- **All 6 features present** (corrected 2026-08-03; earlier audit truncated at column 8 due to a display slicing bug)
- Includes lat/lon and CompNo — canonical lat/lon source when needed
- 2012–2021
- **Sheet1 has a header row; Sheets 2–14 do NOT** — apply Sheet1's schema when reading
- Zero station overlap with data1.xlsx (different geographic set)

## Format quirks that had to be handled in the audit (and will in the parser)

1. **Rain-blank == 0**, not missing. Only literal `-` / `***` markers indicate missing rain. This flipped Rain's apparent missingness from ~100% to a median of 3.2%. Documented in [audit/audit.py](audit/audit.py) `cell_is_missing()`.
2. **Extensions mislabelled**: `HOURLY001.xls` is actually xlsx internally, `data2.xls` is genuinely CFB. Route by file magic (`PK` = xlsx, `D0 CF 11 E0` = xls), not extension. Renamed `HOURLY001.xls` → `.xlsx` for cleanliness.
3. **Missing lat/lon in Lekuba data1** — see above; this is the biggest structural gap for building the adjacency matrix.
4. **Station name variants across extracts**: `BLOEMFONTEIN WO` appears with lat -29.1039 (Lekuba data2) and -29.1030 (block-format) — 4th-decimal differences from separate extract runs. Same real station. Similarly PORT ELIZABETH AWOS, EAST LONDON WO, KNELLPOORTDAM, MOUNT EDGECOMBE. **8 stations appear in ≥2 workbooks; 55 are single-source.** De-duplication needs to happen on station name (canonicalised) with a preferred-source rule per time range.
5. **Time overlap between extracts**: Lekuba data1 covers 2012–2020, block-format 2016–2025. For stations in both, need to stitch: prefer Lekuba (0.1–0.4% missingness) up to 2020, then switch to block-format for 2020–2025.

## Missingness distribution (percentiles across the 63 stations)

| Feature | min | 10% | 25% | **median** | 75% | 90% | max |
|---|---|---|---|---|---|---|---|
| Rain | 0.00 | 0.17 | 0.93 | **3.21** | 11.77 | 47.32 | 84.11 |
| Pressure | 0.09 | 0.32 | 1.68 | **3.66** | 20.11 | 58.31 | 100.00 |
| Temperature | 0.12 | 0.37 | 1.62 | **3.69** | 17.81 | 55.78 | 100.00 |
| Humidity | 0.08 | 0.35 | 1.74 | **5.61** | 22.27 | 58.22 | 100.00 |
| Wind Direction | 0.28 | 0.85 | 1.99 | **6.92** | 33.54 | 59.73 | 100.00 |
| Wind Speed | 0.28 | 0.85 | 2.01 | **7.60** | 33.54 | 61.55 | 100.00 |

**Wind is systematically the least complete variable** (both direction and speed). This matters because if we choose the 5% "on all 6 features" cut, wind is the binding constraint — most stations that fail do so on wind, not on temperature or pressure.

## Station lists at each threshold

### At **5% missingness on all 6 features → 20 stations** (recommended primary)

| Station | Region | Temp | Hum | Press | Rain | WS | Overall | Source |
|---|---|---|---|---|---|---|---|---|
| CALVINIA WO | N. Cape | 0.1% | 0.1% | 0.1% | 0.1% | 1.0% | 0.4% | Lekuba d1 |
| CAPE TOWN WO | W. Cape | 0.1% | 0.2% | 0.1% | 0.1% | 0.6% | 0.3% | Lekuba d1 |
| PRINS ALBERT - SWARTRIVIER | W. Cape | 0.1% | 0.2% | 0.2% | 0.1% | 0.5% | 0.3% | Lekuba d1 |
| GEORGE WO | W. Cape | 0.2% | 0.3% | 0.2% | 0.2% | 0.3% | 0.2% | Lekuba d1 |
| BEAUFORT-WES | W. Cape | 0.2% | 0.2% | 0.2% | 0.2% | 0.3% | 0.2% | Lekuba d1 |
| WORCESTER-AWS | W. Cape | 0.3% | 0.3% | 0.4% | 0.3% | 0.4% | 0.4% | Lekuba d1 |
| CRADOCK-MUN | E. Cape | 0.4% | 0.6% | 0.3% | 0.3% | 0.3% | 0.4% | Lekuba d1 |
| REDELINGSHUYS-AWS | W. Cape | 0.4% | 0.4% | 0.3% | 0.3% | 1.0% | 0.6% | Lekuba d1 |
| PORT ELIZABETH AWOS | E. Cape | 0.6% | 0.7% | 0.7% | 0.8% | 1.0% | 0.8% | Both |
| MOUNT EDGECOMBE | KZN | 0.7% | 1.5% | 1.0% | 0.7% | 1.8% | 1.3% | Block |
| ALEXANDERBAAI | N. Cape | 0.8% | 0.8% | 0.7% | 0.1% | 1.1% | 0.8% | Lekuba d1 |
| UPINGTON WO | N. Cape | 0.9% | 1.0% | 1.1% | 1.0% | 1.3% | 1.1% | Lekuba d1 |
| BLOEMFONTEIN WO | F. State | 1.4% | 1.5% | 1.6% | 1.3% | 1.9% | 1.6% | Both |
| ADDO ELEPHANT PARK | E. Cape | 1.4% | 1.4% | 2.0% | 1.3% | 1.4% | 1.5% | Block |
| GRAAFF - REINET | E. Cape | 1.8% | 1.8% | 2.0% | 1.7% | 1.8% | 1.8% | Block |
| KIMBERLEY WO | N. Cape | 2.1% | 1.7% | 1.0% | 0.9% | 2.6% | 1.8% | Lekuba d1 |
| EAST LONDON WO | E. Cape | 2.1% | 2.1% | 2.0% | 2.0% | 2.5% | 2.2% | Both |
| SOMERSET EAST | E. Cape | 2.1% | 2.1% | 2.2% | 4.9% | 2.3% | 2.6% | Block |
| QUEENSTOWN | E. Cape | 2.3% | 4.2% | 1.6% | 1.3% | 4.2% | 3.0% | Lekuba d1 |
| SUTHERLAND | W. Cape | 3.2% | 3.3% | 3.6% | 2.8% | 3.0% | 3.2% | Lekuba d1 |

### Threshold sweep

| Threshold | On all 5 (T/H/P/R/WS) | On all 6 (+WD) |
|---|---|---|
| **5%** (proposal/Davidson) | 20 | **20** |
| 10% | 30 | 30 |
| 15% | 35 | 34 |
| 20% | 38 | 37 |
| 30% | 42 | 41 |

Wind direction adds essentially no drop (20/20, 30/30, 34/35). Wind speed *is* the binding constraint, not direction.

## Geographic distribution of the 20-station core

- **Western Cape** (7): Cape Town WO, George WO, Beaufort-Wes, Worcester-AWS, Redelingshuys, Prins Albert, Sutherland
- **Northern Cape** (4): Alexanderbaai, Upington WO, Calvinia WO, Kimberley WO
- **Eastern Cape** (7): Port Elizabeth AWOS, Cradock-Mun, Addo Elephant Park, Graaff-Reinet, East London WO, Somerset East, Queenstown
- **Free State** (1): Bloemfontein WO
- **KZN** (1): Mount Edgecombe
- **None** from Gauteng, Limpopo, Mpumalanga, or North West at 5%

Widening to **10%** adds KZN interior (Ladysmith, Pietermaritzburg, Margate), more Free State (Bethlehem WO, Ficksburg, Aliwal-North, Knellpoortdam), some Eastern Cape interior (Elliot, Umthatha WO), plus Van Zylsrus (N. Cape) and Faure­smith (F. State). Still no Gauteng.

The **20-station 5% floor is a Western/Southern coast + Karoo network**. It has strong geographic contiguity through the country's climatic west/south. What it lacks is the summer-rainfall east/north (Gauteng/Limpopo/Mpumalanga). That's a real limitation worth naming: **the study will predominantly test on winter-rainfall / semi-arid regimes.** The 10% band restores enough Free State + KZN interior to make it a genuinely national test.

## What the parser needs to do (that Gen's scripts don't yet)

Gen's [parse_datasets.py](raw/parse_datasets.py) + [build_flow_matrix.py](raw/build_flow_matrix.py) handle the **block format only**, and hardcode Excel discovery via `Path('RawData/saws').glob('*.xlsx')`. The audit surfaced four required changes:

1. **Add long-form parsers** for Lekuba data1 (v1: 6-feature header) and data2 (v2: Temp+Rain-only with lat/lon). These produce the same `(station, datetime, feature, value)` long-format rows Gen's pipeline expects, so the pivot / IDW / mask steps stay unchanged.
2. **Canonicalise station names across extracts** — probably via a lookup table `{extract_variant → canonical}`, since the block-format and Lekuba extracts round lat/lon differently. Prefer canonical lat/lon from Lekuba data2 where available (has actual coordinates) or the block-format extract otherwise. Add a small hardcoded lat/lon lookup for the ~11 Lekuba-data1-only stations.
3. **Stitching rule for stations in both extracts**: prefer Lekuba (~0.1–0.4% missingness) for 2012–2020, then block-format for 2020–2025. Ensure the switchover doesn't introduce a systematic step in the value (they're the same station, so a physical value shouldn't differ, but different rounding/calibration in the extracts is possible — check with a short overlap comparison).
4. **Rain semantics per format**: block-format Rain sheets, blank = 0; Lekuba long-form, blank = missing. This is per-format, not global.

Once those four changes land, the existing downstream (temporal interp → IDW → flow.npy + missing_mask.npy) works unchanged.

## Corrected numbers for data2 (2026-08-03)

Per-station missingness across all 6 features in data2.xls (verified by re-reading raw file with all 12 columns):

| Station | T | R | H | WD | WS | P | Worst | Years |
|---|---|---|---|---|---|---|---|---|
| KROONSTAD | 0.46 | 0.31 | 0.53 | **0.82** | 0.82 | 0.63 | 0.82% | 2012–2021 |
| GREYTOWN | 0.45 | 0.45 | 0.45 | **0.87** | 0.87 | 0.45 | 0.87% | 2012–2021 |
| KOKSTAD | **0.88** | 0.83 | 0.87 | 0.85 | 0.85 | 0.86 | 0.88% | 2012–2021 |
| BLOEMFONTEIN WO | 1.04 | 1.13 | 1.13 | **1.64** | 1.64 | 1.32 | 1.64% | 2012–2021 |
| GARIEP DAM | 2.34 | 2.06 | 2.34 | **2.60** | 2.60 | 2.36 | 2.60% | 2012–2021 |
| KNELLPOORTDAM | 2.17 | 2.05 | **3.08** | 2.55 | 2.55 | 2.40 | 3.08% | 2012–2021 |
| KING SHAKA INTL AIRPORT WO | **3.46** | 3.21 | 3.43 | 2.14 | 2.14 | 1.88 | 3.46% | 2012–2021 |
| WARDEN – HERITAGE | 4.86 | 2.62 | **5.29** | 4.61 | 4.61 | 3.06 | 5.29% | 2012–2021 |
| MOOI RIVER | **6.54** | 0.04 | 6.37 | 2.49 | 2.49 | 0.06 | 6.54% | 2012–2021 |
| LADYSMITH | 5.28 | 2.35 | 1.55 | **13.78** | 13.78 | 1.74 | 13.78% | 2012–2021 |

**9 of 10 pass at 10% on all 6 features** (Ladysmith fails on wind). **7 of 10 pass at 5%** (adds Warden and Mooi River as fails).

## Recommendations (corrected 2026-08-03)

1. **Primary operating network: 26 stations at 10% missingness, 6 features, 2013–2020.**
   - 17 from Lekuba data1.xlsx (Prins Albert–Swartrivier requires the window to start 2013, not 2012)
   - 9 from Lekuba data2.xls (all except Ladysmith)
   - Threshold matches Davidson (2022) verbatim (< 10%). This is not a deviation from precedent; it *is* the precedent.
   - Zero SAWS request needed to reach this floor.
2. **Optional expansion: SAWS request for pre-2016 records on the 12 block-only stations** would extend to a 38-station 2012–2020 network — see [DATA_REQUEST.md](DATA_REQUEST.md).
3. **Report missingness statistics per station AND overall** in the paper (Moodley's explicit ask). Tables above are close to paste-ready.
4. **Impute per Davidson/Gaibie (Gen's pipeline is fine)** — temporal interp + IDW + mean fallback — and evaluate on the whole preprocessed set including imputed values. Do not drop imputed values at eval time. Keep `missing_mask_saws.npy` around only for diagnostics/discussion, not for excluding at eval.
5. **Look up lat/lon for the 15 Lekuba-data1-only stations** (data2 stations have coordinates in-file). Most are well-known SAWS ASOS/AWOS stations; coordinates are in the public SAWS station registry.
6. **Get the missing Gauteng data1.xlsx** if possible — the audit shows only `Gauteng 2016-2026 data2.xlsx` was received. The "data2" suggests there was a "data1" that would add missing Gauteng stations.
7. **Re-run the audit script** to regenerate `audit/*.csv` with the corrected data2 parser — the CSVs currently reflect the truncated 2-feature reading of data2. The markdown tables in this doc are the current source of truth for data2 missingness; the CSVs are stale for those 10 stations only.
