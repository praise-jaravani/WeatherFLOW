# 32-station SAWS Network — Station Registry

**Network scope**: 32 stations passing ≤ 10% overall missingness on all 6 canonical variables (t2m, tp, rh, wd, ws, sp) across 2016-01-01 → 2025-12-31 hourly (87,672 timesteps). Selection follows the Davidson (2022) criterion, applied by [data/saws/audit/audit_60node.py](../data/saws/audit/audit_60node.py); Gen Chikwanha, Zoleka Fuyana, and this pipeline all independently arrived at N=32 (2026-08-07 / 2026-08-08).

**Node indexing**: alphabetical by station name (see [preprocessing/saws_parser.py:295](../preprocessing/saws_parser.py#L295) — `stations = sorted(STATION_WHITELIST_32)`). `node_id` matches the position along the station axis of every array in [preprocessing/cache/saws/](../preprocessing/cache/saws/): `X_train.npy`, `y_train.npy`, `adj.npy`, `coords.npy`, `thresholds*.npy`, etc.

**Coordinate source**: extracted from the raw SAWS block-format workbooks (station header line `Lat:YY.YY  Lon:ZZ.ZZ`) by [preprocessing/saws_parser.py](../preprocessing/saws_parser.py); cached to [preprocessing/cache/saws/coords.npy](../preprocessing/cache/saws/coords.npy).

**Province source**: manual verification against known South African town/city locations for each station name. The `location_note` column identifies the settlement, park, or research facility so a reader can independently verify (map lookup or SAWS gazetteer). Not an automated GIS lookup.

## Station table

| node_id | Station | Latitude (°S) | Longitude (°E) | Province | Location note |
|---|---|---|---|---|---|
| 0 | ADDO ELEPHANT PARK | 33.4420 | 25.7480 | Eastern Cape | national park ~70 km NE of Gqeberha (Port Elizabeth) |
| 1 | ALIWAL-NORTH PLAATKOP | 30.8020 | 26.8830 | Eastern Cape | Aliwal North town on the Orange River, EC |
| 2 | BETHLEHEM WO | 28.2490 | 28.3340 | Free State | Bethlehem town, eastern Free State |
| 3 | BLOEMFONTEIN WO | 29.1030 | 26.2980 | Free State | Bloemfontein (FS capital, national judicial capital) |
| 4 | CAPE ST FRANCIS | 34.2110 | 24.8340 | Eastern Cape | coastal EC, ~90 km SW of Gqeberha |
| 5 | CEDARA | 29.5410 | 30.2650 | KwaZulu-Natal | Cedara Agricultural Research Centre, near Howick, KZN |
| 6 | DOHNE - AGR | 32.5270 | 27.4600 | Eastern Cape | Dohne Agricultural Research Institute, near Stutterheim, EC |
| 7 | EAST LONDON WO | 33.0350 | 27.8160 | Eastern Cape | East London coastal city, EC |
| 8 | ELLIOT | 31.3360 | 27.8400 | Eastern Cape | Elliot town, EC (former Transkei region) |
| 9 | FAURESMITH | 29.7530 | 25.3230 | Free State | Fauresmith town, southern FS |
| 10 | FICKSBURG | 28.8270 | 27.9040 | Free State | Ficksburg on the Lesotho border, eastern FS |
| 11 | GRAAFF - REINET | 32.1930 | 24.5430 | Eastern Cape | Graaff-Reinet, historic Karoo town in EC |
| 12 | GRAHAMSTOWN | 33.2900 | 26.5020 | Eastern Cape | Makhanda (Grahamstown), EC |
| 13 | KNELLPOORTDAM | 29.7860 | 26.9030 | Free State | Knellpoort Dam, near Rosendal, eastern FS |
| 14 | KOKSTAD | 30.5020 | 29.3940 | KwaZulu-Natal | Kokstad, southern KZN (post-2006 boundary; historically EC) |
| 15 | LADYSMITH | 28.5750 | 29.7500 | KwaZulu-Natal | Ladysmith, northern KZN |
| 16 | MAKATINI RESEARCH CENTRE | 27.3940 | 32.1760 | KwaZulu-Natal | Makatini Flats, northern KZN |
| 17 | MARGATE | 30.8580 | 30.3430 | KwaZulu-Natal | Margate, KZN South Coast |
| 18 | MOUNT EDGECOMBE | 29.7060 | 31.0460 | KwaZulu-Natal | Mount Edgecombe, near Durban, KZN |
| 19 | NGQURA - COEGA | 33.8050 | 25.6690 | Eastern Cape | Port of Ngqura / Coega IDZ, near Gqeberha, EC |
| 20 | ORIBI AIRPORT | 29.6470 | 30.3990 | KwaZulu-Natal | Pietermaritzburg Airport, KZN |
| 21 | PIETERMARITZBURG | 29.6270 | 30.4020 | KwaZulu-Natal | Pietermaritzburg, KZN provincial capital |
| 22 | PONGOLA | 27.4130 | 31.5900 | KwaZulu-Natal | Pongola, northern KZN |
| 23 | PORT ALFRED - AIRPORT | 33.5590 | 26.8800 | Eastern Cape | Port Alfred, EC coast |
| 24 | PORT ELIZABETH AWOS | 33.9860 | 25.6160 | Eastern Cape | Gqeberha (Port Elizabeth), EC |
| 25 | PRETORIA UNIVERSITY PROEFPLAAS | 25.7520 | 28.2580 | Gauteng | UP Experimental Farm (Proefplaas), Pretoria, Gauteng |
| 26 | RIVERVIEW | 28.4440 | 32.1820 | KwaZulu-Natal | Riverview, near Mtubatuba/Richards Bay, KZN |
| 27 | ROYAL NATIONAL PARK | 28.6850 | 28.9540 | KwaZulu-Natal | Royal Natal National Park, Drakensberg, KZN |
| 28 | SOMERSET EAST | 33.0490 | 25.7190 | Eastern Cape | Somerset East, EC |
| 29 | UMTHATHA WO | 31.5490 | 28.6730 | Eastern Cape | Mthatha (Umtata), EC (former Transkei) |
| 30 | VAN REENEN | 28.3780 | 29.3850 | KwaZulu-Natal | Van Reenen's Pass, KZN side of the FS/KZN border |
| 31 | ZUURBEKOM AWS | 26.3000 | 27.8130 | Gauteng | Zuurbekom, west of Johannesburg, Gauteng |

**Total: 32 stations**

## Provinces summary

| Province | Count |
|---|---|
| Eastern Cape | 13 |
| KwaZulu-Natal | 12 |
| Free State | 5 |
| Gauteng | 2 |

Geographic coverage: eastern half of South Africa (Eastern Cape → KZN axis, with Free State interior and Gauteng anchors). No stations in Western Cape, Northern Cape, North West, Mpumalanga, or Limpopo — a limitation of the underlying SAWS block-format extract we received. Documented in the paper's Limitations.

## What this table does NOT contain (and why)

The raw SAWS workbook block headers (`<Station name> - Climate Number:XXXXXX  Lat:YY.YY  Lon:ZZ.ZZ  Height:HH`) additionally provide **WMO/SAWS climate number** and **station elevation (m)**. Both are extracted by the parser ([preprocessing/saws_parser.py:118-121](../preprocessing/saws_parser.py#L118-L121)) but **not currently persisted to the cache** — `coords.npy` stores only `(lat, lon)`. If a paper table needs climate numbers and elevations, re-parse the raw workbooks via `_scan_blocks()`. Not blocked, just deferred — not needed for training/evaluation.

## Reproducing the current table

```bash
python3 -c '
from preprocessing.saws_parser import STATION_WHITELIST_32
import numpy as np, pandas as pd
c = np.load("preprocessing/cache/saws/coords.npy")
s = sorted(STATION_WHITELIST_32)
print(pd.DataFrame({"node_id": range(32), "station": s,
                     "latitude": c[:,0], "longitude": c[:,1]}))
'
```

Station whitelist is defined at [preprocessing/saws_parser.py:76-109](../preprocessing/saws_parser.py#L76-L109), asserted to be exactly 32 stations at line 110. The coordinate array is produced by `parse_saws_files()`, indexed by the alphabetical sort of `STATION_WHITELIST_32`. CSV form: [data/saws/stations_32node.csv](../data/saws/stations_32node.csv).
