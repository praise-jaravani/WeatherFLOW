# SAWS 26-Station Pipeline — Implementability Audit & Extension Plan

**Generated:** 2026-08-03 · **Locked:** 2026-08-03 (25 stations after canonical-grid audit)
**Scope:** Build a SAWS-based training corpus from Lekuba data1.xlsx + data2.xls (no SAWS re-request needed). **Target: 25 stations × 6 features × 10% missingness × 2013–2020 (canonical hourly grid), matching Davidson (2022) methodology.**

> **Station-count correction (2026-08-03):** Under the strict canonical-grid missingness rule (which is what the actual pipeline applies), WARDEN – HERITAGE fails at 16.98% because its extract doesn't start until well into 2014. Dropping it takes the network from 26 → 25. All other stations pass comfortably (worst case: Van Zylsrus WindDir 9.57%).

---

## Part 1 — Implementability Audit

### Verdict: **Fully implementable from data in hand.** No SAWS request required.

### Verified against raw files (not the stale audit CSVs)

**Window 2013-01-01 → 2020-12-31 (8 years, 70,128 hours):**

| Source | Total stations | Pass @ 10% on all 6 (canonical-grid rule) | Notes |
|---|---|---|---|
| data1.xlsx | 18 | **17** | PORT ELIZABETH – WK dropped (1 month of data only) |
| data2.xls | 10 | **8** | LADYSMITH dropped (WindDir 13.78%); WARDEN – HERITAGE dropped (16.98% overall — extract starts mid-2014) |
| Total (disjoint sets) | 28 usable | **25** | Zero overlap between data1 and data2 |

**Locked 25-station network is confirmed feasible under the strict canonical-grid rule.**

### Time-window caveat you need to decide on

You said "2012–2020" — but PRINS ALBERT – SWARTRIVIER (data1) starts April 2013, so it has 86.1% of a 2012-onwards window covered. Two ways to resolve:

- **Option A — 2013–2020 (8 years):** 26 stations, all fully covered. **Recommended.** Same window length as Davidson (2012–2019, 8 years). Cleaner pipeline.
- **Option B — 2012–2020 (9 years):** 25 stations (drop Prins Albert, or accept 12 months of imputed data at that node from IDW-of-neighbours which is heavy fabrication for 1 station).

The rest of this plan assumes **Option A (2013–2020)**. If you want B, only the year-range constants change; the 26↔25 delta is a single station.

### Where the numbers came from

Per-station worst-feature missingness inside the 2013–2020 window (verified from raw files):

- data1 range: 0.2% (Beaufort-Wes) → 6.5% (Van Zylsrus). All under 10%.
- data2 range: 0.8% (Kroonstad) → 6.5% (Mooi River). Ladysmith 13.8% (excluded).

Full per-station table is in [AUDIT.md](AUDIT.md).

### Structural risks — audited and clear

| Risk | Status | Notes |
|---|---|---|
| Station-name collisions between data1 and data2 | ✅ None | Verified: zero station overlap between the two Lekuba workbooks |
| Missing lat/lon for data1 stations | 🟡 Partial | 14 of the 17 data1 stations need coordinates from external source. 3 (East London WO, Port Elizabeth AWOS, Umthatha WO) can pull from Gen's block-format extract. All 10 data2 stations have in-file coordinates. **Manual lookup required for 14.** |
| Wind Direction periodicity (0° = 360°) | 🟡 Design choice | Sin/cos decomposition is standard; Gen's script has the code. Decision needed on whether to feed model as 7 channels or recombine to 6 degrees. |
| Rain semantics differ between block-format and Lekuba | ✅ Not an issue | Lekuba uses explicit 0 for "no rain" (verified). Only the block-format has the blank-means-zero convention. Since this plan uses Lekuba only, no per-source special-case needed. |
| Sheets 2-14 of data2.xls have no header row | ✅ Understood | Only Sheet1 has headers. Parser must apply Sheet1's schema to all sheets. |

---

## Part 2 — What's already in the pipeline (reuse as-is)

The existing preprocessing modules are **almost entirely dataset-agnostic**. From [preprocessing/](../../preprocessing/):

| Module | Status for SAWS | Change needed |
|---|---|---|
| `impute.py` | ✅ Reusable | Default `exclusion_threshold=0.05`; override to 0.10 via config for SAWS |
| `graph.py` | ✅ Reusable | Adjust `distance_threshold_km` + `sigma_km` for SAWS station spacing (~50-500km typical) |
| `normalise.py` | ✅ Reusable | Per-node per-variable z-score works fine |
| `split.py` | ✅ Reusable | Default 60/20/20 chronological. Davidson used walk-forward — can add later if needed |
| `dataset.py` | ✅ Reusable | Sliding-window generator is dataset-agnostic |
| `labels.py` | 🟡 Minor rename | Uses hardcoded `"tp"` variable name for wet-hour rule. Standardise SAWS output to use `"tp"` for Rain in the parser (simplest), OR add a `"Rain"` alias in labels.py |
| `pipeline.py` | 🟡 Small change | Manifest fingerprint hardcoded to `*.csv` — needs `*.xlsx` and `*.xls`. Otherwise the SAWS branch is already wired up (calls `parse_saws_files` which is currently a stub) |
| `derive.py` | ❌ Not applicable | ERA5-specific (derives ws10/rh from u10/v10/t2m/d2m). All 6 SAWS features are primary observations. |

---

## Part 3 — What needs to be built

### 3.1 `preprocessing/saws_parser.py` — the main deliverable

Current state: **23-line stub** with `NotImplementedError`. Needs full implementation.

Target contract (matching `parse_era5_files` signature so `pipeline.py` calls it transparently):

```python
def parse_saws_files(raw_dir: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """
    Returns:
      X          : (T=70128, N=26, D=6) float32, may contain NaN for missing
      timestamps : (T,) datetime64[h] array, 2013-01-01 00:00 → 2020-12-31 23:00
      coords     : DataFrame with columns [node_id, station_name, lat, lon]
      var_names  : ["t2m", "tp", "rh", "wd", "ws", "sp"]   (Davidson-lineage names)
    """
```

Internal responsibilities:

1. **Parse data1.xlsx** (long-form v1)
   - Read all 21 sheets, apply header from Sheet1 or assume schema
   - Columns: `StasName, Temperature, Rain, Humidity, WindDirection, WindSpeed, Pressure, DateT`
   - Filter to 2013-01-01 to 2020-12-31
   - Drop `PORT ELIZABETH - WK` (unusable — 1 month of data)

2. **Parse data2.xls** (long-form v2)
   - **Sheet1 has header; Sheets 2-14 don't** — apply Sheet1's schema to all
   - Columns: `CompNo, StasName, Latitude, Longitude, DateT, hour, Temperature, Rain, Humidity, WindDir, WindSpeed, Pressure`
   - Filter to 2013-01-01 to 2020-12-31
   - Drop `LADYSMITH` (WindDir 13.78%)

3. **Merge and station-select**
   - Compute per-(station, feature) missingness inside the 2013–2020 window
   - Retain stations where `max(missing) <= 0.10` on all 6 features
   - Log excluded stations with reasons

4. **Attach lat/lon**
   - Read `data/saws/station_coordinates.csv` (new file — see 3.2)
   - Merge onto station list; assert no missing coordinates

5. **Build canonical time grid**
   - `pd.date_range("2013-01-01 00:00", "2020-12-31 23:00", freq="h")` → 70,128 timestamps
   - Pivot to wide format (datetime × station × feature), reindex to canonical grid, missing → NaN
   - This is what feeds the `(T, N, D)` tensor

6. **Standardise variable naming (Davidson-lineage)**
   - Rename features to lower-snake matching Davidson/ERA5 conventions:
     Temperature → `t2m`, Rain → `tp`, Humidity → `rh`, WindDir → `wd`, WindSpeed → `ws`, Pressure → `sp`
   - Keeps `labels.py`'s `"tp"` wet-hour rule working without modification

7. **Wind direction decomposition** — see Design Choice 3.5

Estimated size: ~200 lines, mostly tabular manipulation. Reference: Gen's `raw/build_flow_matrix.py` (~450 lines) does the block-format equivalent; this is a slimmer per-format version.

### 3.2 `data/saws/station_coordinates.csv` — new file

A CSV mapping station name → (lat, lon, height_m, climate_number) for the 14 data1-only stations that don't have coordinates in the raw file.

**These need manual lookup** — I cannot fabricate coordinates from town names. Sources (in priority order):

1. Davidson (2022) supplementary if available (Fig 1 map has locations; supplementary may have exact coords)
2. SAWS station registry (public site) or SAWS reply email
3. From other UCT-associated theses (Gaibie 2024 supplementary)
4. Google Earth per-town pin (last resort, ±1 km accuracy)

Template:
```csv
station,lat,lon,height_m,climate_number
ALEXANDERBAAI,,,,
BEAUFORT-WES,,,,
CALVINIA WO,,,,
CAPE TOWN WO,,,,
CRADOCK-MUN,,,,
GEORGE WO,,,,
KIMBERLEY WO,,,,
PRINS ALBERT - SWARTRIVIER,,,,
QUEENSTOWN,,,,
REDELINGSHUYS-AWS,,,,
SUTHERLAND,,,,
UPINGTON WO,,,,
VAN ZYLSRUS,,,,
WORCESTER-AWS,,,,
```

The other 12 stations of the 26 already have lat/lon in-file (10 from data2, plus 3 available in Gen's block-format extract: EAST LONDON WO, PORT ELIZABETH AWOS, UMTHATHA WO). The parser will pull those automatically; only the 14 above need manual entry.

### 3.3 `experiments/configs/saws_lekuba.yaml` — new config

Small YAML for the SAWS run. Key fields (matches what `pipeline.py` reads):

```yaml
dataset: saws
task: regression
data:
  input_len: 24         # 24-hour context, Davidson's short setting
  horizon: 24           # 24-hour prediction, per Moodley redirect
  label_mode: per_variable
preprocessing:
  exclusion_threshold: 0.10       # Davidson's 10% threshold, not the default 5%
  distance_threshold_km: 500.0    # tuned for SAWS station spacing
  sigma_km: 200.0                  # bandwidth
  train_ratio: 0.6                 # or Davidson walk-forward — see Design Choice 3.6
  val_ratio: 0.2
training:
  batch_size: 64
```

### 3.4 Small edits to existing modules

- **`preprocessing/pipeline.py`** line 104: change `_manifest_fingerprint(SAWS_RAW_DIR, "*.csv")` → glob pattern covering `*.xlsx` and `*.xls` in `data/saws/extracted/`. Also update `SAWS_RAW_DIR` path to point at `extracted/` rather than `raw/`.

- **`preprocessing/labels.py`** — no change if the parser renames Rain → `tp`. Otherwise add a `TP_ALIAS = ("tp", "Rain")` lookup.

---

## Part 4 — Design choices needing your input

### 4.1 Time window: 2013–2020 or 2012–2020?

- **2013–2020**: 26 stations. Recommended.
- **2012–2020**: 25 stations (drop Prins Albert) or 26 with Prins Albert fabricated for 2012.

### 4.2 Wind direction handling

Three options in order of complexity:

- **A. Sin/cos as separate channels → 7 model channels** (T, R, H, WD_sin, WD_cos, WS, P). Standard for STGNN weather work. Requires labels.py to skip WD_sin/WD_cos in extreme thresholding. Small increase in model input width.
- **B. Sin/cos internally for interpolation, recombine to degrees → 6 model channels.** Gen's approach. Imputation-accurate; normalisation still slightly wrong at the 0°/360° boundary but rarely matters for extremes.
- **C. Keep WD as raw degrees → 6 model channels.** Simplest, gets 0°/360° wraparound wrong during both imputation and normalisation. Not recommended.

Davidson doesn't explicitly say which he uses. I'd recommend **A** for correctness in the model; **B** if you want to match Gen's script as-is.

### 4.3 Train/val/test split

- **Default 60/20/20 chronological**: uses last 20% (~1.6 years) as test. Simple.
- **Davidson walk-forward (27 splits, expanding train)**: adds complexity but is the closer methodological match to the paper. Not required for a first working pipeline.

Recommend: start with default 60/20/20. Add walk-forward as a `split_mode: walk_forward` option later if Moodley asks for direct parity.

### 4.4 Adjacency-matrix parameters

Davidson used a "thresholded Gaussian kernel" but I don't have his exact sigma/threshold from the paper text. Reasonable SAWS defaults given the geographic spread of your 26 nodes:

- `distance_threshold_km`: 500 (roughly the median pairwise distance for a 26-node network spread across 5 provinces)
- `sigma_km`: 200

These can be tuned later; not a blocker for the first pipeline run.

### 4.5 Variable naming

- **Match Davidson/ERA5**: `t2m, tp, rh, wd, ws, sp`. Recommended — keeps `labels.py`'s `"tp"` wet-hour rule working, reuses existing evaluation code.
- **Match SAWS raw**: `Temperature, Rain, Humidity, WindDir, WindSpeed, Pressure`. More faithful to the source but requires touching labels.py.

Recommend option 1 (Davidson-lineage names).

---

## Part 5 — Sequenced work order

**Estimate: 1–2 days of focused work if lat/lon lookup is fast.**

| # | Task | Effort | Blocker for | Verification |
|---|---|---|---|---|
| 1 | Create `station_coordinates.csv` with 14 manual entries | 30 min – 2h | Everything downstream | All 14 rows populated; lat in [-35, -25], lon in [15, 32] |
| 2 | Implement `parse_lekuba_v1(path)` in `saws_parser.py` | 1–2h | Testing | Returns long-form DataFrame with 17 stations, 2013-2020 window |
| 3 | Implement `parse_lekuba_v2(path)` in `saws_parser.py` | 1h (simpler) | Testing | Returns long-form DataFrame with 9 stations, 2013-2020 |
| 4 | Implement `parse_saws_files(raw_dir)` orchestrator | 1–2h | Pipeline run | Returns (X, timestamps, coords, var_names) matching contract |
| 5 | Update `pipeline.py` glob pattern + `SAWS_RAW_DIR` | 15 min | Pipeline run | Manifest hash stable across runs |
| 6 | Write `experiments/configs/saws_lekuba.yaml` | 15 min | Pipeline run | YAML parses; config keys read correctly |
| 7 | Smoke test: `python cli.py --dataset saws --smoke` | 30 min | — | Runs end-to-end; produces flow.npy shape (70128, 26, 6); adjacency (26, 26); no NaN after imputation |
| 8 | Add `tests/test_saws_parser.py` with fixture-based unit tests | 1–2h | — | Parser tests green |

**Suggested checkpoints:**

- After step 4: dump station list, missingness per station, coverage summary to `data/saws/build_report.txt`. Compare against AUDIT.md numbers. If they differ, stop and reconcile before proceeding.
- After step 7: verify flow.npy shape, missingness in raw is <10% per station, imputed flow has zero NaN, mask correctly labels which cells were imputed.
- Before any modelling: **do not begin training until Moodley sees the build_report.txt.** Explicit sign-off on station list, threshold, and window before compute burn.

---

## Part 6 — What this plan does NOT include (deliberately)

- **The SAWS re-request** ([DATA_REQUEST.md](DATA_REQUEST.md)). That's a separate path for expanding the network from 26 → 38 stations. Not needed to start modelling. Can be pursued in parallel.
- **Walk-forward validation** (Davidson's 27-split protocol). Can be added as `split_mode: walk_forward` after the base pipeline works.
- **Adjacency-matrix tuning experiments**. Sensible defaults for now; tune only if adjacency-related model performance is anomalous.
- **Continental ERA5 exploratory runs**. That was the frozen ClimODE branch's optional revival trigger — separate mission, out of scope here.

---

## Immediate next step (recommended)

Confirm answers to Design Choices 4.1 (2013 vs 2012), 4.2 (wind), and 4.5 (variable naming), then I'll start on task 1 (station coordinates template) followed by task 2 (parse_lekuba_v1). Task 1 is where the human-in-the-loop lookup happens; once that's populated the rest is mechanical.
