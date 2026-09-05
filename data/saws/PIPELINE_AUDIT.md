# SAWS 25-Station Pipeline — Regression Tests + Full Audit

**Generated:** 2026-08-03
**Reproduce:** `python data/saws/audit/pipeline_audit.py`
**Regression tests:** `pytest tests/test_impute.py tests/test_saws_parser.py -v`

Result: **46/46 audit checks pass · 33/33 regression tests pass.**

Two real defects were found and fixed during this audit; both had regression tests added to prevent recurrence.

---

## Part 1 — Regression tests added

### `tests/test_impute.py` — 10 tests

Covers `preprocessing.impute` including the new temporal-fill fallback:

| Test | What it pins |
|---|---|
| `test_no_exclusion_when_all_clean` | `exclude_sparse_nodes` no-op on clean data |
| `test_drops_high_missing_node` | drops nodes exceeding threshold |
| `test_at_threshold_boundary` | keeps a node with exactly 5% missing at threshold 5% |
| `test_no_op_when_no_nan` | `idw_impute` no-op on clean data |
| `test_spatial_fill_uses_neighbours` | single-cell IDW yields correct weighted mean |
| `test_temporal_fill_activates_when_all_neighbours_missing` | **SAWS 2020-12-31 regression** — 3-hour outage across ALL neighbours, temporal fallback must fill |
| `test_temporal_fallback_bfill_at_leading_edge` | bfill from first valid hour when outage is at t=0 |
| `test_temporal_fallback_ffill_at_trailing_edge` | ffill from last valid hour — literal SAWS 2020-12-31 pattern |
| `test_spatial_still_preferred_over_temporal` | temporal fallback runs after spatial IDW, not before |
| `test_assertion_when_entire_column_nan` | assertion trips loudly if a whole (station, variable) column is 100% NaN |

### `tests/test_saws_parser.py` — 23 tests

Covers `preprocessing.saws_parser` contract + pipeline defaults for SAWS:

- **Constants (6)**: canonical var names, WD renames for both spellings, KNOWN_FAIL list, window bounds, threshold value
- **Coordinates loader (4)**: missing file / missing station / null lat-lon / valid lookup
- **Parser contract (10)** *(requires raw Lekuba workbooks)*: shape (70128, 25, 6), var order, dtype float32, canonical hourly timestamps, alphabetical node ordering, coord range within SA, missingness ≤ 10%, KNOWN_FAIL absent from output, no all-NaN column, wd in [0, 360°]
- **Pipeline defaults for SAWS (2)**: SAWS defaults produce a connected graph; ERA5 defaults leave ALEXANDERBAAI isolated (regression test naming why the dataset-aware branch exists)
- **Parse report side-effect (1)**: audit CSV written on every parse

---

## Part 2 — Defects found and fixed during the audit

### Defect A: silent NaN in X_test from IDW failure on regional outage

**Symptom:** MTGODE test evaluation returned NaN for base MAE/RMSE/MSE/sMAPE across all variables (extreme/non-extreme sub-metrics were fine).

**Root cause chain:**
1. Lekuba data1.xlsx cuts off at 2020-12-31 00:00 for all 17 Cape stations
2. Parser reindexes to canonical 2013-01-01 → 2020-12-31 23:00 grid → 23-hour NaN block at tail
3. `idw_impute` tries to fill each NaN by weighted mean of 5 nearest spatial neighbours at the same hour
4. For all 9 Cape stations, their 5 nearest neighbours are also Cape stations — also in the outage
5. IDW: "no valid neighbours" → leave as NaN (silently, per [impute.py:88](../../preprocessing/impute.py#L88))
6. `X_test.npy` shipped with 1,242 residual NaN cells (0.06% of test data, but 100% of test hours 14004–14026)
7. Test eval's `per_station_var_mae` uses `.mean()` (not `nanmean`) → NaN propagates into all 25 stations for the affected variable → base metrics all-NaN
8. Extreme/non-extreme metrics use `nanmean` throughout so they gracefully skipped the NaN

**Fix:** Added a temporal-fill fallback at the end of `idw_impute` in [preprocessing/impute.py](../../preprocessing/impute.py) — ffill + bfill per (station, variable) column. Any cell still NaN after spatial IDW inherits from the nearest valid hour at the same station.

Safe because: spatial IDW still runs first, temporal fill only touches cells IDW couldn't handle (0.06% of X_test in this case), and an assertion at the end enforces the pipeline invariant "no NaN reaches the model."

**Regression tests:** `test_temporal_fill_activates_when_all_neighbours_missing`, `test_temporal_fallback_ffill_at_trailing_edge`, `test_temporal_fallback_bfill_at_leading_edge`, `test_spatial_still_preferred_over_temporal`, `test_assertion_when_entire_column_nan`.

### Defect B: ERA5-default preprocessing parameters silently applied to SAWS

**Symptom:** After Defect A's fix, the audit found that ALEXANDERBAAI had zero adjacency edges — a completely isolated node the graph model could not learn about.

**Root cause chain:**
1. My earlier smoke run used explicit CLI overrides for SAWS-specific preprocessing keys (500 km / 200 km / 10%)
2. The follow-up MTGODE smoke run via `cli.py --model mtgode --dataset saws --task regression` did NOT pass those overrides
3. cli.py's `load_config` reads only the model yaml (`mtgode.yaml`), which has no `preprocessing:` block
4. `run_pipeline` fell through to hardcoded defaults `distance_threshold_km=300.0, sigma_km=100.0, exclusion_threshold=0.05` — the ERA5 values
5. Cache hash changed → cache rebuilt with ERA5 params
6. ALEXANDERBAAI (449 km from CALVINIA WO, its nearest neighbour) falls outside the 300 km threshold → all edges zeroed → row sum = 0

**Fix:** Dataset-aware defaults inside `run_pipeline` in [preprocessing/pipeline.py](../../preprocessing/pipeline.py). When `dataset == "saws"`, defaults are 500 km / 200 km / 10%; otherwise ERA5's 300 km / 100 km / 5%. Config-file overrides still take precedence; the defaults just do the right thing when the user doesn't specify.

**Regression tests:** `TestPipelineDefaultsForSAWS::test_saws_defaults_produce_connected_graph` and `test_era5_defaults_leave_alexanderbaai_isolated`. The second is a "test that documents why" pattern — if a future refactor removes the SAWS branch, this test fails and points at the exact reason it existed.

---

## Part 3 — Full pipeline audit

Numbers from a fresh cache rebuild (2026-08-03 20:28) with the fixes above applied.

### 1. Shapes + finiteness (9/9 pass)

- `X_train` = (42076, 25, 6) float32, 0 NaN
- `X_val` = (14025, 25, 6) float32, 0 NaN
- `X_test` = (14027, 25, 6) float32, 0 NaN
- Adjacency, mean, std, thresholds: all finite

### 2. Split integrity (4/4 pass)

- Train + val + test sum to 70,128 (exact match for 2013-01-01 → 2020-12-31 hourly)
- Fractions: train 60.0%, val 20.0%, test 20.0%

### 3. Adjacency (9/9 pass)

- Shape (25, 25), non-negative, zero diagonal, row-normalised (all row sums = 1.0)
- Sparsity 41% non-zero (~256 edges out of 600 possible)
- Every node has ≥1 edge, minimum degree 3
- Best neighbour for every station is ≤ 500 km
- Distance matrix symmetric to 1e-3

### 4. Normalisation (7/7 pass)

- Per-(station, variable) stats: shape (1, 25, 6), no NaN, all std > 0.083
- Round-trip `normalise → denormalise` recovers original to 9e-5
- Per-(station, variable) mean on normalised train data ≈ 0 (max |mean| = 1.7e-3; small deviation from zero is from `clip_sigma=10` clipping the tails)
- Per-(station, variable) std on normalised train data ≈ 1 (max deviation 4e-4)

### 5. Labels — 95th percentile per (station, variable) (7/7 pass)

Positive rates on train set:

| Variable | Rate |
|---|---|
| t2m | 4.92% |
| tp | 0.18% |
| rh | 4.34% |
| wd | 4.88% |
| ws | 4.80% |
| sp | 4.91% |

- All non-tp variables in [3, 7]% as expected for 95th percentile
- **tp wet-hour rule fired at all 25/25 stations** — the naive 95th percentile of hourly rain sits near zero at every station (most hours are dry), so the ETCCDI-style wet-hour override kicks in and defines "extreme rain" as 95th percentile of wet hours only (≥ 0.1 mm/h). Minimum tp threshold: 2.2 mm/h. Sensible.
- Positive rate for tp drops to 0.18% because wet hours themselves are rare — the extreme threshold is now high enough to be truly extreme, not just "any measurable rain."

### 6. Wind direction handling (1/1 pass)

- Denormalised wd values sit in [-10, 370]° for 100% of cells — z-score inversion recovers raw degrees correctly
- **Design note (not a defect):** wd is z-scored as raw degrees, which is periodicity-unaware (a wind heading of 359° and 1° normalize very differently despite being 2° apart). Documented in [experiments/configs/saws_lekuba.yaml](../../experiments/configs/saws_lekuba.yaml). If wind direction becomes a Phase-2 target, the sin/cos decomposition can be added later.

### 7. Sliding-window dataloader (5/5 pass)

- Sample counts match `T - input_len - horizon + 1` exactly (42053 / 14002 / 14004 for train/val/test)
- Batch shape (32, 12, 25, 6) for both x and y

### 8. Geographic sanity of the 25-node network (1/1 pass)

Pairwise distance summary (km):

| Percentile | Distance |
|---|---|
| min | 60 km (Greytown ↔ Mooi River, KZN interior) |
| p25 | 353 km |
| median | 559 km |
| p75 | 804 km |
| max | 1422 km (Alexanderbaai ↔ King Shaka Intl, N. Cape coast to KZN coast) |

Full 5-province spread confirmed. Distance-to-nearest-neighbour per station documented in the audit output — Alexanderbaai (449 km) is the largest, everyone else is under 200 km.

### 9. Cross-model adapter compatibility (3/3 pass)

MTGODE, AGCRN, and CLCRN adapters all instantiate cleanly. None have hardcoded ERA5 dimensions. Full training smoke run for MTGODE completed at `outputs/mtgode_saws_regression_20260803_200347/`.

---

## Part 4 — Known design choices (documented, not defects)

These are not audit failures; they are places where the pipeline makes an opinionated choice that should be visible to anyone reading the code.

1. **Wind direction is z-scored as raw degrees**, not decomposed to sin/cos. Periodicity is technically wrong (359° and 1° are 2° apart in reality, ~360σ apart after z-scoring) but the model can learn around it. If WD becomes a headline target, revisit.
2. **`tp` wet-hour rule** overrides the naive 95th percentile at all 25 stations. This is the ETCCDI convention (already documented in [preprocessing/labels.py](../../preprocessing/labels.py)) — hourly rain is dry-hour dominated, so the naive percentile would sit at zero. Rule fires 25/25, so every station gets a real extreme-rain threshold.
3. **PRINS ALBERT – SWARTRIVIER** coordinates are set to the specific Swartrivier Farm GPS (−33.1731, 22.0302), not the Meteostat WMO 68823 point (−33.1833, 22.0167, 1.5 km SW). Documented in `station_coordinates.csv`. If it matters for adjacency (it doesn't at 500 km threshold — either point puts the same neighbours in scope), we can revisit.
4. **REDELINGSHUYS-AWS** has no WMO SYNOP index; coordinates are the village centroid (~1–5 km positional uncertainty). Well under the 10-km systematic-misassignment threshold.
5. **60/20/20 chronological split** rather than Davidson's 27-split walk-forward. Simpler; walk-forward can be added as `split_mode: walk_forward` if Moodley wants direct parity.

---

## Part 5 — What's ready

- 25-station × 6-feature × 8-year (2013–2020) tensor cached at `preprocessing/cache/saws/`
- Adjacency matrix (25×25, sparse, connected)
- Parser + coordinates + pipeline defaults all working
- 33 regression tests in place covering the fixes
- 46-check audit script `data/saws/audit/pipeline_audit.py` runnable anytime the cache is rebuilt

Ready for multi-seed production runs:

```
python cli.py --model mtgode --dataset saws --task regression --runs 10 --config.training.epochs 100
python cli.py --model agcrn  --dataset saws --task regression --runs 10 --config.training.epochs 100
python cli.py --model clcrn  --dataset saws --task regression --runs 10 --config.training.epochs 100
```

No CLI overrides needed for SAWS-specific preprocessing — the dataset-aware defaults now handle it.
