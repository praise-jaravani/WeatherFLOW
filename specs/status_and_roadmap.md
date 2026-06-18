# WeatherFLOW — Status & Roadmap

**UCT CSC4019Z Honours 2026 — Praise Jaravani, Génevieve Chikwanha, Zoleka Fuyana**  
**Last updated: 18 June 2026**

---

## At a glance

| Phase | What | Status |
|---|---|---|
| Phase 0 | Framework scaffold + shared infrastructure | **Complete** |
| Phase 1 | Data pipeline fixes + 2000–2026 ERA5 dataset | **Complete (download running)** |
| Phase 2 | Classification head — fix `(B,H,N,d)` → `(B,H,N)` shape | **Not started — spec ready** |
| Phase 3 | Full experimental runs on HPC (all 8 models × 10 seeds) | **Blocked — awaiting data + teammate adapters** |

---

## ERA5 download progress

### Current state

| Metric | Value |
|---|---|
| Files on disk | 67 / 317 |
| Coverage on disk | 2000-01, then 2010-01 → 2015-06 |
| Files remaining | 250 (2000-02 → 2009-12 backfill + 2015-07 → ~2026-05 continuation) |
| Est. wall time remaining | 12–21 hours at 3–5 min/file |
| Download process | PID 39330, running in background |

### How to check progress

```bash
# Count files currently on disk
ls data/era5/raw/era5_southafrica_*.nc | wc -l

# Show first and last file on disk
ls data/era5/raw/era5_southafrica_*.nc | sort | head -1
ls data/era5/raw/era5_southafrica_*.nc | sort | tail -1

# Check if download process is still alive
ps aux | grep download_era5
```

### What the download is doing

One clean process running (PID 39330). The download queues one CDS request per month — each takes a few minutes (CDS queues the job, downloads ~28 MB). It works chronologically from 2000-01 to 2026-05.

Key properties of the download script:
- **Resumable** — already-downloaded files are skipped on existence check; re-run `python3 scripts/download_era5.py` at any time to continue from where it left off.
- **Safe** — the script writes to `.nc.tmp` and renames to `.nc` only on success. An interrupted download never leaves a corrupt file.
- **Future-month guard** — `_is_complete_month(year, month)` skips any month that ended fewer than 5 days ago, so the current partial month is never requested.
- **66 existing files will be skipped** — the 2010-01 → 2015-06 block is already on disk and won't be re-downloaded.

### Post-download checklist

After `ls data/era5/raw/ | wc -l` reads **317**:

- [ ] Run cache rebuild on HPC:
  ```bash
  python3 cli.py \
    --model agcrn --dataset era5 --task regression \
    --runs 1 --max-nodes 10 --no-cache
  ```
  The `verify_cache_coverage()` call at CLI startup will confirm all 317 months are present before training begins. Expected output: `status: ok, n_cached == n_on_disk == 317`.

- [ ] Confirm `meta.json` shows the full date range:
  ```bash
  cat preprocessing/cache/era5/meta.json | python3 -m json.tool | grep -E '"date_range"|"n_months"'
  # Expected: "date_range": ["2000-01", "2026-05"], "n_months": 317
  ```

- [ ] Run the full test suite to confirm no regressions:
  ```bash
  pytest tests/ -x -q
  # Expected: 99 pass, 3 skip (CLCRN)
  ```

- [ ] Run a 3-epoch smoke test for each implemented model at 10 nodes:
  ```bash
  for model in agcrn mtgode climode; do
    python3 cli.py --model $model --dataset era5 --task regression \
      --runs 1 --max-nodes 10 --config.training.epochs 3
  done
  ```

- [ ] Before scaling to full runs — complete Phase 2 (classification head) so both tasks are runnable in one batch.

---

## Phase 1: Data pipeline — COMPLETE

All five issues identified in the reviewer audit have been addressed.

### What was fixed

| Issue | Fix | File |
|---|---|---|
| B9: forged cache hash left 5-month test-set gap | Deleted forged hash + stale cache; implemented manifest-based hashing | `preprocessing/pipeline.py` |
| mtime-based invalidation (fragile to `touch`, `rsync`) | `_file_fingerprints` → `_manifest_fingerprint` using filename stems | `preprocessing/pipeline.py` |
| OOM parser (`xr.concat` across all months at once) | `_load_all_months` → `_parse_one_month` loop; each month opened, numpy chunk extracted, xarray freed | `preprocessing/era5_parser.py` |
| Download script only targeted 2010–2023 | Extended default range to 2000–2026 with `_is_complete_month` future-month guard | `scripts/download_era5.py` |
| CLI had no stale-cache warning | `verify_cache_coverage()` called at the start of every `cli.py` run | `cli.py`, `preprocessing/pipeline.py` |

### Key design choices made

**Manifest-based cache invalidation** — the cache key is now the sorted list of ERA5 filename stems (e.g. `era5_southafrica_2010_01`). Adding a new month changes the hash and invalidates the cache; re-touching or rsync-copying files does not. This is stable across all file system operations that don't change the set of months on disk.

**Incremental parser** — each monthly ZIP file is opened independently, the numpy chunk extracted, and the xarray Dataset freed before the next month is opened. Peak memory is bounded to ~50 MB (one month of xarray) plus the growing numpy accumulator, not the full ~300 MB+ xarray concat that caused OOM at scale.

**`isel` not `sel`** — the parser uses integer index selection throughout to avoid floating-point coordinate equality bugs across files.

**`verify_cache_coverage(dataset)`** — compares the `manifest` stored in `meta.json` against months currently on disk. Returns `status: ok | stale | no_cache` plus `missing_from_cache` and `extra_in_cache` lists. Called at every CLI run; a `stale` result prints a warning with the count and date range of missing months.

### Tests added

32 new tests in `tests/test_data_pipeline.py` covering:
- `_manifest_fingerprint`: stable after touch, changes on new file, empty dir
- `_era5_month_list`: extracts tuples, sorted, empty, ignores non-ERA5 files
- `verify_cache_coverage`: all four status cases, old cache without manifest key
- `TestPipelineMetaManifest`: manifest present after run, date_range format, n_months matches
- `TestParseOneMonth`: output shape, stride subsampling, no NaN
- `TestParseEra5Grid`: total timesteps, node count with stride, monotone timestamps, canonical var_names, no NaN, coords structure, empty dir raises
- `TestIsCompleteMonth`: year 2000, current month, next month, previous month, December wrap

2 integration tests added to `tests/test_pipeline_integration.py`:
- `test_hash_uses_stems_not_mtimes`: touch a file, verify hash unchanged
- `test_new_month_changes_hash`: add a file, verify hash changes

---

## Phase 2: Classification head — NOT STARTED

See [specs/WeatherFLOW_Classification_Head.md](WeatherFLOW_Classification_Head.md) for the full implementation spec.

### The problem

All adapters' `forward()` and `predict()` return `(B, H, N, d)` unconditionally. Classification labels are `(B, H, N)` — one binary "extreme / not-extreme" per node per horizon step. Running `--task classification` raises:

```
ValueError: Target size ([512, 12, 10]) must be the same as input size ([512, 12, 10, 8])
```

### The fix (three touch-points per adapter)

1. **`models/heads.py`** (new file) — `ClassificationHead(d)`: a `Linear(d, 1)` that maps `(B, H, N, d)` → `(B, H, N)` logits.
2. **`model.py` wrapper `__init__`** — read `task` from config, instantiate `ClassificationHead(d)` if `task == "classification"`.
3. **`model.py` wrapper `forward`** — after the final `(B, H, N, d)` reshape, branch: if classification, pass through the head and return `(B, H, N)` logits; else return `(B, H, N, d)`.
4. **`adapter.py` `build_model`** — pass `"task"` key into the config dict handed to the model wrapper.
5. **`adapter.py` `predict`** — if `task == "classification"`, apply sigmoid to convert logits → probabilities `∈ [0, 1]`.

### What's already correct (no changes needed)

- `training/losses.py` already uses `nn.BCEWithLogitsLoss(pos_weight=pos_weight)` — not `BCELoss`. Feed logits directly from `forward()`; do not sigmoid before the loss.
- `pos_weight` is computed from actual train-label class balance, not hard-coded.
- `preprocessing/dataset.py:59` already defines the label as `max(dim=-1)` across the d=8 variable axis — a node is "extreme" if any of its 8 variables exceeds its 95th-percentile threshold. This design choice is settled.

### Phase 2 implementation checklist

- [ ] Create `models/heads.py` with `ClassificationHead`
- [ ] Wire into `models/continuous_time/climode_model.py` + `climode_adapter.py`
- [ ] Wire into `models/continuous_time/mtgode_model.py` + `mtgode_adapter.py`
- [ ] Wire into `models/baselines/agcrn_model.py` + `agcrn_adapter.py` (baseline sanity check)
- [ ] Wire into `models/baselines/clcrn_model.py` + `clcrn_adapter.py`
- [ ] Add tests to `tests/test_adapters.py`: classification shape, predict ∈ [0,1], regression unchanged, loss runs without error
- [ ] Local smoke test (CPU, 3 epochs, 10 nodes):
  ```bash
  python3 cli.py --model mtgode --dataset era5 --task classification \
    --runs 1 --max-nodes 10 --config.training.epochs 3
  ```
- [ ] No `ValueError` on shape mismatch
- [ ] Classification `summary.json` carries precision / recall / F1 / AUC-ROC
- [ ] Same smoke test passes for ClimODE and AGCRN
- [ ] Tell teammates the three-touch-point pattern so STLGRU / Lite-STGNN / DiffSTG / USTD are implemented consistently

---

## Phase 3: Full experimental runs — BLOCKED

### Pre-conditions (all must be green before running full experiments)

- [ ] ERA5: all 317 months on disk and cache rebuilt
- [ ] Phase 2 classification head wired and smoke-tested
- [ ] STLGRU adapter (Zoleka)
- [ ] Lite-STGNN adapter (Zoleka)
- [ ] DiffSTG adapter (Génevieve)
- [ ] USTD adapter (Génevieve)
- [ ] Per-variable denormalised metrics wired into `training/trainer.py` (see below)
- [ ] MAPE replaced with sMAPE or dropped (see below)
- [ ] ClimODE audit vs Verma et al. completed

### What to run on HPC

```bash
# Full regression matrix — all 8 models × 10 seeds
for model in agcrn clcrn mtgode climode stlgru litestgnn diffstg ustd; do
  python3 cli.py --model $model --dataset era5 --task regression --runs 10
done

# Full classification matrix
for model in agcrn clcrn mtgode climode stlgru litestgnn diffstg ustd; do
  python3 cli.py --model $model --dataset era5 --task classification --runs 10
done

# Cross-model comparison
python3 compare_results.py --results-dir results/ --task regression
python3 compare_results.py --results-dir results/ --task classification
```

### Full-node timing check (before committing to full matrix)

Run MTGODE and ClimODE at 999 nodes on HPC before the full experimental matrix to confirm they don't OOM or run too slowly:

```bash
python3 cli.py --model mtgode --dataset era5 --task regression \
  --runs 1 --config.training.epochs 1
python3 cli.py --model climode --dataset era5 --task regression \
  --runs 1 --config.training.epochs 1
```

---

## Outstanding items

### High priority — needed before any HPC experimental runs

- [ ] **ERA5 download complete** — 250 files remaining; `python3 scripts/download_era5.py` resumes automatically
- [ ] **Phase 2: classification head** — shape mismatch unblocks `--task classification` for all models
- [ ] **Per-variable denormalised metrics** — add `compute_per_variable` to `evaluation/regression.py`; pass `mean`/`std` into `Trainer`; call `denormalise()` before reporting. Without this, all published MAE/RMSE values are in z-score units and cannot be compared against Gaibie (2024).
- [ ] **MAPE fix** — `eps=1e-8` in `evaluation/regression.py:43` causes blowup on z-scored data. Switch to sMAPE or drop MAPE; primary metrics remain MAE and RMSE.
- [ ] **STLGRU adapter** (Zoleka) — stub at `models/lightweight/STLGRU/`
- [ ] **Lite-STGNN adapter** (Zoleka) — stub at `models/lightweight/Lite-STGNN/`
- [ ] **DiffSTG adapter** (Génevieve) — stub at `models/diffusion/DiffSTG/`
- [ ] **USTD adapter** (Génevieve) — stub at `models/diffusion/USTD/`

### Medium priority — needed before reporting final results

- [ ] **ClimODE audit vs Verma et al. (2024)** — line-by-line check of emission function architecture, gate function, and GRU encoder shape conventions against the paper
- [ ] **Wire Wilcoxon signed-rank tests** — `evaluation/stats.py` is implemented; `compare_results.py` is not calling it
- [ ] **Full-node timing run on HPC** — run MTGODE and ClimODE at 999 nodes (1 epoch) before committing to the full matrix
- [ ] **`tp` handling** — log-transform before normalisation (`log(tp + ε)`) or exclude from regression metrics; Gaibie did not benchmark precipitation
- [ ] **Cross-model plots in `compare_results.py`** — bar/box charts across models not yet implemented

### Low priority

- [ ] **SAWS parser** — blocked on data receipt from Dr. Moodley; `saws_parser.py` is a stub
- [ ] **Domain-aware anomaly detection** — current ±10σ clip is a numerical workaround; proper extreme-event handling is a proposal requirement
- [ ] **matplotlib locally** — `pip install matplotlib` in the `weatherflow` conda env
- [ ] **CLCRN name audit** — confirm all doc/comment references say "Conditional Local Convolution Recurrent Network", not "Cluster-based CRNN"

---

## Current test suite

| File | Count | What it covers |
|---|---|---|
| `test_data_pipeline.py` | 32 | manifest fingerprinting, month list parsing, cache coverage checks, incremental parser |
| `test_pipeline_integration.py` | 43 | full `run_pipeline` with mocked parser, all 6 stages (d=8), classification path, hash invalidation (incl. mtime stability) |
| `test_preprocessing.py` | 24 | split ordering, label leakage, normalise stats, graph utils, DataLoader shapes |
| `test_evaluation.py` | ~20 | regression and classification metric correctness |
| `test_adapters.py` | 35 | AGCRN/MTGODE/ClimODE shape/dtype/NaN/grad/step/reproducibility; CLCRN skipped; stub imports |

**Total: 126 pass / 3 skip (CLCRN — no `torch_scatter` locally)**

---

## Smoke test results (18 June 2026, 10 nodes, 3 epochs, regression)

| Model | Val loss (ep 3) | MAE | RMSE | Params | Notes |
|---|---|---|---|---|---|
| AGCRN | 0.229 | 0.318 | 0.486 | 0.78 M | Converges fastest |
| MTGODE | 0.287 | 0.360 | 0.545 | 0.06 M | Approaching AGCRN at 0.06M params |
| ClimODE | 0.520 | 0.516 | 0.724 | 0.02 M | Physics ODE — needs more epochs; loss still declining |
| CLCRN | — | — | — | — | HPC only (`torch_scatter`) |

All values are in **z-score units** — not comparable to Gaibie (2024) until per-variable denormalised metrics are wired.

---

## Known issues with current metrics

**MAPE is unreliable.** After z-score normalisation, targets are centred at 0. The `eps=1e-8` in `evaluation/regression.py:43` means any near-zero target causes a MAPE ratio blowup. Smoke-test MAPE values of 45–70 reflect this artifact, not actual prediction quality. Fix: switch to sMAPE or remove MAPE.

**MAE/RMSE are in z-score units.** All published training metrics are normalised. Per-variable denormalised values (Kelvin, %, m/s, Pa) are needed for comparison with Gaibie (2024). `preprocessing/normalise.py:denormalise()` is already implemented; it just needs to be called from `training/trainer.py`.

**REPORT_VARS** — the four Gaibie primary targets: `t2m` (K), `rh` (%), `ws10` (m/s), `sp` (Pa).

---

## Key commands reference

```bash
# Resume ERA5 download (safe to re-run; skips existing files)
python3 scripts/download_era5.py

# Check download progress
ls data/era5/raw/era5_southafrica_*.nc | wc -l

# Run full test suite
pytest tests/ -x -q

# Run slow (ODE) tests
pytest tests/ -m slow -q

# Smoke test — 3 epochs, 10 nodes, regression
python3 cli.py --model agcrn --dataset era5 --task regression \
  --runs 1 --max-nodes 10 --config.training.epochs 3

# Force cache rebuild (after download completes)
python3 cli.py --model agcrn --dataset era5 --task regression \
  --runs 1 --max-nodes 10 --no-cache

# Classification smoke (after Phase 2 is wired)
python3 cli.py --model mtgode --dataset era5 --task classification \
  --runs 1 --max-nodes 10 --config.training.epochs 3
```
