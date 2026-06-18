# WeatherFLOW — Project Overview & Progress

**University of Cape Town — CSC4019Z Honours Project 2026**  
**Team:** Génevieve Chikwanha, Zoleka Fuyana, Praise Jaravani  
**Supervisor:** Dr. Deshen Moodley  
**Last updated:** 18 June 2026

---

## What

Unified benchmarking framework comparing 8 STGNN architectures on South African weather data, producing results directly comparable to Gaibie (2024) for the four target variables: temperature, humidity, wind speed, and surface pressure.

**Models by family:**

| Family | Models | Owner | Adapter status |
|---|---|---|---|
| Baselines | AGCRN | Shared | Done — tested |
| Baselines | CLCRN | Shared | Done — needs `torch_scatter` (HPC only) |
| Continuous-time | MTGODE | Praise | Done — tested |
| Continuous-time | ClimODE | Praise | Done — tested |
| Lightweight | STLGRU | Zoleka | Stub |
| Lightweight | Lite-STGNN | Zoleka | Stub |
| Diffusion | DiffSTG | Génevieve | Stub |
| Diffusion | USTD | Génevieve | Stub |

**Workstreams:**
- Praise → continuous-time models + shared framework (preprocessing, training, evaluation)
- Zoleka → lightweight models (STLGRU, Lite-STGNN)
- Génevieve → diffusion models (DiffSTG, USTD)

---

## Datasets

| Dataset | Status | Notes |
|---|---|---|
| ERA5 | 67 / 317 months on disk (2000-01, 2010-01→2015-06); download running | Download resumes with `python scripts/download_era5.py`; see `specs/status_and_roadmap.md` for full tracker |
| SAWS | Not yet received | `saws_parser.py` is a stub; both parse functions raise `NotImplementedError` |

### ERA5 variables (d = 8 after derivation)

The preprocessing pipeline first parses 6 raw ERA5 variables, then appends 2 derived variables before any normalisation or splitting occurs.

| # | Name | Description | Source |
|---|---|---|---|
| 0 | `t2m` | 2m temperature [K] | ERA5 raw |
| 1 | `d2m` | 2m dewpoint temperature [K] | ERA5 raw |
| 2 | `tp` | Total precipitation [m/hr] | ERA5 raw |
| 3 | `u10` | 10m U-component of wind [m/s] | ERA5 raw |
| 4 | `v10` | 10m V-component of wind [m/s] | ERA5 raw |
| 5 | `sp` | Surface pressure [Pa] | ERA5 raw |
| 6 | `ws10` | 10m wind speed = √(u10²+v10²) [m/s] | Derived |
| 7 | `rh` | Relative humidity [%] via Magnus formula from t2m+d2m | Derived |

**Why derived variables?** Gaibie (2024) predicted wind speed and relative humidity directly. The raw ERA5 variables that drive both of those (U/V components, temperature, dewpoint) are available; the derivation step makes our feature set directly comparable and adds physically meaningful signals without discarding the directional wind information already encoded in u10/v10.

### Current cache state (as of 18 June 2026)

Cache was intentionally deleted and invalidated as part of the Phase 1 data pipeline fix (B9 forged hash). It will be rebuilt automatically on the next `cli.py` run once all 317 months are on disk. The rebuilt cache will cover 2000-01 → ~2026-05 (≈228,528 timesteps, stride=2 → 999 nodes, d=8).

Cache auto-invalidates when new months are added to `data/era5/raw/` (manifest-based hashing) or when preprocessing config changes.

---

## Execution environments

| Environment | Used for |
|---|---|
| Local (WSL2, conda `weatherflow`) | Development, preprocessing, debugging (CUDA available) |
| UCT HPC + Singularity container | All full training runs (A100 GPU); `torch_scatter` available here |

---

## Shared framework — implementation status

### Preprocessing (`preprocessing/`)

| Module | Status | Notes |
|---|---|---|
| `era5_parser.py` | Done | Grid-native (stride=2 → 999 nodes) and station-interpolated modes |
| `derive.py` | Done | Appends ws10 and rh after parsing, before imputation |
| `impute.py` | Done | IDW missing-value fill; ERA5 has no NaN so typically skipped |
| `split.py` | Done | Chronological 60/20/20; no leakage |
| `labels.py` | Done | 95th-percentile extreme-event labels from train only |
| `normalise.py` | Done | Z-score from train only; `denormalise()` available for metrics |
| `graph.py` | Done | Haversine + Gaussian kernel + row normalisation |
| `dataset.py` | Done | Sliding-window `STGNNDataset`; regression and classification modes |
| `pipeline.py` | Done | Orchestrates all 6 steps; SHA-256 hash-based cache invalidation |
| `saws_parser.py` | Stub | Both functions raise `NotImplementedError`; blocked on data receipt |

### Training (`training/`)

| Module | Status | Notes |
|---|---|---|
| `seed.py` | Done | Seeds Python, NumPy, PyTorch CPU+CUDA, cuDNN determinism |
| `losses.py` | Done | MSE (regression), weighted BCE (classification) |
| `scheduler.py` | Done | CosineAnnealingWarmRestarts |
| `trainer.py` | Done | Full train/val loop, early stopping, checkpoint saving, test evaluation |
| `plotting.py` | Done — matplotlib absent locally | Loss curves, metric plots; runs on HPC |

### Evaluation (`evaluation/`)

| Module | Status | Notes |
|---|---|---|
| `regression.py` | Done | MAE, RMSE, MSE, MAPE, NRMSE per horizon, VPT, stratified MAE |
| `classification.py` | Done | Precision, Recall, F1, AUC-ROC |
| `computational.py` | Done | Parameter count, FLOPs (torchinfo), inference time |
| `stats.py` | Done but unwired | Wilcoxon signed-rank tests implemented; not yet called from `compare_results.py` |
| `classification.py` | Done | Full suite |

### CLI & orchestration

| File | Status | Notes |
|---|---|---|
| `cli.py` | Done | Single entry point; `--max-nodes` flag for smoke tests |
| `compare_results.py` | Partial | Cross-model comparison exists; Wilcoxon calls not wired |

---

## Model adapters — implementation detail

All implemented adapters follow `BaseAdapter` (`models/base_adapter.py`): `build_model()`, `forward()`, `predict()`, `get_param_count()`.

### AGCRN (`models/baselines/`)
Adaptive Graph Convolutional Recurrent Network. Learns node embeddings that parameterise the graph convolution weights; no predefined adjacency needed at inference.  
Status: **fully implemented and smoke-tested.**

### CLCRN (`models/baselines/`)
Conditional Local Convolution Recurrent Network. Uses graph-based local convolutions conditioned on node neighbourhood; requires `torch_scatter` (PyG extension).  
Status: **fully implemented**; unavailable locally — runs correctly in the Singularity container on HPC.

### MTGODE (`models/continuous_time/`)
Multi-scale Temporal Graph ODE. Two nested ODE integrations: outer temporal ODE over the full sequence, inner graph ODE per STBlock step.  
Key wrapper decisions:
- Input permuted `(B,T,N,d) → (B,d,N,T)` to match upstream convention.
- `out_dim = H × d`; output reshaped `(B,H×d,N,1) → (B,H,N,d)`.
- `buildA_true=True` — learns its own adjacency; static adj not used.
- `self.net.idx` and `self.net.gc.device` synced manually in `forward()` to avoid CUDA device mismatch.
- Upstream `model.py` loaded under namespaced key `_weatherflow_mtgode_upstream_model` to avoid `sys.modules` collision with AGCRN's bare `model` entry.

Status: **fully implemented and smoke-tested.**

### ClimODE (`models/continuous_time/`)
Physics-inspired graph ODE: `dX/dt = −gate(X)·(A·X) + emission(X)`. GRU encoder produces initial state; `torchdiffeq` integrates forward. Tighter ODE tolerances in eval mode vs train mode.  
Status: **fully implemented and smoke-tested.**

### STLGRU, Lite-STGNN, DiffSTG, USTD
Stubs — adapters are importable, `build_model()` raises `NotImplementedError`. Pending teammate implementation.

---

## Test suite

| Test file | Coverage | Count |
|---|---|---|
| `test_derive.py` | `derive_era5_variables`: shape, naming, ws10≥0, RH∈[0,100], spot-check formulas, dtype, missing-var error | 9 |
| `test_pipeline_integration.py` | Every stage (impute/split/label/normalise/dataset) with d=8; full `run_pipeline` with mocked parser; classification path; cache hash invalidation | 41 |
| `test_preprocessing.py` | Split ordering, label leakage safety, normalise stats, graph utils, DataLoader shapes | 24 |
| `test_evaluation.py` | Regression and classification metric correctness | ~20 |
| `test_adapters.py` | AGCRN/MTGODE/ClimODE: shape, dtype, NaN, gradients, single training step, reproducibility; CLCRN skipped; stub import checks | 28 slow + 7 fast |

**Current totals:** 99 pass / 3 skip (CLCRN, no torch_scatter) — no failures.  
Slow tests (ODE adapters) run with `pytest -m slow`.

---

## Smoke test results — 18 June 2026

3 epochs, 10 nodes, batch size 512, 1 seed, ERA5 regression, CUDA.

| Model | Epoch 1 train | Epoch 2 train | Epoch 3 train | Val (ep 3) | MAE | RMSE | Params |
|---|---|---|---|---|---|---|---|
| AGCRN | 0.487 | 0.288 | 0.242 | 0.229 | 0.318 | 0.486 | 0.78M |
| MTGODE | 0.646 | 0.382 | 0.335 | 0.287 | 0.360 | 0.545 | 0.06M |
| ClimODE | 0.800 | 0.613 | 0.558 | 0.520 | 0.516 | 0.724 | 0.02M |
| CLCRN | — | — | — | — | — | — | (HPC only) |

All losses finite and monotonically decreasing. No NaN anywhere. All models ran on CUDA and correctly used d=8 features.

**Interpretation:**
- AGCRN converges fastest — expected for a straightforward GRU-based model with no ODE overhead on a small 10-node toy problem.
- MTGODE converges well and is approaching AGCRN's MAE after only 3 epochs despite being a much smaller model (0.06M vs 0.78M).
- ClimODE's slower convergence is normal — the physics-constrained ODE formulation needs more epochs. Its loss is still clearly declining and should approach the others by epoch 15–20.

---

## Note on MAPE

**MAPE is unreliable on z-scored data and should not be used as a primary metric.**

MAPE is defined as `mean(|ŷ − y| / (|y| + ε))`. After z-score normalisation, the target `y` is centred at 0, so roughly 8% of values have `|y| < 0.1` and ~0.8% have `|y| < 0.01`. With `ε = 1e-8`, any of these near-zero targets cause the ratio to blow up — a target with `|y| = 0.001` and a typical error of 0.4 contributes a MAPE of 400 for that single value. The expected value of `1/|y|` is infinite for any continuous distribution centred at 0, so aggregate MAPE is dominated by this tail regardless of how good the predictions are.

**Practical consequence:** in the current smoke test, MAPE values across models range from ~45 to ~70 for normalised predictions, driven almost entirely by `sp` (surface pressure) and `v10` (northward wind) z-scores near 0. These numbers do not reflect predictive quality.

**Fix before final experiments:** compute MAPE on denormalised predictions (in physical units: Kelvin, m/s, Pa, %). The `denormalise()` function in `preprocessing/normalise.py` is already implemented for this purpose. Alternatively, raise `ε` from `1e-8` to `0.1` (about one-tenth of a z-score standard deviation) in `evaluation/regression.py:43` to cap the per-sample ratio at a sensible maximum.

**Primary metrics for the paper:** MAE and RMSE (both in normalised units for training, denormalised for reported results). NRMSE per horizon and VPT are used for temporal robustness analysis.

---

## Outstanding items before full experimental runs

| Item | Priority | Owner | Notes |
|---|---|---|---|
| Finish ERA5 download (2000–2026, 250 remaining) | High | Praise | `python scripts/download_era5.py` resumes automatically; don't run full experiments until complete. See `specs/status_and_roadmap.md` for full post-download checklist. |
| Per-variable denormalised metrics | High | Praise | Add `compute_per_variable` to `evaluation/regression.py`; pass `mean`/`std` into `Trainer`; call `denormalise()` before metric computation |
| Fix MAPE — switch to sMAPE or drop | High | Praise | `eps=1e-8` in `regression.py:43` causes blowup; denormalising won't fix `tp`/calm-ws10; sMAPE or removal is safer |
| STLGRU adapter | High | Zoleka | Stub ready; submodule at `models/lightweight/STLGRU/` |
| Lite-STGNN adapter | High | Zoleka | Stub ready; submodule at `models/lightweight/Lite-STGNN/` |
| DiffSTG adapter | High | Génevieve | Stub ready; submodule at `models/diffusion/DiffSTG/` |
| USTD adapter | High | Génevieve | Stub ready; submodule at `models/diffusion/USTD/` |
| ClimODE audit vs Verma et al. | Medium | Praise | Verify emission function, gate, GRU encoder shapes against paper before reporting results |
| Full-node timing run on HPC | Medium | Praise | Run MTGODE and ClimODE at 999 nodes before committing to the full experimental matrix |
| Wire Wilcoxon tests | Medium | Praise | `evaluation/stats.py` done; `compare_results.py` not calling it |
| `tp` handling | Medium | Praise | Log-transform before normalisation, or exclude from regression metrics (Gaibie did not benchmark `tp`) |
| SAWS parser | Low | TBD | Blocked on data receipt from supervisor |
| Domain-aware anomaly detection | Low | TBD | Proposal requirement; current ±10σ clip is a numerical workaround only |
| Cross-model plots in `compare_results.py` | Low | TBD | Bar/box charts across models not yet implemented |
| matplotlib locally | Low | Praise | `pip install matplotlib` in the `weatherflow` conda env |

---

## Code audit — responses to reviewer questions (18 June 2026)

All answers below are code-verified against the current codebase.

### Are metrics reported per variable or pooled?

**Status: Pooled — fix required before reporting final results.**

`evaluation/regression.py` reduces prediction arrays of shape `(T, N, d)` via `axis=(0, 2)`, collapsing both the time dimension and the variable/channel dimension simultaneously into a single scalar. The MAE and RMSE values in the smoke-test table above are averages over all 8 channels equally — temperature, humidity, pressure, precipitation, and wind components are weighted identically.

For the paper, per-variable denormalised metrics are needed for the four Gaibie target variables: temperature (`t2m`, Kelvin), relative humidity (`rh`, %), wind speed (`ws10`, m/s), and surface pressure (`sp`, Pa). A `compute_per_variable` path must be added to `evaluation/regression.py` that omits the axis-2 reduction, and `training/trainer.py` must call it with `mean`/`std` for denormalisation.

### Are metrics computed in physical units (denormalised)?

**Status: Not yet — the full evaluation path uses z-scored values.**

`training/trainer.py:_evaluate_test()` calls `adapter.predict()` and passes the raw numpy arrays directly to `regression.compute_all()`. There is no call to `preprocessing.normalise.denormalise()` anywhere in the training or evaluation code. `Trainer` receives no `mean` or `std` parameters.

All published MAE/RMSE numbers are in z-score units (approximately: 1 unit ≈ 1 standard deviation of the training distribution). They are internally consistent for model comparison but cannot be meaningfully compared against Gaibie (2024) or other works that report in physical units.

**Fix:** pass `mean` and `std` from `run_pipeline()` output into `Trainer`; call `denormalise()` on predictions and targets inside `_evaluate_test()` before computing reported metrics.

### Is MAPE reliable?

**Status: Not yet fixed — MAPE is unreliable and should not be used as a primary metric.** (See "Note on MAPE" section above for the full diagnosis.)

`evaluation/regression.py:43` still uses `eps=1e-8`. Denormalising before computing MAPE would resolve the blowup for temperature and pressure but not for `tp` (precipitation) or calm-wind `ws10` values, which are near zero even in physical units.

**Recommended fix:** switch to sMAPE or drop MAPE entirely; primary metrics remain MAE and RMSE.

### Is the seed loop wired?

**Status: Confirmed wired in `cli.py`.**

`cli.py` builds `seeds = list(range(42, 42 + args.runs))` — for `--runs 10` this is seeds 42–51. `seed_everything(seed)` is called before each `Trainer` instantiation, and a fresh `Trainer` (and thus a fresh model via `build_model()`) is created for each seed. Outputs go to per-seed subdirectories under the run output folder.

### Is the horizon set correctly, and is VPT meaningful?

**Status: Confirmed correct. Horizon = 12 in all implemented model configs.**

`experiments/configs/agcrn.yaml` and `experiments/configs/mtgode.yaml` both set `horizon: 12` (12-step-ahead prediction). VPT is computed in `evaluation/regression.py` via `nrmse_per_horizon()` (returns a length-12 array) followed by `valid_prediction_time()`, which returns the last horizon step where NRMSE < 1.0. VPT is included in the `compute_all` return dict and logged by `Trainer`. A multi-step horizon means VPT is a meaningful diagnostic.

### CLCRN name

**Status: Corrected in this spec.**

CLCRN stands for **Conditional Local Convolution Recurrent Network** — not "Cluster-based CRNN" as previously written. The model applies local graph convolutions conditioned on neighbourhood structure, not cluster assignments. The adapter implementation and test descriptions used the wrong expansion; this document now uses the correct name.

### ClimODE — fidelity to Verma et al. (2024)

**Status: Implemented as described in the adapter doc-string; independent audit against the paper is outstanding.**

The wrapper (`models/continuous_time/climode_model.py`) implements the advection-diffusion ODE `dX/dt = −gate(X)·(A·X) + emission(X)` with a GRU encoder for the initial hidden state and `torchdiffeq` for integration. ODE tolerances are relaxed in train mode (`rtol=1e-3, atol=1e-4`) and tightened in eval mode (`rtol=1e-5, atol=1e-6`) to speed up training without degrading test accuracy. The upstream ClimODE submodule under `models/continuous_time/ClimODE/` is the original unmodified repository; only the adapter wrapper is custom.

A line-by-line audit against the Verma et al. paper to verify the emission function architecture, gate function, and GRU encoder shape conventions is outstanding and should be completed before final results are reported.

### Total precipitation (`tp`) — zero-inflation and MAPE

**Status: Known issue; no domain-specific handling implemented yet.**

ERA5 `tp` is zero for most dry hours and has a highly right-skewed distribution with a heavy tail at precipitation events. After z-score normalisation, the near-zero mass remains and causes MAPE blowup for `tp` even after denormalisation. The 95th-percentile extreme-event labels are physically meaningful for `tp` (they capture heavy rainfall), but the z-score normalisation compresses the scale in ways that make MAE comparisons with physical-unit benchmarks difficult.

**Recommended fix before reporting `tp` results:** log-transform before normalisation (`log(tp + ε)` with ε ≈ 1e-4 m/hr), or treat `tp` as a classification-only target and exclude it from regression metrics. This is low-priority because Gaibie (2024) did not benchmark precipitation prediction.

### Where is the ±10σ clip applied?

**Status: Applied after normalisation, before dataset windowing.**

The `±10σ` clip is applied in `preprocessing/pipeline.py` after `normalise_all_splits()` returns. It clips z-scored values to `[−10, 10]` on all three splits. This affects < 0.01% of values in practice and prevents ODE integrator instabilities from extreme outlier z-scores. It is currently a blunt numerical workaround; see "Outstanding items" for the domain-aware anomaly detection that should replace it.

---

## Key file locations

```
cli.py                                  ← single entry point for all experiments
experiments/configs/<model>.yaml        ← per-model hyperparameter configs
preprocessing/
  pipeline.py                           ← full pipeline with caching
  derive.py                             ← ws10 + rh derivation
  era5_parser.py                        ← NetCDF/ZIP parser; VARIABLE_NAMES list
  normalise.py                          ← z-score + denormalise()
models/
  base_adapter.py                       ← interface all adapters implement
  baselines/{agcrn,clcrn}_adapter.py
  continuous_time/{mtgode,climode}_adapter.py
  continuous_time/{mtgode,climode}_model.py   ← wrapper logic and known-issue docs
training/trainer.py                     ← train/val/test loop
evaluation/regression.py               ← MAE, RMSE, MAPE, NRMSE, VPT
scripts/download_era5.py               ← resumable ERA5 downloader
tests/                                  ← 99 passing tests
```

## Repo

`git@github.com:praise-jaravani/WeatherFLOW.git`
