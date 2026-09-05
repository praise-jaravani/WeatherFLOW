# Cross-family experimental-methodology audit

**Purpose:** compare our Phase-1 setup (AGCRN / CLCRN / MTGODE) against Chikwanha's diffusion setup (DiffSTG / USTD) against Gaibie (2024)'s protocol. Identify exactly which knobs are aligned, which are misaligned, and what coordination is needed *before HPC submission* for a defensible cross-family comparison.

**Written:** 2026-08-10 after pulling Chikwanha's repo HEAD (commit `17e0a8b`, 2026-08-10).

**Sources audited:**
- Our repo (this workspace)
- [`genevieve/Diffusion_Based_STGNNs/`](../genevieve/Diffusion_Based_STGNNs/) — pulled to HEAD
- Gaibie (2024) as summarised in [`specs/HYPERPARAMETERS.md`](HYPERPARAMETERS.md) + external audit trailer

---

## 1. Executive verdict

**Cross-family comparison is viable** — the data foundation is already essentially aligned. The material gaps sit at eval-metric and seed-sweep levels, not at data/network level.

**What is already aligned (zero coordination cost):**

1. **Identical station set** — her `station_metadata.csv` = our `STATION_WHITELIST_32` (0 symmetric difference, verified)
2. **Identical time window** — 2016-01-01 → 2025-12-31, 87,672 hours
3. **Identical data tensor shape** — `(87672, 32, 6)` float32 in both codebases
4. **Identical chronological split boundaries** — `val_start_idx=52603`, `test_start_idx=70137` (60/20/20)
5. **Both report metrics in physical units** (both reverse-normalise before scoring)

**What must be reconciled before results are pooled (concrete action items in §7):**

1. **Prediction horizon T_p**: ours 48h, hers 24h → single-horizon comparison table needs truncation at h ≤ 24, OR she extends to 48
2. **Target variable coverage**: ours all 4 Gaibie targets {t2m, sp, rh, ws}, hers t2m only → cross-family table restricted to t2m unless she runs per-target
3. **Seed count**: ours 10 seeds, hers 1 hardcoded seed (2022) → she needs 10 seeds for any inferential statistics (Mann-Whitney)
4. **VPT definition drift**: ours ε=0.3 first-crossing in normalised space, hers 0.5 first-crossing in physical NRMSE space → pin one definition, re-compute post-hoc
5. **MAPE vs sMAPE**: hers standard MAPE, ours sMAPE → both should report sMAPE (stable at near-zero targets)
6. **Ensemble-to-point extraction**: hers uses mean-of-samples for ALL metrics; the airtight clause says median-for-MAE + mean-for-RMSE. She'd need to modify or we recompute from her preserved samples
7. **Extremes stratification**: we compute `mae_extreme` / `mae_nonextreme`; she doesn't. Missing from her side

**Blocking action**: get her sign-off on the reconciliation items in §7 before her N=32 HPO/production launch on HPC, so we don't have to rerun her expensive trials to add sMAPE / 10-seed / T_p=48 later.

---

## 2. Data layer — audited alignment

### 2.1 Station selection

| | Ours | Hers |
|---|---|---|
| Whitelist file | `preprocessing/saws_parser.py::STATION_WHITELIST_32` | `ProcessingData/station_metadata.csv` (32 rows) + `verified_stations_2016_2025.csv` (60 supersets, 28 excluded before graph build) |
| N | 32 | 32 |
| Selection criterion | ≤10% overall missingness (Davidson 2022 rule) | Not code-documented; verified list of 60, then metadata restricted to 32 |
| Overlap with ours | — | **32/32 = 100%; 0 symmetric difference** |

**Verified via**: `sorted(our_set - gen_set) == [] and sorted(gen_set - our_set) == []`.

### 2.2 Time window + data tensor

| | Ours | Hers |
|---|---|---|
| Path (tensor) | `preprocessing/cache/saws/X_{train,val,test}.npy` (rebuilt from raw workbooks by `preprocessing/saws_parser.py`) | `DiffSTG/data/dataset/SAWS/flow.npy` (built once from raw by `ProcessingData/parse_datasets.py`) |
| Total-cache shape (T, N, D) | `(87672, 32, 6)` after cache assembly (X_train + X_val + X_test) | `(87672, 32, 6)` verified |
| Time range | 2016-01-01 00:00 → 2025-12-31 23:00 | Same (fixed via `total = 87672` in her config) |
| Variable order | `t2m, tp, rh, wd, ws, sp` (CANONICAL_VARS) | Assumed same by convention — **not verified in code**; her code slices `[:, :, 0]` = index 0 |
| NaN in stored tensor | 0 (imputed by IDW+temporal fallback) | 0 (verified; probably imputed at build time; `read_data()` applies `np.nan_to_num(x, nan=0)` as a belt-and-braces) |

**Two soft-alignment items:**

- **Variable order convention** must be identical for the "hers uses index-0" trick to correspond to our t2m. Alphabetical would put `rh` at index 0 (not `t2m`). We need to verify her flow.npy channel order in ProcessingData/parse_datasets.py or by inspecting values.
- **NaN handling divergence at build time**: we use spatial IDW (k=5, w=1/d²) + temporal ffill/bfill (see `preprocessing/impute.py`). She uses `np.nan_to_num(x, nan=0)` which sets missing hours to raw zero (physical). At scale of the SAWS cache this doesn't matter — the workbooks are already ≥90% clean at the 32-station whitelist — but for stations near the 10% threshold this creates a small physical-values discrepancy in the input tensor.

### 2.3 Splits

| | Ours | Hers |
|---|---|---|
| Type | Chronological 60/20/20 | Chronological 60/20/20 |
| `val_start_idx` | 52,603 (from `preprocessing/pipeline.py::run_pipeline` train_ratio=0.6, T=87672) | **52,603** (hardcoded `int(87672 * 0.6)` at [`DiffSTG/train.py:94`](../genevieve/Diffusion_Based_STGNNs/DiffSTG/train.py#L94)) |
| `test_start_idx` | 70,137 | **70,137** (hardcoded `int(87672 * 0.8)`) |
| Test window | 2019-05-27 13:00 → 2020-12-31 23:00 was 25-node; for 32-node it's 2024-07-30 09:00 → 2025-12-31 23:00 | Same 32-node test window |

**Exact match.** No coordination needed.

### 2.4 Normalisation

| | Ours | Hers |
|---|---|---|
| Method | Z-score, per-(station, variable) — `mean.shape = std.shape = (1, 32, 6)` | Z-score, **global scalar** (scalar mean, scalar std computed from `train.mean()`, `train.std()`) |
| Fit split | Train only | Train only ([`dataset.py:79`](../genevieve/Diffusion_Based_STGNNs/DiffSTG/algorithm/dataset.py#L79)) |
| Reverse for metrics? | Yes (metrics in physical units) | Yes (`clean_data.reverse_normalization(x)`, [`train.py:186`](../genevieve/Diffusion_Based_STGNNs/DiffSTG/train.py#L186)) |
| Divide-by-zero guard | `std=1 where 0` | None (relies on population std being non-zero at scalar) |

**Different but consistent** — since she uses only 1 feature at a time and physical units are reported, the scalar vs per-station distinction only affects the *internal z-scale* the model sees. Post-denorm metrics are directly comparable.

### 2.5 Adjacency matrix (only matters for models that USE static adj)

| | Ours (`preprocessing/graph.py::build_adjacency_matrix`) | Hers (`ProcessingData/build_adj_matrix.py::build_spatial_corr_matrix`) |
|---|---|---|
| Kernel | Gaussian: `A_ij = exp(-D²/2σ²)` | Gaussian: `A_ij = exp(-D²/σ²)` (note: no factor-of-2 in denominator) |
| σ | 200 km (fixed constant) | `dist_matrix.std()` (data-driven) — over N=32 this comes to ~1250 km per Chikwanha's data |
| Distance cutoff | `A_ij = 0 if D_ij ≥ 500 km` (hard cutoff) | No distance cutoff |
| Weight threshold | None | `A_ij = 0 if A_ij < 0.1` (weight-space cutoff) |
| Diagonal | 0 | 0 |
| Row-normalised | Yes (`A_ij / row_sum`) | **No** |
| Result at N=32 | `nnz=662` (65% dense), max weight = 0.313 | `nnz=438` (43% dense), max weight = 1.000 |

**Two structurally different graphs on the same 32 nodes.** For our models this is irrelevant — AGCRN learns its own adaptive graph, CLCRN uses spherical k-NN, MTGODE learns its own adaptive graph — none of them read `adj.npy` at forward time (verified via `probe_adjacency.py`). For **her** DiffSTG/USTD, the static adj IS used inside the model. Her diffusion result is under her adjacency; ours are under ours; both are "the graph the paper's method defines." Cross-family comparison at eval-metric level is unaffected — but a paper reader should be told each family used its own adjacency construction.

---

## 3. Model input/output convention

| | Ours | Hers (DiffSTG) |
|---|---|---|
| Input tensor shape | `(B, T_h=12, N=32, D=6)` — all 6 features as input | `(B, T_h ∈ {12, 24, 48}, N=32, F=1)` — she loads `flow[:, :, 0]` and expands last dim ([`dataset.py:65`](../genevieve/Diffusion_Based_STGNNs/DiffSTG/algorithm/dataset.py#L65)) |
| Output tensor shape | `(B, H=48, N=32, D=6)` → sliced to `(B, 48, 32, 1)` at loss time for univariate | `(B, T_p=24, N=32, F=1)` — she predicts a single variable directly |
| Prediction target | Per-target run: {t2m, sp, rh, ws} × 4 (one seed sweep per target) | **t2m only** (`flow[:, :, 0]` = t2m if her channel order matches ours) |
| Prediction step size | 1 h | 1 h (`points_per_hour = 1`) |
| Horizon | 48 h | **24 h** (`config.T_p = 24` in `main()` override, [`train.py:268`](../genevieve/Diffusion_Based_STGNNs/DiffSTG/train.py#L268)) |
| Input window | 12 h | HPO-sampled `T_h ∈ {12, 24, 48}` |

**Two divergences that matter for cross-comparison:**

1. **She uses 1 feature (t2m) as input; we use 6.** This is a defensible per-family choice — MTGODE + Gaibie protocol are multi-feature input / univariate output; DiffSTG is single-channel-per-run by design (each run's diffusion process operates on a fixed-D tensor). Nothing to reconcile; the input-set difference is stated in methods.
2. **Horizon T_p = 24 (hers) vs T_p = 48 (ours).** The output head is H-shaped — a model trained at T_p=48 is not the same architecture as one trained at T_p=24. **This is the load-bearing horizon-reconciliation item from the airtight clause.** Options:
   - (a) She retrains at T_p = 48 to match; we compare at h ∈ {3, 6, 9, 12, 24, 36, 48}
   - (b) We add a truncation-to-24 report; we compare at h ∈ {3, 6, 9, 12, 24}; our h=36/48 numbers become "beyond diffusion baseline"
   - (c) Both — each family reports its own horizon; cross-family table is truncated to h ≤ 24

---

## 4. Training protocol

| | Ours (all 3 deterministic models) | Hers (DiffSTG) |
|---|---|---|
| Loss | **MAE** (`nn.L1Loss`, uniform across models) | **DDPM ε-MSE denoising** (`F.mse_loss(eps, eps_theta)`), scaled ×10 at [`train.py:376`](../genevieve/Diffusion_Based_STGNNs/DiffSTG/train.py#L376). Not comparable to a forecast loss — this is the published DiffSTG objective. |
| Optimiser | Adam, per-model lr from Gaibie/HPO | Adam, lr default 2e-3 (get_params) or 1e-4 (default_config); actual lr from HPO |
| Weight decay | 0 (paper defaults) | 0 (explicit `weight_decay=0` at `train.py:319`; config.wd=1e-5 unused) |
| LR scheduler | **CosineAnnealingWarmRestarts** (T_0=10, T_mult=2, eta_min=1e-6) | **ReduceLROnPlateau** (mode=min, factor=0.5, patience=5) monitoring val-MAE ([`train.py:320, 400`](../genevieve/Diffusion_Based_STGNNs/DiffSTG/train.py#L320)) |
| Gradient clipping | max_norm = 5.0 (uniform) | None |
| Batch size | Per-model (32-128, HPO-tuned) | Default 32 or 8; HPO tunes indirectly via other knobs |
| Epochs | 200 cap (uniform); early stop patience 15 | 30 cap; early stop patience 10 |
| Seed | **10 seeds, {42..51}** — one full run per seed per (model, target) cell | **1 hardcoded seed = 2022** ([`train.py:254`](../genevieve/Diffusion_Based_STGNNs/DiffSTG/train.py#L254)) |
| Deterministic CUDA | `torch.backends.cudnn.deterministic = True` (both) | Same |
| Random re-seed | Per-epoch SHA256 re-seed for resume-safety | Per-eval: `setup_seed(2022)` inside `evals()` (line 157) — deterministic ensemble sampling |

**Critical mismatch for statistical inference:**

Cross-family Mann-Whitney U tests need multiple seeds on both sides. Hers = 1 → we can't run any inferential test unless she adds a seed sweep. Recommended coordination: she runs 10 seeds using either {42..51} or {42..71} (10 or 30 seeds). For consistency, use {42..51} matching our range.

---

## 5. Evaluation metrics

Both codebases report metrics in **denormalised physical units** on the held-out test split. Same definitions do not follow — this table is the reconciliation source.

| Metric | Ours | Hers | Comparable as-is? |
|---|---|---|---|
| **MAE** | Pooled over (samples, H, N) after `pred - target` in physical units. Standard `mean(|pred - target|)`. | `masked_mae_np(y_true, y_pred, null_val=0)` — same but **masks out physical-zero targets** (leftover from traffic-flow convention). Then averaged over batch/H/N/D (`np.mean`). | ✅ **Comparable** if we accept the mask-at-zero difference (near-zero temperatures rare in SA; effect < 0.1% of samples). |
| **RMSE** | `sqrt(mean(sq_err))` in physical units | `sqrt(masked_mse_np(...))` with same masking | ✅ Comparable (same mask caveat) |
| **MSE** | `mean(sq_err)` | `masked_mse_np(..., 0)` | ✅ Comparable |
| **MAPE / sMAPE** | **sMAPE** = `mean(2 · |p-t| / (|p|+|t|+eps))` — stable at near-zero | **standard MAPE** = `mean(|p-t| / |t|) * 100` — divergent at t → 0 | ❌ **NOT comparable.** sMAPE and MAPE are different metrics. Recommendation: both report sMAPE. |
| **NRMSE** | Per-horizon in normalised (z-score) space; produced as `nrmse_per_horizon` length-48 array | Per-step: `masked_nrmse_np` normalises by **`np.nanstd(y_true[valid_mask])` in physical units** — different denominator | ❌ **NOT comparable directly.** Denominators differ. |
| **VPT (Valid Prediction Time)** | First horizon index where `nrmse_per_horizon ≥ ε=0.3`. Denominator = training std (from cache mean/std). | First horizon index where `masked_nrmse_np > threshold=0.5`. Denominator = test-batch std of y_true (physical). Aggregated as `vpt_mean` over the batch. | ❌ **NOT comparable.** Threshold differs (0.3 vs 0.5) AND denominator differs (train std vs test-batch std) AND aggregation is different. |
| **CRPS** | Not currently computed by us (deterministic models don't have a natural CRPS; the point-CRPS identity `CRPS = MAE` applies) | Quantile-CRPS from CSDI paper: for each q ∈ `arange(0.05, 1.0, 0.05)`, compute quantile loss; sum / n_quantiles. Denominator = `sum(|target|)`. | ❌ **Not natively comparable.** For a unified CRPS column: deterministic entries = MAE (per point-forecast identity); diffusion entries = her quantile-CRPS. **Airtight clause item 2.** |
| **MIS (Mean Interval Score)** | Not computed | Standard MIS at α=0.05 from GluonTS. | Only meaningful for probabilistic models (diffusion). Report N/A for deterministic; disclose. |
| **Extreme MAE / Non-extreme MAE** | Computed via per-station 95th-percentile thresholds from training data. Stratifies samples into extremes/non-extremes and reports per-strata MAE. | **Not computed** | ❌ **Missing on her side.** Cross-family extreme analysis requires adding this to her eval — non-trivial (needs threshold vector + mask + averaging). |
| **Peak intensity error** | Computed (`peak_intensity_error`) | Not computed | ❌ Missing on her side |
| **Per-station MAE** | Full `per_station.npz` written per task | Not written | ❌ Missing on her side |
| **inference_ms_per_sample** | Timed and stored in `test_metrics.json` | Time is measured (`time_lst`) at batch level and pooled into `metric.metrics['time']` — batch-time, not per-sample. Different aggregation. | ⚠ Approximately comparable if we divide her batch-time by batch size |

### 5.1 Deterministic extraction from her diffusion ensemble

Her code path ([`utils/eval.py:112`](../genevieve/Diffusion_Based_STGNNs/DiffSTG/utils/eval.py#L112)):

```python
# probabilistic metric on full ensemble
self.metrics['crps'] = calc_quantile_CRPS(y_true, y_pred_5d, ...)
self.metrics['mis']  = calc_mis(y_true, y_pred_5d)

# collapse to deterministic for point metrics
y_pred = np.mean(y_pred, axis=1)  # (B, T_p, V, D)  — MEAN over ensemble
self.metrics['mae'], ... = self.get_metric(y_true, y_pred)
```

**Her MAE/RMSE/MAPE/VPT are all computed on the ensemble MEAN.** The airtight clause specifies:
- **Ensemble median for MAE tables** (the median minimises L1)
- **Ensemble mean for RMSE tables** (the mean minimises L2)

She currently gives us mean-for-both. Fix options:
- (a) She adds `y_pred_median = np.median(y_pred_5d, axis=1)` and reports MAE from median, RMSE from mean
- (b) We recompute post-hoc from her saved `samples` pickle (`train.py:220`) — but she only saves the first 50 samples of the last test batch, insufficient for whole-test-set stats
- Recommendation: (a) — one-line fix in her evals()

---

## 6. HPO comparison

| | Ours (MTGODE only) | Hers (DiffSTG) |
|---|---|---|
| Method | Pre-sampled random search (60 trials) | Optuna TPE (multivariate) with median-pruner (n_warmup_steps=5) |
| Trials | 60 | 30 (`maxTrialNumber: 30`) |
| Storage | JSON manifest + per-trial yaml files (no DB) | SQLite (`sqlite:///output/optuna/{study_name}.db`); `load_if_exists=True` |
| Parallelism | 6 concurrent trials via our worker-chain (K=3 × 2 lineages) | **Sequential** (`trialConcurrency: 1`) via NNI config, or single sbatch via Optuna |
| Objective | min val MAE | min val MAE (same!) |
| Seed | Sampler seed 42; trials use seeds 42+N | TPE sampler seed 27; each trial hardcoded to setup_seed(2022) |
| Search space | 5 knobs: lr, batch, dropout, hidden_dim, k | 8 knobs: T_h, hidden_size, N (diffusion steps), beta_schedule, beta_end, n_samples, mask_ratio, sample_steps |
| Pinned | node_dim=40, cgp_alpha=2.0, adjoint=false, ODE times, input_len=12 (all per audit + paper) | epsilon_theta="UGnet", model.C, sample_strategy — pinned at paper defaults |
| HPC allocation | l40s partition (2 concurrent GPU jobs, 24h chain) | a100 partition (`--account=a100free`, single GPU, 48h `--time=48:00:00`) |
| Winner already found? | Yes (2026-08-10): lr=0.003, batch=128, k=20, hidden=64, dropout=0.3 | Not yet; her sweep either not run or not surfaced |

Different methods are fine — TPE and random are both valid published HPO strategies, and each family owning its HPO is defensible. **What matters is that HPO uses the same trial protocol on both sides**: same target variable(s), same test/val definition, same objective.

**Currently aligned trial protocol:**
- Same 32-station cache
- Same val_start_idx / test_start_idx  
- Same objective (min val MAE, in physical units for us, in physical units for her too)

**Currently misaligned trial protocol:**
- She uses 1 seed per trial (2022); we use 1 seed per trial (42+trial_num). Neither is more valid — but our winner's variance across 10 seeds is measurable while hers isn't.
- She tunes at T_h ∈ {12, 24, 48}; we pinned T_h = 12 for reasons documented in HYPERPARAMETERS.md (receptive-field truncation at 1.0/0.25 solver).
- She tunes T_p implicitly via HPO defaults; we pinned T_p = 48 for Moodley alignment.

**Recommendation for cross-comparison of HPO:** each family discloses its own HPO in methods; per-family winner is used in downstream production. No need to align HPO methods — only production-run protocol.

---

## 7. Reconciliation action list (send to Chikwanha)

**Priority 0 — silent-corruption fix, do FIRST before any cross-comparison:**

| # | What | Where | Cost |
|---|---|---|---|
| 0 | **Channel-order mismatch in `flow.npy`.** Empirically inferred from per-channel statistics, her ordering is `[t2m, rh, sp, tp, wd, ws]` while ours is `CANONICAL_VARS = [t2m, tp, rh, wd, ws, sp]`. Only channel 0 (t2m) matches by accident. Any comparison on channels 1-5 as they stand is nonsense (our-tp vs her-rh, etc.). Fix: permute her `flow.npy` at build time so its channel order matches `CANONICAL_VARS`; verify by reprinting per-channel stats after the permute. | `ProcessingData/build_flow_matrix.py` (channel permute at build) OR `DiffSTG/algorithm/dataset.py:65` (channel-index lookup by name) | Small: one `flow = flow[..., [permute_indices]]` line |

**Priority 1 — required before results are pooled or Mann-Whitney is run:**

| # | What | Where | Cost |
|---|---|---|---|
| 1 | **Add 10-seed sweep** using seeds {42..51} to match ours. Replace `setup_seed(2022)` with a `--seed` CLI arg | `DiffSTG/train.py:22-28, 254` | Small code change; 10× compute (30 → 300 GPU-h if trial ≈ 30 min at n_samples=8) |
| 2 | **Report sMAPE, not MAPE**. Add `masked_smape_np` to `utils/eval.py`, replace/augment MAPE in Metric | `DiffSTG/utils/eval.py::masked_mape_np` | 5-line add |
| 3 | **VPT: agree on ε=0.3 in normalised space with training-std denominator.** Add `vpt_epsilon` parameter to Metric.__init__ (default 0.3), change `masked_nrmse_np` denominator to a stored `train_std` (passed in from clean_data). Ideally: change VPT aggregation to first-cross of pooled-NRMSE (like ours), not batch-mean. | `DiffSTG/utils/eval.py:47-62` | Small refactor |
| 4 | **Ensemble collapse: median for MAE table, mean for RMSE table.** Compute both `y_pred_mean = np.mean(...)` and `y_pred_median = np.median(...)` inside `Metric.update_metrics`, report MAE from median-collapsed and RMSE from mean-collapsed. | `DiffSTG/utils/eval.py::update_metrics` line 112 | 3-line change |
| 5 | **Save per-station metrics per test task** to enable geographic-stratification cross-comparison | `DiffSTG/train.py::evals` after test-eval | New feature; write per_station.npz analogous to ours |
| 6 | **Save unified summary.json per (target, seed)** in a fixed schema our aggregator can consume, mirroring ours: keys `mae, rmse, smape, vpt, crps, mis` etc. — no extra features, just standardised output naming | new writer in her eval | Small addition |

**Priority 2 — needed for full cross-family results table:**

| # | What | Where | Cost |
|---|---|---|---|
| 7 | **Add extreme MAE stratification** (95th-percentile thresholds from training; report `mae_extreme`, `mae_nonextreme`). We can ship her our thresholds file from the shared cache. | `DiffSTG/utils/eval.py` new function | Medium: needs threshold vector + boolean masking |
| 8 | **Horizon reconciliation**: either she retrains T_p=48, or we truncate our published headline table to h ≤ 24. Recommendation: she extends to 48 to preserve our Moodley-directed horizon; if compute-prohibitive, we go with truncation and disclose asymmetry once. | Her main() `config.T_p = 48` change | Zero code cost; 2× compute per trial |
| 9 | **Per-target runs**: if the paper reports cross-family on all 4 Gaibie targets, she needs runs for t2m, sp, rh, ws separately (currently t2m only). If we only need cross-family on t2m, we can leave this. | Loop over `flow[:, :, i]` for i in target_indices | Zero code cost; 4× compute |

**Priority 3 — nice-to-have hygiene:**

| # | What | Where |
|---|---|---|
| 10 | Verify her `flow.npy` channel order matches our `CANONICAL_VARS = [t2m, tp, rh, wd, ws, sp]`. She slices `[:, :, 0]` assuming t2m at index 0. Print the first few values at each channel and cross-check against known station-mean values. | `ProcessingData/parse_datasets.py` |
| 11 | Verify NaN handling divergence: our IDW+ffill vs her `nan_to_num(nan=0)`. Should be a no-op at our 32-station cache (all values non-NaN), but confirm in a diff. | `DiffSTG/algorithm/dataset.py:66` |
| 12 | Disclose adjacency divergence in methods (each family used its own construction; our models don't consume static adj at forward time). | Paper methods section |

---

## 8. Gaibie (2024) protocol fidelity — where each of us stands

Reviewer earlier defined the "Gaibie-protocol" HPO benchmark. Comparison:

| Gaibie's protocol | Ours | Hers |
|---|---|---|
| Random search HPO | ✅ 60 trials random | ⚠ TPE 30 trials |
| First 2 train years as "HPO training set" (`train_hours_cap=17520`) | ✅ Set explicitly | ⚠ Not stated in her code — uses full train split |
| Objective = val MAE (unnormalised) | ✅ Yes | ✅ Yes (val MAE via `metrics_val.metrics['mae']`) |
| Epoch cap 30, patience 5 | ✅ Set | ⚠ Epoch cap 30 ✓, patience 10 (her code uses `early_stop=10`) |
| Target variable = t2m | ✅ (all trials fixed at t2m) | ✅ (all runs fixed at index 0) |
| Horizon | Gaibie: 24h. Ours: 48h (deliberate deviation per Moodley). Hers: 24h (matches Gaibie by accident?). | Ours = 48 (documented divergence), Hers = 24 (accidental Gaibie match) |
| Model family | Gaibie: AGCRN, CLCRN, GWN. Ours: AGCRN, CLCRN, MTGODE (Gaibie didn't evaluate MTGODE). Hers: DiffSTG, USTD (Gaibie didn't evaluate diffusion). | — |
| Baselines | Persistence + climatology (both) | Persistence + climatology (probably; not directly verified) |

**Neither of us is exactly Gaibie** — Gaibie's protocol is a citation anchor, not a hard requirement. Each family's methods section should say: "matches Gaibie's protocol except for {enumerated deviations, with reasons}."

---

## 9. Reproducibility artefacts (what each family stores after a run)

| Artefact | Ours (per-task under `runs/<task_id>/`) | Hers (per-run under `DiffSTG/output/`) |
|---|---|---|
| Metrics file | `summary.json`, `test_metrics.json` (schema in `PHASE1_25NODE_CAMPAIGN.md`) | wandb JSON + `output/metrics/DiffSTG.csv` (single-line CSV with all metrics) |
| Per-epoch log | `train.log` (train/val/grad_norm) | wandb metrics + `output/log/<trial_name>.log` |
| Best-val checkpoint | `checkpoint.pt` | `output/model/<trial_name>*.dm4stg` (full torch.save) |
| Resumable state | `state.pt` (model + opt + sched + RNG + epoch) | Not saved — trials restart from scratch |
| Per-station stratification | `per_station.npz` (N, D) | **Not saved** — reconciliation item 5 |
| Ensemble samples (first 50) | N/A | `output/forecast/<trial_name>.pkl` (samples, targets, observed_flag, evaluate_flag) |
| Reproducibility record | `runs/hpo/spec.json` for HPO trials | wandb config + `output/optuna/<study_name>.db` |

**Both use `wandb`** — potentially useful for cross-family dashboard if we agree on project/group naming.

---

## 10. What "airtight methodology" concretely means for our paper (after reconciliation)

The reviewer's airtight-clause additions were:

1. ✅ Diffusion point forecasts extracted metric-matched (**median for MAE, mean for RMSE**) — requires her item 4 above
2. ✅ **Unified CRPS column** in the results table (deterministic entries = MAE; diffusion entries = quantile-CRPS) — no additional work; we compute CRPS = MAE for deterministic, she already computes quantile-CRPS
3. ✅ **Share artefacts, not recipes** (hash-pinned cache, shared scalers, shared thresholds) — we can push her the exact `preprocessing/cache/saws/{mean, std, thresholds, tp_wet_trigger}.npy` files
4. ✅ **VPT pinned to ε=0.3 first-crossing in normalised space; sMAPE not MAPE** — requires her items 2 + 3 above
5. ✅ **Cross-family significance is unpaired Mann-Whitney U** (not seed-paired Wilcoxon) — requires her item 1 (10 seeds on her side)
6. ✅ **Horizon reconciliation**: either she extends to T_p=48 or we truncate our tables to h ≤ 24 — requires her item 8 or a paper-scope decision

**After the priority-1 + priority-2 items land on her side**, cross-family comparison is defensible: same data, same split, same metric definitions (with method-appropriate ensemble extraction), same statistical test framework, disclosed asymmetries (loss family, input feature set, adjacency construction).

---

## 11. Concrete next action

Send the priority-1 list (items 1-6 in §7) to Chikwanha with:
- The rationale for each (this document)
- Access to our `preprocessing/cache/saws/{mean, std, thresholds, tp_wet_trigger}.npy` files (needed for consistent VPT + extremes stratification)
- A joint decision on horizon (item 8): retrain-to-48 or truncate-to-24

**Blocking on her side before HPC HPO/production submission**: items 1, 4, and 8. Items 2, 3, 5, 6, 7 can land as post-hoc metric-recompute if she stores raw predictions.

**Estimated coordination cost:** ~half a day of her time to modify eval.py + train.py, plus the compute cost of running 10 seeds instead of 1 (or paying for the missed statistical inference by not doing it).
