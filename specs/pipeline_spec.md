# WeatherFLOW — Shared Experimental Framework Specification

Extracted directly from the project proposal and codebase. This is the authoritative
reference for every parameter, metric, threshold, and output that the shared pipeline
produces. When the proposal and the code agree, both are recorded. When something is
still a TODO, it is flagged explicitly.

---

## 1. Problem Formulation

- **Graph:** G = (V, E) where nodes V are weather stations and edges E encode spatial relationships
- **Input tensor:** X ∈ ℝ^{T × N × d}
  - T = length of historical observation window (input_len)
  - N = |V| = number of nodes (stations / ERA5 grid points)
  - d = number of meteorological feature variables (d > 1)
- **Output:** sequence y₁, y₂, …, y_H for each node v ∈ V over forecast horizon H
- **Two tasks** sharing the same input structure (X, A):
  - **Regression** — predict continuous-valued future weather variables
  - **Classification** — predict binary extreme event labels for future timesteps

---

## 2. Datasets

### ERA5 (WeatherBench)
- Source: Copernicus Climate Data Store (CDS), accessed via `scripts/download_era5.py`
- Coverage: global reanalysis, South Africa bounding box
- Temporal range: 2010–2023 (downloading; target is full period)
- Temporal resolution: hourly
- Spatial resolution: 0.25° grid (~28 km at the equator)
- Subsampled grid at stride=2: 53 lat × 73 lon → 999 nodes at ~0.5°

### SAWS (South African Weather Service)
- Source: received directly from SAWS
- Coverage: 70 stations across all 9 provinces
- Temporal range: 2011–2025
- Temporal resolution: hourly (same as ERA5)
- Status: not yet received

---

## 3. Feature Variables (d = 6)

| Short name | Long name | ERA5 stream |
|---|---|---|
| `t2m` | 2m temperature | instant |
| `d2m` | 2m dewpoint temperature | instant |
| `u10` | 10m u-component of wind | instant |
| `v10` | 10m v-component of wind | instant |
| `sp` | surface pressure | instant |
| `tp` | total precipitation | accumulated |

- Variable order is canonical and fixed: `[t2m, d2m, tp, u10, v10, sp]`
- SAWS variables (when received): maximum temperature, minimum temperature, rainfall — additional variables TBC depending on what SAWS delivers

---

## 4. Preprocessing Parameters

### Missing value handling
- **Exclusion threshold:** stations with > 5% missing values across the full time range are excluded before imputation
- **Imputation method:** Inverse Distance Weighting (IDW) — missing values estimated from neighbouring stations, weighted by geographic distance
- ERA5 grid data has no missing values — imputation step is skipped for ERA5

### Data split
- **Ratio:** 60 / 20 / 20 (train / validation / test)
- **Method:** strictly chronological — no random shuffling at any point
- **Order of operations (critical for leakage prevention):**
  1. Parse raw data
  2. Impute missing values
  3. **Split** ← happens here, before any statistics are computed
  4. Label (thresholds computed on train only, applied to all three splits)
  5. Normalise (stats computed on train only, applied to all three splits)
  6. Build graph
  7. Build DataLoaders

### Extreme event labelling
- **Method:** 95th-percentile threshold per variable per station
- **Threshold source:** computed on the **training partition only**, then applied to val and test — using the full dataset would leak future distributional information
- **Output:** binary label matrix — 1 if observation exceeds threshold, 0 otherwise
- **Class imbalance:** ~5% positive (extreme), ~95% negative (non-extreme) by construction

### Z-score normalisation
- **Formula:** X_norm = (X − μ) / σ
- **Statistics source:** μ and σ computed on the **training partition only**
- **Scope:** per variable per node (each of the N×d combinations gets its own mean and std)
- **Applied to:** training, validation, and test sets using training-set statistics
- **Clipping:** normalised values clipped to ±10σ at DataLoader build time to prevent numerical instability from ERA5 total precipitation (tp), which reaches up to 57σ after z-score normalisation due to extreme right skew
  - Parameter: `clip_sigma = 10.0` (configurable in preprocessing config)

### ERA5-specific spatial subsampling
- **lat_stride:** 2 (default) — subsample every 2nd latitude point
- **lon_stride:** 2 (default) — subsample every 2nd longitude point
- Result: 999 nodes at ~0.5° resolution

### Graph construction
- **Method:** inverse-distance weighting with a distance threshold
- **Distance metric:** Haversine (great-circle distance from lat/lon coordinates)
- **distance_threshold_km:** 300.0 km — pairs further apart than this receive zero weight
- **sigma_km:** 100.0 km — controls the decay rate of edge weights with distance
- **Normalisation:** row-normalised (D⁻¹A) so each row sums to 1
- **Self-loops:** excluded (diagonal set to zero)
- **Result shape:** (N, N) float32 adjacency matrix, saved to cache as `adj.npy`

---

## 5. Sliding Window (DataLoader)

- **Input window length:** T = `input_len` = 12 timesteps *(⚠ TODO: confirm from Gaibie et al.)*
- **Forecast horizon:** H = `horizon` = 12 timesteps *(⚠ TODO: confirm from Gaibie et al.)*
- **Input shape per sample:** (T, N, d) = (12, N, 6)
- **Target shape per sample (regression):** (H, N, d) = (12, N, 6) — next H timesteps of X (clipped)
- **Target shape per sample (classification):** (H, N) = (12, N) — binary extreme labels for next H timesteps, max across variables (any variable extreme = node extreme)
- **Stride:** 1 timestep (each window shifts by one step)
- **Number of training windows (ERA5, N=999):** T_train − T − H + 1 ≈ 19,690 windows
- **Shuffle:** training DataLoader only; val and test are sequential

---

## 6. Training Protocol (Shared Across All Models)

### Run structure
- **Number of runs:** 10 per experiment
- **Seeds:** 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
- **Seed scope:** Python `random`, NumPy, PyTorch CPU, PyTorch CUDA — all set simultaneously via `seed_everything(seed)`
- **Aggregation:** final results = mean ± std of each metric across 10 seeds

### Optimiser
- **Type:** Adam
- **Learning rate:** model-specific (see Section 9)
- **Weight decay:** not applied (default Adam, no L2 regularisation)

### Learning rate schedule
- **Type:** CosineAnnealingWarmRestarts (all models use this)
- **T_0:** 10 (first restart cycle length in epochs)
- **T_mult:** 2 (each subsequent cycle is 2× longer)
- **eta_min:** 1e-6 (minimum LR floor)
- **Step timing:** once per epoch (after train + val pass)

### Gradient clipping
- **Method:** `torch.nn.utils.clip_grad_norm_`
- **max_norm:** model-specific (see Section 9) — default 5.0, reduced to 1.0 for memory-intensive models

### Early stopping
- **Monitor:** validation loss
- **Patience:** 15 epochs (default for all models)
- **Checkpoint:** best model weights saved as `checkpoint.pt` (lowest val loss)

### Loss functions
- **Regression (Phase 1):** Mean Squared Error (MSE) — `torch.nn.MSELoss`
- **Classification (Phase 2):** Weighted Binary Cross-Entropy (WBCE):

  L_WBCE = −(1/N) Σᵢ [ w₁ yᵢ log(ŷᵢ) + w₀ (1−yᵢ) log(1−ŷᵢ) ]

  - **w₁** (extreme class weight) and **w₀** (non-extreme class weight) computed from training label distribution at the start of each experiment
  - At 95th-percentile threshold: w₁ ≈ 19 × w₀
  - Class weights logged to MLflow alongside all other hyperparameters

---

## 7. Regression Evaluation Metrics (Phase 1)

All metrics computed on the test set using the best checkpoint (lowest val loss).

### Standard accuracy metrics

- **MAE (Mean Absolute Error)** — primary metric for regression
  ```
  MAE = (1/N) Σᵢ |yᵢ − ŷᵢ|
  ```

- **MSE (Mean Squared Error)** — training loss reference
  ```
  MSE = (1/N) Σᵢ (yᵢ − ŷᵢ)²
  ```

- **RMSE (Root Mean Squared Error)** — penalises large errors more than MAE
  ```
  RMSE = √[(1/N) Σᵢ (yᵢ − ŷᵢ)²]
  ```

- **MAPE (Mean Absolute Percentage Error)** — scale-independent error
  ```
  MAPE = (100/N) Σᵢ |yᵢ − ŷᵢ| / |yᵢ|
  ```

### Temporal robustness metrics

- **NRMSE(t_f) — Normalised RMSE at forecast step t_f** — one value per horizon step, producing a curve
  ```
  NRMSE(t_f) = √[ (1/N) Σᵢ (y_{t_f,i} − ŷ_{t_f,i})² / σᵢ² ]
  ```
  where σᵢ is the standard deviation of the ground truth at node i
  - Output: list of H values, one per forecast step (nrmse_per_horizon)

- **VPT (Valid Prediction Time)** — maximum forecast lead time at which NRMSE remains below threshold ε
  ```
  VPT = argmax_{t_f} { NRMSE(t_f) < ε }
  ```
  - **ε (vpt_epsilon):** 0.5 *(⚠ TODO: confirm from Gaibie et al.)*
  - If NRMSE exceeds ε at the very first step, VPT = 0
  - Output: integer number of timesteps

### Extreme-event stratified metrics

- **Stratified MAE (extreme)** — MAE computed only on timesteps where the ground truth exceeds the 95th-percentile threshold
- **Stratified MAE (non-extreme)** — MAE computed only on timesteps below the threshold
- **Peak Intensity Error** — MAE computed on the predicted maximum value within each extreme event window; measures how well each model captures the peak magnitude of extreme events

---

## 8. Classification Evaluation Metrics (Phase 2)

All metrics computed on the test set using the best checkpoint.

- **Accuracy** — overall correctness (note: misleading under class imbalance; reported for completeness)
  ```
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  ```

- **Precision** — fraction of predicted extremes that are true extremes
  ```
  Precision = TP / (TP + FP)
  ```

- **Recall** — fraction of true extremes that are correctly predicted (most important metric given ~5:95 class imbalance)
  ```
  Recall = TP / (TP + FN)
  ```

- **F1-Score** — harmonic mean of precision and recall; primary classification metric
  ```
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ```

- **AUC-ROC** — area under the ROC curve; threshold-independent

- **Confusion matrix** — saved per run as part of test metrics

### Statistical significance test (end of Phase 2)

- **Wilcoxon signed-rank test** — non-parametric paired test across technique families
  - Applied to F1 scores across the 10 runs per model
  - Compares each pair of technique families (e.g. continuous-time vs baselines, diffusion vs baselines)
  - Chosen over t-test because the 10 runs are not independent samples (same data, different seeds)
  - Output: (statistic, p-value) per pair

---

## 9. Computational Efficiency Metrics (Research Question 4)

Recorded for every model regardless of task.

- **Parameter count** — total trainable parameters in millions (M)
  - Formula: `Σ p.numel() for p in model.parameters() if p.requires_grad`
  - Recorded once before the run loop (deterministic given model + config)

- **FLOPs** — floating-point operations per forward pass in billions (G)
  - Measured via `torchinfo` on a single batch of standard input shape (B=1, T=12, N, d=6)
  - Recorded once before the run loop

- **Inference time** — wall-clock time in milliseconds per sample (ms/sample)
  - Averaged over the full test set, model in eval mode, `torch.no_grad()`
  - Recorded after loading the best checkpoint (reflects the trained model's behaviour)
  - Must be measured on the same hardware for all models to be comparable — final numbers from UCT HPC A100

---

## 10. Per-Model Hyperparameters

### AGCRN (baseline)
- `hidden_dim`: 64 *(⚠ TODO: confirm from Gaibie et al.)*
- `num_layers`: 2
- `embed_dim`: 10 (node embedding dimension for adaptive graph generation)
- `cheb_k`: 2 (Chebyshev polynomial order)
- `learning_rate`: 0.001 *(⚠ TODO: confirm from Gaibie et al.)*
- `batch_size`: 32
- `epochs`: 100
- `early_stopping_patience`: 15
- `gradient_clip_max_norm`: 5.0
- `scheduler.T_0`: 10 | `scheduler.T_mult`: 2 | `scheduler.eta_min`: 1e-6

### CLCRN (baseline)
- `hidden_dim`: 64 *(⚠ TODO: confirm from Gaibie et al.)*
- `num_layers`: 2
- `embed_dim`: 16
- `k_neighbors`: 25 (k-NN graph size for spherical convolution kernel)
- `lck_structure`: [4, 8] (CLConv kernel layer sizes)
- `max_view`: 2 (max spatial hops in CLConv)
- `learning_rate`: 0.001 *(⚠ TODO: confirm from Gaibie et al.)*
- `batch_size`: 32
- `epochs`: 100
- `early_stopping_patience`: 15
- `gradient_clip_max_norm`: 5.0
- `scheduler.T_0`: 10 | `scheduler.T_mult`: 2 | `scheduler.eta_min`: 1e-6

### MTGODE (continuous-time — Praise)
- `hidden_dim`: 40
- `num_layers`: 2
- `dropout`: 0.3
- `graph_constructor.k`: 10 (top-k neighbours in learned adaptive graph)
- `graph_constructor.alpha`: 3 (tanh saturation rate)
- `learning_rate`: 0.001
- `batch_size`: 32
- `epochs`: 100
- `early_stopping_patience`: 15
- `gradient_clip_max_norm`: 5.0
- ODE solver (fixed-step only — MTGODE raises ValueError for adaptive solvers):
  - `ode.method_1`: euler (outer temporal ODE)
  - `ode.method_2`: euler (inner graph ODE inside CGP layers)
  - `ode.adjoint`: true (adjoint method for memory-efficient backprop)
  - `ode.time_1`: 1.2 | `ode.step_size_1`: 0.4
  - `ode.time_2`: 1.0 | `ode.step_size_2`: 0.25
  - `ode.rtol`: 1e-4 | `ode.atol`: 1e-3

### ClimODE (continuous-time — Praise)
- `hidden_dim`: 64
- `velocity_channels`: 5 (velocity field parameterisation)
- `learning_rate`: 5e-4 (lower than baselines — ClimODE is memory-intensive)
- `batch_size`: 16
- `epochs`: 100
- `early_stopping_patience`: 15
- `gradient_clip_max_norm`: 1.0 (tighter clip for stability)
- ODE solver (adaptive — uses dopri5):
  - `ode.method`: dopri5
  - `ode.adjoint`: true
  - Training: `ode.train_rtol`: 1e-3 | `ode.train_atol`: 1e-4 (loose — faster)
  - Evaluation: `ode.eval_rtol`: 1e-5 | `ode.eval_atol`: 1e-6 (tight — more accurate)

### STLGRU (lightweight — Zoleka)
- `hidden_dim`: 32 (smaller than baselines by design)
- `num_layers`: 1
- `dropout`: 0.1
- `learning_rate`: 0.001
- `batch_size`: 64 (larger batch leverages lightweight architecture)
- `epochs`: 100
- `early_stopping_patience`: 15
- `gradient_clip_max_norm`: 5.0

### Lite-STGNN (lightweight — Zoleka)
- `hidden_dim`: 32
- `num_layers`: 2
- `dropout`: 0.1
- `learning_rate`: 0.001
- `batch_size`: 64
- `epochs`: 100
- `early_stopping_patience`: 15
- `gradient_clip_max_norm`: 5.0

### DiffSTG (diffusion — Génevieve)
- `hidden_dim`: 64
- `num_heads`: 4 (attention heads in denoising network)
- `learning_rate`: 2e-4
- `batch_size`: 16 (diffusion models are memory-intensive)
- `epochs`: 100
- `early_stopping_patience`: 15
- `gradient_clip_max_norm`: 1.0 (tighter clip)
- Diffusion schedule:
  - `diffusion.schedule`: linear
  - `diffusion.beta_start`: 1e-4
  - `diffusion.beta_end`: 0.02
  - `diffusion.n_train_steps`: 10 (shortened chain in `forward()` during training for speed)
  - `diffusion.n_infer_steps`: 100 (full chain in `predict()` at evaluation for quality)

### USTD (diffusion — Génevieve)
- `hidden_dim`: 64
- `num_heads`: 4
- `learning_rate`: 2e-4
- `batch_size`: 16
- `epochs`: 100
- `early_stopping_patience`: 15
- `gradient_clip_max_norm`: 1.0
- Diffusion schedule:
  - `diffusion.schedule`: cosine (Nichol & Dhariwal 2021 — differs from DiffSTG)
  - `diffusion.n_train_steps`: 10
  - `diffusion.n_infer_steps`: 100

---

## 11. Shared Training Hyperparameters Summary Table

| Parameter | All models | Notes |
|---|---|---|
| input_len (T) | 12 | ⚠ TODO: confirm |
| horizon (H) | 12 | ⚠ TODO: confirm |
| seeds | 42–51 | 10 seeds, one run each |
| epochs | 100 | max; early stopping applies |
| early_stopping_patience | 15 | epochs without val loss improvement |
| scheduler | CosineAnnealingWarmRestarts | all models |
| scheduler.T_0 | 10 | first cycle length |
| scheduler.T_mult | 2 | cycle doubling factor |
| scheduler.eta_min | 1e-6 | LR floor |
| vpt_epsilon (ε) | 0.5 | ⚠ TODO: confirm |
| clip_sigma | 10.0 | normalised feature clip bound |
| exclusion_threshold | 0.05 | 5% missing → exclude station |
| train/val/test ratio | 0.6 / 0.2 / 0.2 | chronological |
| extreme threshold | 95th percentile | computed on train split only |
| distance_threshold_km | 300.0 | adjacency cutoff |
| sigma_km | 100.0 | edge weight decay |

---

## 12. Output Files per Experiment Run

### Per seed (inside `outputs/<experiment>/run_<seed>/`)
- `checkpoint.pt` — PyTorch state dict of best model (lowest val loss)
- `train.log` — full per-epoch log: epoch, train_loss, val_loss, grad_norm
- `history.json` — structured per-epoch history:
  - `seed` — integer
  - `train_loss` — list of floats, one per epoch
  - `val_loss` — list of floats, one per epoch
  - `grad_norm` — list of floats, one per epoch (pre-clip norm)
  - `stopped_epoch` — epoch at which training stopped (early stop or max epochs)
  - `test_metrics` — dict of all test-set metrics (see below)
- `test_metrics.json` — all evaluation metric values for this seed

### Per experiment (inside `outputs/<experiment>/`)
- `summary.json` — aggregated results across all 10 seeds:
  - For every metric key K: `K_mean` and `K_std`
  - Plus: `model`, `dataset`, `task`, `seeds`, `params_M`, `flops_G`
- `plots/` — auto-generated plots:
  - `loss_curves.png` — overlaid train/val loss per seed
  - `nrmse_horizon.png` — NRMSE vs forecast step (regression only)
  - `metric_boxplot.png` — boxplot of primary metric across seeds

### MLflow (per run, logged to `tracking/mlflow.db`)
- **Parameters:** all config YAML values, seed, model name, dataset, task, params_M, flops_G
- **Metrics (per epoch):** train_loss, val_loss, grad_norm
- **Metrics (final, logged once):** test/mae, test/rmse, test/mse, test/mape, test/mae_extreme, test/mae_nonextreme, test/peak_intensity_error, test/nrmse_per_horizon (list), test/vpt, test/inference_ms_per_sample (regression); test/precision, test/recall, test/f1, test/auc_roc (classification)

---

## 13. Tensor Shapes Reference

| Name | Shape | Description |
|---|---|---|
| X (raw) | (T_total, N, d) | full preprocessed time series |
| X_train | (T_train, N, d) | ~60% of T_total |
| X_val | (T_val, N, d) | ~20% |
| X_test | (T_test, N, d) | ~20% |
| y_train | (T_train, N, d) | binary extreme labels (float) |
| adj | (N, N) | row-normalised adjacency matrix |
| coords | (N, 2) | lat/lon per node |
| mean | (1, N, d) | per-node per-var training mean |
| std | (1, N, d) | per-node per-var training std |
| thresholds | (N, d) | 95th-percentile per node per var |
| x_batch | (B, T, N, d) | model input batch |
| y_batch regression | (B, H, N, d) | regression target batch |
| y_batch classification | (B, H, N) | classification target batch |
| model output regression | (B, H, N, d) | predicted values |
| model output classification | (B, H, N) | predicted logits |

---

## 14. Experiments Defined in the Proposal

- **Experiment 1:** each model evaluated against AGCRN baseline — can it achieve better predictive performance under identical conditions?
- **Experiment 2:** each model evaluated against CLCRN baseline — same question
- Both experiments applied across:
  - Both datasets: SAWS and ERA5
  - Both tasks: regression (Phase 1) and classification (Phase 2)
- Total experiment matrix: 6 experimental models × 2 baselines × 2 datasets × 2 tasks

---

## 15. Open TODOs (Parameters Not Yet Locked)

These are explicitly flagged in the configs and codebase pending team + supervisor confirmation from Gaibie et al. (2024):

- `input_len` (T) and `horizon` (H) — currently 12/12; may need to change
- `learning_rate` for AGCRN and CLCRN — currently 0.001
- `hidden_dim` for AGCRN and CLCRN — currently 64
- `vpt_epsilon` (ε) — currently 0.5; should come from Gaibie et al.
- SAWS feature set — depends on what variables SAWS actually delivers
- ERA5 spatial alignment method — currently grid-native; station-interpolated mode (bilinear) is implemented but not the default
- FLOPs measurement library — currently `torchinfo`; `fvcore` is an alternative
