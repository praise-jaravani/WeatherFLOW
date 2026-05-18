# WEATHER-FLOW: Pipeline Architecture & Setup Plan

**University of Cape Town — CSC4019Z Honours Project**  
**Authors:** Génevieve Chikwanha, Zoleka Fuyana, Praise Jaravani  
**Supervisor:** Dr. Deshen Moodley  

---

## Overview

This document is the canonical reference for the WEATHER-FLOW shared experimental pipeline. It describes the full architecture, directory structure, component responsibilities, and setup instructions. It is intended to be used as a bootstrapping guide — a Claude Code agent or a human developer should be able to read this document and set up the complete pipeline from scratch.

The pipeline is a **unified benchmarking framework** for comparing eight STGNN architectures across two tasks (regression and classification) on two datasets (SAWS and ERA5/WeatherBench). Every model passes through the same preprocessing, training loop, and evaluation suite, ensuring results are directly comparable across all three workstreams.

The architecture is directly inspired by **OpenSTL (NeurIPS 2023)**, a spatio-temporal benchmarking framework that demonstrated the value of decomposing STL systems into three explicit abstraction layers. We adopt this layering explicitly.

---

## Architectural Principles

### 1. Three-Layer Architecture (from OpenSTL, NeurIPS 2023)

The entire codebase is organised into three abstraction layers, arranged bottom-up:

**Core Layer** — the foundation. Contains all data loading, preprocessing utilities, reusable neural modules, and evaluation metrics. Nothing in the core layer knows about specific models. It provides the building blocks that everything else uses.

**Algorithm Layer** — the model implementations. Each model family (continuous-time, lightweight, diffusion-based) and each baseline lives here, isolated in its own folder. The cloned repositories are kept entirely untouched inside these folders. Following OpenSTL's internal decomposition, each algorithm in the algorithm layer is further broken into three explicit components:
- **method** (`*_adapter.py`) — the training and prediction logic: how the model is driven through forward passes, how loss is computed, and how `predict()` differs from `forward()` at evaluation time
- **model** (`*_model.py`) — the network architecture itself, which wraps the cloned repo's `nn.Module` and exposes a clean interface
- **modules** (`modules/`) — reusable sub-components that multiple models in the same family share (e.g., ODE integration logic shared between MTGODE and ClimODE; denoising utilities shared between DiffSTG and USTD)

**User Interface Layer** — the CLI, configuration system, and output management. This is the only layer a user ever touches directly. It reads configs, dispatches to the algorithm layer, and routes results to the output system.

This separation means: adding a new model requires only adding a new cloned repo and writing adapter, model, and (optionally) module files. The core layer and user interface layer never change.

### 2. Cloned Repositories Are Never Modified

All eight model repositories are cloned as-is and treated as read-only third-party code. The adapter pattern is the mechanism that bridges between each model's internal conventions and the shared pipeline's interface. This preserves reproducibility — the model code is exactly as published by the original authors.

### 3. Preprocessing Is Cached

Raw data is parsed and preprocessed once, saved to `.npz` files, and reused across all runs. A hash file records a fingerprint of the raw data and preprocessing config. On each CLI invocation, the pipeline checks the hash first — if nothing has changed, it loads the cache and skips preprocessing entirely. If the raw data or config has changed, it reruns preprocessing and updates the cache.

### 4. Every Run Is Fully Reproducible

Each experiment run fixes seeds for Python, NumPy, and PyTorch simultaneously using a `seed_everything()` utility. Seeds are explicit and stored in run metadata. The 10 runs per experiment use seeds 42 through 51 (inclusive). Given the same seed, config, and dataset, a run must produce identical results.

### 5. Experiment Tracking via MLflow (Local)

MLflow is used as the experiment tracking backend. It runs entirely locally — no cloud account required. Every run automatically logs hyperparameters, per-epoch metrics, system metrics (GPU utilisation, memory), and final evaluation results. The MLflow UI can be launched at any time to compare runs across models, datasets, and tasks. This is valuable both for day-to-day debugging and for producing the cross-model comparison tables in the final paper.

### 6. Rich Terminal Display

The CLI uses the `rich` library for terminal output, wrapping `tqdm` progress tracking. This provides clean, coloured, real-time display of nested progress (outer: run 1/10; inner: epoch 14/30) alongside live metrics and ETA. All terminal output is simultaneously written to a log file in the run output folder.

### 7. Dual Execution Modes

The pipeline is designed to run in two distinct modes without any code changes. The same `cli.py` entry point is used in both.

**Local development mode** — teammates run the code directly on their own machines using a conda environment defined by `environment.yml` and `requirements.txt`. This is for development, debugging, preprocessing, adapter testing, and single-epoch smoke tests. GPU is not required — CPU is sufficient for all development-phase tasks. Invoked via a `Makefile` with simple human-readable targets.

**HPC cluster mode** — all serious training runs execute on UCT HPC via SLURM batch jobs. The code runs inside a Singularity container (`container.sif`) which provides a frozen, reproducible environment. SBATCH scripts in `scripts/jobs/` wrap the same `cli.py` call. Invoked with `sbatch scripts/jobs/run_experiment.sh`.

The separation is deliberate: local mode optimises for fast iteration; HPC mode optimises for scale and reproducibility. Neither mode requires changes to the pipeline code itself.

### 8. Singularity Containerisation for HPC

The UCT HPC cluster runs Singularity (Apptainer) as its container runtime — Docker is explicitly not supported on multi-user HPC environments. A single `container.def` file at the project root defines the complete execution environment. It is built once into a `container.sif` file and uploaded to HPC. All SBATCH scripts reference this container.

The container base image is an NVIDIA NGC PyTorch image (`nvcr.io/nvidia/pytorch`) pre-validated for the A100's CUDA architecture (`sm_80`). This is important — generic PyTorch Docker images may not include `sm_80` in their architecture list and will fail on the A100. GPU passthrough is handled by Singularity's `--nv` flag, which exposes the host's NVIDIA device entries and CUDA libraries inside the container at runtime.

`container.sif` is never committed to version control (it will be several GB). It is listed in `.gitignore`. `container.def` is committed — it is the source of truth for the HPC environment. When dependencies change, `container.def` is updated and the container is rebuilt.

Singularity does not need to be installed by all team members. Only the person responsible for building the container needs it locally (Linux native, or Windows via WSL2). Once built and uploaded, all three team members reference the same `.sif` from their SBATCH scripts.

---

## Directory Structure

```
weatherflow/
│
├── ARCHITECTURE.md              ← this document
├── README.md                    ← project overview and quickstart
├── requirements.txt             ← pip dependencies for local development
├── environment.yml              ← conda environment for local development
├── Makefile                     ← local development targets (run, preprocess, compare, etc.)
├── container.def                ← Singularity definition file (source of truth for HPC env)
├── container.sif                ← built Singularity container (gitignored; built once, lives on HPC only — never in local working tree)
├── .gitignore                   ← includes container.sif, tracking/mlflow.db, __pycache__
│
│
│── data/                        ← RAW DATA ONLY — never preprocessed outputs
│   ├── saws/
│   │   └── raw/                 ← original .xls files from SAWS, as received
│   └── era5/
│       └── raw/                 ← original .nc NetCDF files from Copernicus CDS
│
│
├── preprocessing/               ← CORE LAYER: data processing
│   ├── __init__.py
│   ├── cache/                   ← cached .npz outputs + hash files
│   │   ├── saws_preprocessed.npz
│   │   ├── era5_preprocessed.npz
│   │   └── preprocessing_hash.json
│   ├── saws_parser.py           ← reads raw SAWS .xls → long-format DataFrame
│   ├── era5_parser.py           ← reads .nc files → station-aligned DataFrame
│   ├── impute.py                ← IDW missing value imputation, 5% exclusion rule
│   ├── normalise.py             ← z-score normalisation per variable per station
│   ├── graph.py                 ← adjacency matrix construction from lat/lon coords
│   ├── labels.py                ← 95th percentile extreme event binary labelling
│   ├── split.py                 ← chronological 60/20/20 train/val/test split
│   ├── dataset.py               ← PyTorch Dataset class + DataLoader factory
│   └── pipeline.py              ← orchestrates the full preprocessing sequence,
│                                   checks hash, loads cache or reruns as needed
│
│
├── models/                      ← ALGORITHM LAYER: model implementations
│   │
│   ├── base_adapter.py          ← abstract base class all adapters must implement
│   ├── shared_modules/          ← reusable sub-components shared across families
│   │   ├── __init__.py
│   │   └── graph_utils.py       ← graph construction helpers used by multiple models
│   │
│   ├── baselines/
│   │   ├── AGCRN/               ← cloned repo (untouched)
│   │   ├── CLCRN/               ← cloned repo (untouched)
│   │   ├── agcrn_adapter.py     ← method: training + prediction logic
│   │   ├── agcrn_model.py       ← model: wraps AGCRN nn.Module
│   │   ├── clcrn_adapter.py
│   │   └── clcrn_model.py
│   │
│   ├── continuous_time/         ← Praise's workstream
│   │   ├── MTGODE/              ← cloned repo (untouched)
│   │   ├── ClimODE/             ← cloned repo (untouched)
│   │   ├── modules/             ← shared ODE sub-components (MTGODE + ClimODE)
│   │   │   ├── __init__.py
│   │   │   └── ode_integrator.py ← torchdiffeq wrapper shared by both models
│   │   ├── mtgode_adapter.py
│   │   ├── mtgode_model.py
│   │   ├── climode_adapter.py
│   │   └── climode_model.py
│   │
│   ├── lightweight/             ← Zoleka's workstream
│   │   ├── STLGRU/              ← cloned repo (untouched)
│   │   ├── Lite-STGNN/          ← cloned repo (untouched)
│   │   ├── modules/             ← shared lightweight sub-components
│   │   │   └── __init__.py
│   │   ├── stlgru_adapter.py
│   │   ├── stlgru_model.py
│   │   ├── litestgnn_adapter.py
│   │   └── litestgnn_model.py
│   │
│   └── diffusion/               ← Génevieve's workstream
│       ├── DiffSTG/             ← cloned repo (untouched)
│       ├── USTD/                ← cloned repo (untouched)
│       ├── modules/             ← shared diffusion sub-components
│       │   ├── __init__.py
│       │   └── denoising.py     ← shared denoising utilities (DiffSTG + USTD)
│       ├── diffstg_adapter.py
│       ├── diffstg_model.py
│       ├── ustd_adapter.py
│       └── ustd_model.py
│
│
├── training/                    ← CORE LAYER: shared training logic
│   ├── __init__.py
│   ├── trainer.py               ← shared training loop, 10-run orchestration
│   ├── losses.py                ← MSE (regression), WBCE with class weights (classification)
│   ├── scheduler.py             ← cosine annealing LR + gradient clipping config
│   └── seed.py                  ← seed_everything(): fixes Python, NumPy, PyTorch seeds
│
│
├── evaluation/                  ← CORE LAYER: evaluation suite
│   ├── __init__.py
│   ├── regression.py            ← MAE, RMSE, MSE, MAPE, Stratified MAE,
│   │                               Peak Intensity Error, NRMSE(t_f), VPT
│   ├── classification.py        ← Precision, Recall, F1-Score, AUC-ROC
│   ├── computational.py         ← params (M), FLOPs (G), inference time (ms/sample)
│   └── stats.py                 ← Wilcoxon signed-rank test across technique families
│
│
├── experiments/                 ← USER INTERFACE LAYER: configs
│   └── configs/
│       ├── agcrn.yaml
│       ├── clcrn.yaml
│       ├── mtgode.yaml
│       ├── climode.yaml
│       ├── stlgru.yaml
│       ├── litestgnn.yaml
│       ├── diffstg.yaml
│       └── ustd.yaml
│
│
├── tracking/                    ← MLflow local tracking backend
│   └── mlflow.db                ← SQLite database, auto-created on first run
│
│
├── outputs/                     ← all experiment outputs, timestamped per run
│   └── 2026-05-18_14-32-01__mtgode__saws__regression/
│       ├── run_01/
│       │   ├── metrics.json     ← per-epoch and final metrics for this seed
│       │   ├── checkpoint.pt    ← best model checkpoint
│       │   └── train.log        ← full terminal output captured to file
│       ├── run_02/ ...
│       ├── run_10/
│       ├── summary.json         ← averaged metrics across all 10 runs + std dev
│       └── plots/               ← loss curves, metric plots for this experiment
│
│
├── scripts/
│   ├── download_era5.py         ← pulls ERA5 data via Copernicus CDS API
│   ├── verify_setup.py          ← checks all repos cloned, stubs importable, configs valid
│   ├── run_baselines.py         ← convenience: runs AGCRN + CLCRN back-to-back on a dataset
│   ├── compare_results.py       ← loads multiple summary.json files, prints cross-model table
│   └── jobs/                    ← SBATCH job scripts for UCT HPC
│       ├── run_experiment.sh    ← parametrised: any model/dataset/task combination
│       ├── run_baselines.sh     ← runs AGCRN and CLCRN back-to-back on HPC
│       └── run_all_phase1.sh    ← submits all Phase 1 regression experiments
│
│
└── tests/
    ├── test_preprocessing.py
    ├── test_adapters.py
    └── test_evaluation.py
```

---

## Component Descriptions

### `data/`

Holds raw data files only. Nothing in this folder is ever generated by the pipeline — it is populated manually by the team as data is received from SAWS and downloaded from ERA5.

**SAWS data** arrives as `.xls` files in a specific SAWS format: one sheet per file, with a legend block at the top, followed by one block per station per year. Each block contains a Day × Month table for a single meteorological variable. Multiple variables (Maximum Temperature, Minimum Temperature, Rain) are stacked sequentially within the same file. These files are placed in `data/saws/raw/` as received.

**ERA5 data** is downloaded from the Copernicus Climate Data Store (CDS) as NetCDF (`.nc`) files. The `scripts/download_era5.py` helper automates this. ERA5 covers global reanalysis data at hourly resolution and must be spatially aligned to SAWS station coordinates before use. ERA5 files are placed in `data/era5/raw/`.

Neither folder should ever contain processed or derived data — those live in `preprocessing/cache/`.

---

### `preprocessing/`

This is the most critical component of the core layer. Its job is to take raw data from both sources and produce a clean, unified, ready-to-train tensor representation that every model in the algorithm layer can consume without further data handling.

The preprocessing pipeline runs in the following fixed sequence:

**Step 1 — Parse.** `saws_parser.py` reads each raw SAWS `.xls` file, navigates the file's station-block structure, and outputs a long-format DataFrame indexed by `(station_id, date, variable)`. Similarly, `era5_parser.py` reads NetCDF files and interpolates ERA5 grid values to SAWS station coordinates, producing a matching long-format DataFrame.

**Step 2 — Impute.** `impute.py` applies the IDW (Inverse Distance Weighting) method to fill missing values using neighbouring stations, weighted by geographic distance. Stations with more than 5% missing data across the full time range are excluded entirely before imputation. This threshold is configurable in the preprocessing config.

**Step 3 — Split.** `split.py` performs a strict chronological 60/20/20 train/validation/test split. No random shuffling is applied at any point — the split is always the first 60% of time for training, next 20% for validation, last 20% for test. The split is done at this stage — before labelling or normalisation — so that all subsequent statistics are computed exclusively from the training partition, preventing any form of data leakage into the validation and test sets.

**Step 4 — Label.** `labels.py` computes the 95th percentile threshold for each variable from the **training partition only** and creates a binary extreme event label for every timestep at every station across all three splits. A value of 1 indicates the observation exceeds the 95th percentile; 0 otherwise. Computing the threshold on the training set only and applying it to validation and test sets is essential — using the full dataset would leak future distributional information into the threshold. The threshold values are saved alongside the `.npz` cache.

**Step 5 — Normalise.** `normalise.py` applies z-score normalisation per variable per station, computing mean and standard deviation from the **training partition only** and applying those statistics to validation and test sets. This prevents the normalisation statistics from incorporating any information from future data. Normalisation statistics are saved alongside the `.npz` cache so that inverse-normalisation can be applied to model outputs at evaluation time.

**Step 6 — Construct Graph.** `graph.py` builds the adjacency matrix from station coordinates. The default method uses inverse distance weighting between station lat/lon pairs with a distance threshold to produce a sparse adjacency matrix. The exact construction method (distance threshold, normalisation scheme) is configurable. This produces the graph `A` that is passed to every model alongside the feature tensor `X`.

**Step 7 — Build DataLoaders.** `dataset.py` wraps the split tensors in a PyTorch `Dataset` subclass and returns `DataLoader` objects for train, validation, and test sets. The input format to the model is a sliding window of `T` historical timesteps; the output target is the next `H` timesteps (regression) or binary extreme event labels for the next `H` timesteps (classification).

**Caching.** `pipeline.py` orchestrates the above steps. Before running, it computes a hash of the raw data files and the preprocessing configuration. It compares this against the stored hash in `preprocessing/cache/preprocessing_hash.json`. If they match, it loads the cached `.npz` files directly and returns. If they differ (or no cache exists), it runs the full pipeline, saves the outputs to `preprocessing/cache/`, and updates the hash file.

---

### `models/`

This is the algorithm layer. It contains all model code and the adapter files that connect each model to the shared pipeline.

#### `base_adapter.py`

Defines the abstract base class that every adapter must implement. This contract has four required methods:

- `build_model(config, graph)` — instantiates and returns the model object from the cloned repo, configured according to the experiment config and graph structure
- `forward(x, adj)` — runs a forward pass **during training**, accepting the standardised input tensor `x` of shape `(batch, T, N, d)` and adjacency matrix `adj` of shape `(N, N)`, and returning predictions of shape `(batch, H, N, d)` for regression or `(batch, H, N)` for classification. Called with model in train mode, gradients enabled.
- `predict(x, adj)` — runs a forward pass **at evaluation time**. Called with model in eval mode, no gradient computation (`torch.no_grad()`). This is a separate method from `forward` because some models behave differently at evaluation — diffusion models use a full denoising chain at inference, while during training they may use a shortened chain for speed; MTGODE may use a tighter ODE solver tolerance at evaluation. The trainer always calls `predict()` on the test set, never `forward()`.
- `get_param_count()` — returns total trainable parameter count, used in the computational efficiency comparison

Every adapter inherits from this base class. The training loop in `training/trainer.py` calls `forward()` during training epochs and `predict()` during validation and test evaluation. It has no knowledge of any model's internals beyond these four methods.

#### Cloned Repositories

Each model's repository is cloned directly into its family subfolder. The folder name matches the repository name exactly. These folders are completely untouched — no files are added, removed, or modified inside them. They are treated as external dependencies, equivalent to installed packages.

Git submodules are the recommended mechanism for managing these clones, which means the main `weatherflow` repository tracks the exact commit hash of each model repo. This ensures full reproducibility — checking out the main repo at any point in time will restore all eight model repos to the exact versions used.

#### Per-Model File Decomposition

Following OpenSTL's internal decomposition, each model in the algorithm layer is represented by three types of files rather than a single adapter:

**`*_adapter.py` (method)** — the training and prediction logic. Its job is purely orchestration: it holds the `forward()` and `predict()` implementations, manages the optimiser step within training, and handles any model-specific training quirks such as diffusion warmup phases or ODE solver tolerancing differences between training and evaluation. This is the only file the training loop ever calls directly.

**`*_model.py` (model)** — the network architecture wrapper. It adds the cloned repo's directory to `sys.path` so internal imports resolve, instantiates the repo's `nn.Module` according to the config, and exposes it as a clean `torch.nn.Module` that the adapter can call. Keeping this separate from the adapter means the architecture is independently inspectable and testable.

**`modules/` (modules)** — reusable sub-components within a family. For `continuous_time/`, this contains `ode_integrator.py`, which wraps `torchdiffeq.odeint` and `torchdiffeq.odeint_adjoint` with shared configuration logic used by both MTGODE and ClimODE. For `diffusion/`, this contains `denoising.py`, which implements the shared denoising schedule logic used by both DiffSTG and USTD. Not all families will have non-trivial modules — `lightweight/modules/` may remain nearly empty.

**`models/shared_modules/`** — sub-components that cross family boundaries. Currently contains `graph_utils.py`, which provides graph construction helpers (distance matrix computation, adjacency normalisation) that multiple models require during their forward passes independently of the preprocessing pipeline's `graph.py`.

---

### `training/`

The shared training loop. This is the engine that every model runs through.

#### `trainer.py`

The `Trainer` class accepts a model adapter, DataLoaders, a config, and an output directory. It runs the training loop for a single seed. The outer loop in `cli.py` calls `Trainer` ten times with different seeds and aggregates the results.

Key responsibilities:
- Per-epoch train and validation passes
- Gradient clipping applied uniformly across all models (max norm configurable per model in YAML)
- Early stopping on validation loss (patience configurable)
- Checkpoint saving: saves the best model weights (lowest validation loss) as `checkpoint.pt`
- Per-epoch metric logging to both the terminal (via rich/tqdm) and MLflow
- ETA computation updated after each epoch

The Trainer has no knowledge of which model it is training — it calls `adapter.forward()` during training epochs, `adapter.predict()` during validation and final test evaluation, and computes loss via the appropriate loss function.

#### `losses.py`

Contains two loss functions:

**MSE** — used for Phase 1 regression. Standard mean squared error.

**WBCE (Weighted Binary Cross-Entropy)** — used for Phase 2 classification. Class weights `w_1` (extreme) and `w_0` (non-extreme) are computed from the training set class distribution at the start of each experiment and stored in the run's metadata. The class imbalance at the 95th percentile threshold is severe (approximately 5% positive, 95% negative), so `w_1` will typically be around 19× `w_0`. These weights are logged to MLflow alongside all other hyperparameters.

#### `scheduler.py`

Configures the learning rate scheduler and gradient clipping. The default schedule is cosine annealing with warm restarts (`CosineAnnealingWarmRestarts`). Gradient clipping uses `torch.nn.utils.clip_grad_norm_` with a configurable max norm. Both defaults follow Gaibie et al. (2024) unless a model-specific config overrides them.

#### `seed.py`

A single `seed_everything(seed: int)` function that sets the random seed for Python's `random` module, NumPy, PyTorch CPU, and PyTorch CUDA simultaneously. Called at the start of every run. The seed value is logged to MLflow and stored in the run's `metrics.json`.

---

### `evaluation/`

The shared evaluation suite. Called after training completes on the test set.

#### `regression.py`

Computes all regression metrics against the ground truth test set:

- **MAE** — mean absolute error, primary metric
- **RMSE** — root mean squared error, penalises large errors
- **MSE** — mean squared error, training loss reference
- **MAPE** — mean absolute percentage error, scale-independent
- **Stratified MAE** — MAE computed separately for extreme timesteps (above 95th percentile) and non-extreme timesteps. This is a key metric for the project's research question about extreme event performance.
- **Peak Intensity Error** — MAE computed only on the predicted maximum value within each extreme event window. Measures how well the model captures the peak magnitude of extremes.
- **NRMSE(t_f)** — normalised RMSE at each forecast step `t_f`, computed as the RMSE at that step divided by the standard deviation of the ground truth. Produces a curve of NRMSE against forecast horizon.
- **VPT (Valid Prediction Time)** — the maximum forecast lead time at which NRMSE remains below a threshold `ε`. Formally: `VPT = max { t_f : NRMSE(t_f) < ε }`. The threshold `ε` is set in the shared config. VPT is the primary metric for measuring temporal robustness, connecting directly to the chaos theory framework.

All metrics are computed per station and then aggregated (mean ± std) across stations. Per-station results are also saved to support the geographic stratification analysis in Phase 3.

#### `classification.py`

Computes all classification metrics for Phase 2:

- **Precision** — fraction of predicted extremes that are true extremes
- **Recall** — fraction of true extremes that are correctly predicted (most important metric given class imbalance)
- **F1-Score** — harmonic mean of precision and recall
- **AUC-ROC** — area under the ROC curve, threshold-independent

All metrics are computed per station and aggregated. The confusion matrix is also saved per run.

#### `computational.py`

Computes efficiency metrics for Research Question 4 (trade-offs between accuracy and computational efficiency). These metrics fall into two groups with different recording points:

**Recorded once before the run loop begins** (deterministic given the model and config, independent of training):

- **Parameter count** — total trainable parameters in millions (M), extracted via `sum(p.numel() for p in model.parameters() if p.requires_grad)`
- **FLOPs** — floating-point operations per forward pass in billions (G), measured using `torchinfo` or `fvcore`. Computed on a single batch of the standard input shape.

**Recorded once after the best checkpoint is loaded** (must reflect the trained model in eval mode):

- **Inference time** — wall-clock milliseconds per sample, averaged over 100 forward passes on the test set with the model in eval mode and gradients disabled. Recording this after training is essential: some models in this project have data-dependent or solver-dependent computation (e.g. ODE solvers with adaptive step size in MTGODE/ClimODE, full denoising chains in DiffSTG/USTD) whose cost at inference reflects the trained model's behaviour, not the initialisation-time model. Measured on the same hardware (UCT HPC A100) for all models to ensure comparability.

All three values are logged to MLflow as parameters (not metrics, since they do not change across epochs) and included in `summary.json` for every run. They feed directly into the efficiency comparison table in the final paper.

#### `stats.py`

Runs the Wilcoxon signed-rank test across technique families at the end of Phase 2. This non-parametric test compares the distribution of F1 scores across runs between each pair of technique families and between each experimental model and the baselines. The test corrects for the fact that the 10 runs per model are not independent (same data, different seeds). Results are stored in the Phase 2 summary output.

---

### `experiments/configs/`

One YAML file per model. Each config file specifies:

- Model-specific hyperparameters (hidden dimensions, number of layers, dropout, etc.)
- Training hyperparameters (learning rate, batch size, number of epochs, early stopping patience)
- ODE solver settings where applicable (for MTGODE and ClimODE: solver type, step size, tolerance)
- Diffusion settings where applicable (for DiffSTG and USTD: number of denoising steps, noise schedule)
- Gradient clipping max norm
- Whether to override any shared defaults from Gaibie et al. (2024)

The YAML structure is flat with a small number of top-level keys. CLI flags can override any individual config value at runtime without editing the file (e.g., `--config.lr 0.001`).

---

### `scripts/`

Utility scripts that support the project workflow but are not part of the pipeline itself.

**`download_era5.py`** — automates ERA5 data acquisition via the Copernicus Climate Data Store (CDS) API. Requires a CDS API key. Downloads hourly reanalysis data for the variables and geographic bounding box covering South Africa, saves to `data/era5/raw/`.

**`verify_setup.py`** — a verification script that should be run after completing the setup steps. It checks: all eight cloned repositories are present and non-empty; all stub Python modules are importable from the project root without errors; all eight config YAML files exist and are valid YAML; the `tracking/` directory exists; `data/` directories exist even if empty. Prints a status table (pass/fail per component) using `rich`. Running this script green is the gate condition before any implementation begins.

**`run_baselines.py`** — a convenience script that runs AGCRN and then CLCRN back-to-back on a specified dataset and task with a single command. Useful during the Setup phase to establish baseline performance before the individual workstreams begin. Wraps two calls to `cli.py` with consistent config.

**`compare_results.py`** — loads multiple `summary.json` files from the `outputs/` directory and prints a formatted cross-model comparison table. Accepts a list of output folder paths or a glob pattern. Useful during Phase 3 for assembling the final results table without opening individual JSON files. Output can be rendered as a `rich` table in the terminal or exported as CSV.

#### `scripts/jobs/`

SBATCH job scripts for submitting experiments to UCT HPC. Each script wraps a `singularity exec --nv` call around `cli.py`. The scripts are parametrised via environment variables or command-line arguments so any model/dataset/task combination can be submitted without editing the script.

**`run_experiment.sh`** — the primary job script. Accepts MODEL, DATASET, and TASK as parameters. Requests one A100 GPU from the `a100` partition via `--gres=gpu:ampere:1`. Sets the wall time to 48 hours (the partition limit). Invokes `cli.py` inside the container with the specified parameters. Standard output and error are written to the run's log file in `outputs/`.

**`run_baselines.sh`** — submits AGCRN and CLCRN as two consecutive SLURM jobs with a dependency chain (second starts when first completes). Used during the Setup phase.

**`run_all_phase1.sh`** — a batch submission script that loops over all six experimental models and submits one job per model for a specified dataset and task. Useful for kicking off all Phase 1 regression experiments at once.

---

### `Makefile`

Provides simple human-readable targets for local development. Wraps `cli.py` calls so teammates don't need to remember the full CLI syntax. Key targets:

- `make run MODEL=mtgode DATASET=saws TASK=regression` — run a single experiment locally
- `make preprocess DATASET=saws` — run preprocessing pipeline for a dataset
- `make baselines DATASET=saws` — run both baselines back-to-back
- `make compare` — run `scripts/compare_results.py` on all outputs
- `make mlflow` — launch MLflow UI in the browser at `localhost:5000`
- `make verify` — run `scripts/verify_setup.py`
- `make test` — run the test suite

The Makefile is purely a convenience wrapper. Every target translates directly into a `python cli.py` or `python scripts/...` call. It does not introduce any additional logic.

---

### `container.def`

The Singularity definition file. This is the source of truth for the HPC execution environment. It is committed to version control and should be updated whenever a new dependency is added.

The definition file specifies:
- **Base image:** an NVIDIA NGC PyTorch container (`nvcr.io/nvidia/pytorch:XX.XX-py3`) that is pre-validated for the A100's CUDA architecture (`sm_80`). This is critical — generic PyTorch images may omit `sm_80` and fail on the A100.
- **`%post` section:** installs all pip packages from `requirements.txt` on top of the base image. Also clones or copies the weatherflow project code into the container.
- **`%environment` section:** sets necessary environment variables (PYTHONPATH, CUDA_VISIBLE_DEVICES, etc.)

The container is built with:
```
singularity build --fakeroot container.sif container.def
```

This requires Singularity to be installed on the build machine. Linux users install it natively; Windows users build inside WSL2. Once built, `container.sif` is uploaded to `/home/<username>/` on UCT HPC and shared across all three workstreams.

`container.sif` is listed in `.gitignore` — it is typically several GB and should not be committed. The `.def` file is the reproducible artefact; the `.sif` is the build output.

---

### `environment.yml` and `requirements.txt`

These two files define the local development environment for teammates running the code on their own machines without Singularity.

`environment.yml` defines a conda environment named `weatherflow` with Python 3.10 (or whichever version matches UCT HPC's module) and all pip dependencies. Created once with `conda env create -f environment.yml` and activated with `conda activate weatherflow`.

`requirements.txt` is the flat pip dependency list used both by conda (referenced inside `environment.yml`) and directly inside the `container.def` to install packages into the Singularity container. This means `requirements.txt` is the single source of truth for package versions — it is used by both execution modes.

**Critical:** whenever a new dependency is added to `requirements.txt`, the conda environment must be updated locally (`conda env update -f environment.yml`) and the Singularity container must be rebuilt and re-uploaded to HPC. These three artefacts must always be kept in sync: `requirements.txt`, `environment.yml`, `container.def`/`container.sif`.

The local environment does not require a GPU. All development tasks — preprocessing, adapter logic, evaluation metrics, debugging — run on CPU. Only full training runs require GPU, and those always go through HPC.

---

### `tracking/`

MLflow's local SQLite tracking database. Created automatically on the first run. To browse the experiment dashboard, run `mlflow ui --backend-store-uri sqlite:///tracking/mlflow.db` from the project root. The UI shows all runs organised by experiment, with metric comparison plots and hyperparameter tables.

MLflow logs the following automatically for every run:
- All config YAML values as parameters
- Per-epoch train loss and validation loss as metrics
- Final evaluation metrics (MAE, RMSE, F1, etc.) as metrics
- GPU utilisation and memory usage as system metrics
- The model checkpoint as an artifact
- The seed used as a parameter

---

### `outputs/`

Every experiment run creates a new timestamped folder here. The naming convention is:

```
YYYY-MM-DD_HH-MM-SS__<model>__<dataset>__<task>/
```

For example: `2026-06-03_09-14-22__mtgode__saws__regression/`

Inside, one subfolder per seed (`run_01/` through `run_10/`) plus a `summary.json` and a `plots/` folder. The summary contains averaged metrics and standard deviations across all 10 seeds. The plots folder contains loss curves and metric-vs-horizon plots generated automatically at the end of the experiment.

This structure means multiple runs of the same model and dataset produce separate timestamped folders and never overwrite each other.

---

### `cli.py`

The single entry point for all experiments. Interface:

```
python cli.py --model <name> --dataset <saws|era5|both> --task <regression|classification> [options]
```

Available models: `agcrn`, `clcrn`, `mtgode`, `climode`, `stlgru`, `litestgnn`, `diffstg`, `ustd`

Optional flags:
- `--runs N` — number of seeds (default 10)
- `--horizon H` — forecast horizon in timesteps
- `--config.<key> <value>` — override any config YAML value at runtime
- `--no-cache` — force preprocessing to rerun even if cache is valid
- `--mlflow-off` — disable MLflow tracking for a quick debug run

The CLI display uses `rich` for terminal output. The outer run loop (run 1/10) and the inner epoch loop (epoch 1/N) are displayed simultaneously as nested progress bars, each showing the current loss and ETA. All output is simultaneously tee'd to `outputs/<run_folder>/run_XX/train.log`.

---

## Setup Instructions

These instructions are for bootstrapping the project from scratch. A Claude Code agent should follow them in order.

### Step 1 — Create Directory Structure

Create all directories as specified in the directory tree above. Create empty `__init__.py` files in all Python package directories (`preprocessing/`, `training/`, `evaluation/`). Create `.gitkeep` files in `data/saws/raw/`, `data/era5/raw/`, `outputs/`, and `preprocessing/cache/` so that the empty directories are tracked by git.

### Step 2 — Clone Model Repositories

Clone all eight model repositories into their designated folders. The exact clone commands are:

```
# Baselines
git clone https://github.com/LeiBAI/AGCRN              models/baselines/AGCRN
git clone https://github.com/BIRD-TAO/CLCRN             models/baselines/CLCRN

# Continuous-Time (Praise)
git clone https://github.com/TrustAGI-Lab/MTGODE        models/continuous_time/MTGODE
git clone https://github.com/Aalto-QuML/ClimODE          models/continuous_time/ClimODE

# Lightweight (Zoleka)
git clone https://github.com/Kishor-Bhaumik/STLGRU       models/lightweight/STLGRU
git clone https://github.com/HTMoges/Lite-STGNN           models/lightweight/Lite-STGNN

# Diffusion (Génevieve)
git clone https://github.com/wenhaomin/DiffSTG            models/diffusion/DiffSTG
git clone https://github.com/hjf1997/USTD                 models/diffusion/USTD
```

After cloning, optionally convert to git submodules so the main repo tracks each model at a fixed commit hash. Do not make any modifications to any file inside a cloned repository.

### Step 3 — Create Stub Files

Create stub (empty or minimally documented) Python files for every module listed in the directory structure. Each stub should contain a module-level docstring describing what the file is responsible for, and `pass` or `raise NotImplementedError` placeholders for all functions and classes. This gives the full importable structure of the package without any implementation yet.

Files to stub:
- All files in `preprocessing/`
- All files in `training/`
- All files in `evaluation/` (including `computational.py`)
- `models/base_adapter.py`
- `models/shared_modules/__init__.py` and `models/shared_modules/graph_utils.py`
- All `*_adapter.py` and `*_model.py` files in `models/baselines/`, `models/continuous_time/`, `models/lightweight/`, `models/diffusion/`
- All `modules/__init__.py` files and named module files within each family directory
- `cli.py`
- All four scripts in `scripts/`

### Step 4 — Create Config Files

Create a YAML config file for each of the eight models in `experiments/configs/`. Each config should have the following top-level sections: `model`, `training`, `evaluation`. Populate with placeholder values where the exact hyperparameter is not yet known; annotate these with a comment indicating the default should follow Gaibie et al. (2024) once confirmed by the team.

### Step 5 — Create Environment Files

Create two environment files that together define the local development environment and serve as input to the Singularity container build.

**`requirements.txt`** — the flat pip dependency list. This is the single source of truth for package versions. It is used by conda locally and by `container.def` on HPC. Both must reference the same file.

Dependencies to include (versions to be pinned once confirmed against UCT HPC's CUDA driver):

- `torch` — PyTorch, version with `sm_80` CUDA architecture support (A100 requirement)
- `torch-geometric` — PyTorch Geometric, for graph operations
- `torchdiffeq` — ODE solver for MTGODE and ClimODE
- `numpy`
- `pandas`
- `scipy` — for IDW imputation and Wilcoxon test
- `scikit-learn` — for AUC-ROC and other sklearn metrics
- `torchinfo` — for FLOPs and parameter count in `evaluation/computational.py`
- `xlrd` — for reading legacy `.xls` SAWS files
- `netCDF4` or `xarray` — for reading ERA5 `.nc` files
- `pyyaml` — for config file parsing
- `tqdm` — progress bars
- `rich` — enhanced terminal display
- `mlflow` — experiment tracking
- `matplotlib` — plotting
- `seaborn` — statistical visualisation

**`environment.yml`** — the conda environment file. Specifies Python version, the conda environment name (`weatherflow`), and references `requirements.txt` for pip dependencies. Teammates create their local environment with `conda env create -f environment.yml`.

**Keeping environments in sync:** whenever a dependency is added or a version is changed in `requirements.txt`, three things must happen: the conda environment is updated locally (`conda env update -f environment.yml`), the Singularity container is rebuilt (`singularity build --fakeroot container.sif container.def`), and the new `container.sif` is uploaded to HPC. All three must be done together — a mismatch between local and HPC environments is a common source of hard-to-debug failures.

### Step 6 — Initialise MLflow

Create the `tracking/` directory. MLflow will create `tracking/mlflow.db` automatically on the first run. Add `tracking/mlflow.db` to `.gitignore` — the database should not be committed to version control as it will grow large and is machine-specific.

### Step 7 — Build the Singularity Container

This step is performed once by one team member (the person on Linux or WSL2) and the resulting `.sif` is shared on HPC.

Write `container.def` specifying an NVIDIA NGC PyTorch base image that includes `sm_80` in its CUDA architecture list. The `%post` section installs all packages from `requirements.txt`. Build with:

```bash
singularity build --fakeroot container.sif container.def
```

Upload `container.sif` to a shared location on UCT HPC (e.g. `/home/<username>/weatherflow/container.sif`). Add `container.sif` to `.gitignore`. All three team members update their SBATCH scripts to reference this path.

Verify the container works on HPC before proceeding:

```bash
singularity exec --nv container.sif python -c "import torch; print(torch.cuda.is_available())"
```

This must return `True`. If it returns `False`, the base image does not support the A100's `sm_80` architecture and a different NGC image tag must be selected.

### Step 8 — Configure HPC Access

All three team members need:
- A general UCT HPC account (apply at `https://ucthpc.uct.ac.za/index.php/account/` with Dr. Moodley as PI)
- GPU partition access (email `eresearch@uct.ac.za` with Dr. Moodley CC'd — see GPU access motivation email template in project documentation)
- UCT VPN installed and tested (`https://icts.uct.ac.za/services-working-remotely/virtual-private-network`) — required for off-campus access to `hpc.uct.ac.za` and `ondemand.uct.ac.za`

GPU access is not granted by default and requires a supervisor-approved motivation email. This must be done before Phase 1 begins (deadline: before 2 June 2026).

### Step 9 — Verify Structure

Run `scripts/verify_setup.py` from the project root. This script checks all components and prints a pass/fail status table. Every item must pass before implementation begins:

- All eight cloned repositories present and non-empty
- All stub Python modules importable without errors
- All eight config YAML files present and valid
- `tracking/` directory exists
- `data/saws/raw/` and `data/era5/raw/` directories exist
- `preprocessing/cache/` directory exists
- `container.def` present in project root
- `container.sif` confirmed present on HPC (checked via SSH)

A fully green verification table is the gate condition for moving from setup into implementation.

---

## Data Flow Summary

The complete data flow through the pipeline for a single experiment run is as follows. Steps 1–9 are identical regardless of whether the run is local or on HPC — the only difference is the execution context in step 0.

**Step 0 — Execution context**
- *Local:* `make run MODEL=mtgode DATASET=saws TASK=regression` → calls `python cli.py ...` directly in the conda environment
- *HPC:* `sbatch scripts/jobs/run_experiment.sh` → SLURM schedules the job on an A100 node → `singularity exec --nv container.sif python cli.py ...` runs inside the container

1. **CLI invocation** — user specifies model, dataset, task, and any config overrides
2. **Config loading** — YAML config is loaded and CLI overrides are applied
3. **Preprocessing check** — hash is compared; cache is loaded or pipeline reruns
4. **DataLoader construction** — train/val/test DataLoaders are built from cached tensors
5. **Adapter and model initialisation** — the appropriate adapter is instantiated, which in turn instantiates the model wrapper (`*_model.py`), which loads the cloned repo and builds the `nn.Module`
6. **Static computational metrics** — parameter count and FLOPs are recorded once here via `evaluation/computational.py`, before the run loop begins. These are deterministic given the model and config and do not change across seeds.
7. **Outer run loop (10 seeds)** — for each seed:
   a. `seed_everything(seed)` is called
   b. MLflow run is started, config parameters and static computational metrics are logged
   c. Model weights are reinitialised
   d. Training loop runs for N epochs with cosine annealing and gradient clipping; `adapter.forward()` is called each epoch
   e. `adapter.predict()` is called on the validation set for checkpoint selection; best checkpoint saved
   f. Best checkpoint is loaded; `adapter.predict()` runs on the test set
   g. Inference time is recorded via `evaluation/computational.py` (100 forward passes on the test set, eval mode, no grad). Recorded here, after the trained checkpoint is loaded, so it reflects the converged model's actual inference behaviour.
   h. All evaluation metrics are computed and logged to MLflow and saved to `run_XX/metrics.json`
   i. MLflow run is closed
7. **Aggregation** — mean and std dev of all metrics across 10 seeds are computed and saved to `summary.json`
8. **Plotting** — summary plots are generated and saved to `plots/`
9. **Terminal summary** — final results table is printed via rich

---

## Key Design Decisions and Rationale

**Why Singularity over Docker?** UCT HPC explicitly does not support Docker — it is architecturally incompatible with multi-user HPC schedulers like SLURM because it requires a root-owned daemon process. Singularity was built specifically for HPC environments and integrates natively with SLURM and NVIDIA GPUs.

**Why an NGC base image rather than a generic PyTorch image?** The UCT HPC A100 uses CUDA architecture `sm_80`. Generic PyTorch Docker images often omit `sm_80` from their compiled architecture list, causing silent failures or errors at runtime. NVIDIA NGC images are pre-validated for specific GPU architectures including the A100 and are the recommended starting point for any HPC deep learning container.

**Why a Makefile for local development?** The CLI flags for a full experiment run are verbose. A Makefile with named targets (`make run`, `make baselines`, `make mlflow`) reduces cognitive overhead for day-to-day development and ensures all three teammates invoke commands consistently. It also documents the available operations in one place.

**Why does only one person need to build the container?** Singularity requires Linux kernel features that are only natively available on Linux. Windows and Mac users can build via WSL2 or a VM respectively, but this adds friction. Since the `.sif` is uploaded to a shared HPC path that all three team members can reference, only the build step requires Linux — everything else is just SSH and sbatch.

**Why adapters instead of rewriting models?** Rewriting would introduce divergence from the published implementations, undermining the validity of comparisons. Adapters preserve the original code while providing a clean shared interface.

**Why separate `*_adapter.py` and `*_model.py` instead of one file?** Following OpenSTL's method/model decomposition, separating training logic from architecture definition makes each independently testable and inspectable. The model file can be imported and interrogated (parameter counts, FLOPs) without touching the training logic.

**Why a `predict()` method separate from `forward()`?** Training and evaluation are not the same operation for all models in this project. Diffusion models run a shortened denoising chain during training for speed, but the full chain at test time for quality. MTGODE may use a looser ODE solver tolerance during training for speed and a tighter one at test time for accuracy. Having a distinct `predict()` method in the base contract makes this distinction explicit and prevents the common bug of accidentally evaluating with the training-mode forward pass.

**Why `models/shared_modules/`?** Graph construction utilities are needed both during preprocessing (to build the static adjacency matrix saved in the cache) and during certain model forward passes (e.g., models that recompute the graph dynamically). Rather than duplicating this logic, `shared_modules/graph_utils.py` provides it as a shared dependency.

**Why computational metrics in the evaluation suite?** Research Question 4 explicitly asks about accuracy/efficiency trade-offs. Without params, FLOPs, and inference time recorded alongside accuracy metrics, this question cannot be answered. Adding `computational.py` to the evaluation suite ensures these are captured for every model automatically.

**Why `.npz` cache instead of re-parsing every run?** SAWS parsing and IDW imputation are non-trivial operations. Caching eliminates this overhead for the hundreds of runs the project will execute. The hash-based invalidation ensures the cache is never stale.

**Why MLflow instead of just JSON files?** JSON files per run are necessary for portability but insufficient for comparison. MLflow provides the cross-run comparison infrastructure (metric plots, parameter tables, run filtering) that would otherwise require writing custom analysis scripts. It runs locally and adds no external dependencies or accounts.

**Why `rich` over plain `tqdm`?** With 10 seeds × 8 models × 2 datasets × 2 tasks = up to 320 experiment runs, clear terminal feedback is important. `rich` allows nested progress bars (run + epoch simultaneously visible) and colour-coded status without any significant overhead. It wraps `tqdm` internally so the tqdm API still works.

**Why seeds 42–51?** Fixed consecutive seeds ensure reproducibility while providing genuine variance across runs. The specific values are arbitrary but documented — any team member or future researcher can reproduce any run exactly.

**Why Wilcoxon and not t-test?** The 10 runs per model are not independent samples from a population — they are the same model on the same data with different initialisations. The Wilcoxon signed-rank test is a non-parametric test appropriate for this paired comparison structure and does not assume normality of the metric distributions.

---

## Open Questions (To Be Resolved by Team)

The following decisions are not yet locked in and need team alignment before implementation:

1. **Input window T and forecast horizon H** — the proposal references these but does not fix values. Default should follow Gaibie et al. (2024). As a concrete external reference point: OpenSTL's WeatherBench-S benchmark (single-variable, hourly) uses T=12, H=12; their WeatherBench-M benchmark (multi-variable, 6-hourly) uses T=4, H=4. Given that WEATHER-FLOW targets hourly data with multiple variables, T=12/H=12 is a reasonable candidate to propose to the team and supervisor for confirmation.
2. **Feature set d** — which meteorological variables are included as input features. All available from SAWS (temperature, humidity, pressure, wind speed, wind direction, rainfall), or a subset? The current SAWS extract only contains max temp, min temp, and rain — the full variable set depends on what SAWS delivers.
3. **Adjacency matrix construction method** — distance-based with threshold, or fully connected with distance-weighted values? The threshold value needs to be decided. `models/shared_modules/graph_utils.py` will implement whichever method is chosen.
4. **VPT threshold ε** — the NRMSE threshold below which a prediction is considered "valid". Literature convention and Gaibie et al. should inform this.
5. **ERA5 spatial alignment method** — bilinear interpolation from ERA5 grid to SAWS station coordinates, or nearest-neighbour? Bilinear is standard but adds complexity.
6. **FLOPs measurement library** — `torchinfo` is listed in `requirements.txt` as the default for `evaluation/computational.py`. Confirm this is available on UCT HPC, or substitute `fvcore` (Facebook Research) which is an equivalent alternative.
7. **Git submodule vs. plain clone** — whether to formalise model repos as submodules. Submodules are cleaner for reproducibility but add friction for contributors unfamiliar with git submodules.
