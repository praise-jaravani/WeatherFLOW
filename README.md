# WeatherFLOW

**UCT CSC4019Z Honours Project 2026**  
Génevieve Chikwanha · Zoleka Fuyana · Praise Jaravani  
Supervisor: Dr. Deshen Moodley

---

## What this is

WeatherFLOW is a shared benchmarking pipeline for comparing eight spatio-temporal graph neural network (STGNN) architectures on South African weather data. The goal is to find out which model families — continuous-time ODEs, lightweight GNNs, and diffusion models — perform best on weather forecasting and extreme event detection in our region, where published benchmarks are sparse.

The design is directly inspired by **OpenSTL (NeurIPS 2023)**, a spatio-temporal benchmarking framework. The key insight from OpenSTL is that every benchmarking framework should separate itself into three layers — data/evaluation utilities at the bottom, model implementations in the middle, and a CLI on top — so that every model goes through exactly the same data pipeline, training loop, and evaluation suite. That is what this codebase does.

Every model we compare shares: the same preprocessed input tensors, the same train/val/test split, the same loss functions, the same scheduler, the same evaluation metrics, and the same 10-seed run protocol. The only thing that differs per model is the adapter file that drives it. This is what makes the results directly comparable.

---

## Models

| Family | Models | Workstream |
|---|---|---|
| Baselines | AGCRN, CLCRN | shared |
| Continuous-time | MTGODE, ClimODE | Praise |
| Lightweight | STLGRU, Lite-STGNN | Zoleka |
| Diffusion | DiffSTG, USTD | Génevieve |

Model repositories are cloned into `models/<family>/<Name>/` and are **never modified**. All integration happens through adapter files (`*_adapter.py`, `*_model.py`) that wrap each repo's internal code in the shared interface. This preserves reproducibility — we run the exact code the original authors published.

---

## Datasets and tasks

**ERA5** — hourly reanalysis data (6 variables: 2m temperature, dewpoint, surface pressure, u/v wind components, total precipitation) over a South Africa bounding box from Copernicus CDS. Currently downloaded for 2010–2023. Files live in `data/era5/raw/`.

**SAWS** — South African Weather Service station data. Not yet received. Will live in `data/saws/raw/` when it arrives.

**Phase 1 — Regression:** predict future weather values. Metrics: MAE, RMSE, MAPE, stratified MAE (extreme vs. non-extreme events), NRMSE(t_f) curve, VPT (Valid Prediction Time).

**Phase 2 — Classification:** predict whether a future timestep exceeds the 95th-percentile threshold for each variable (extreme event detection). Metrics: Precision, Recall, F1-Score, AUC-ROC, Wilcoxon signed-rank across model families.

---

## What's implemented

**Preprocessing pipeline** — all six stages are complete and tested:
1. ERA5 parser (reads `.nc` files, subsamples the spatial grid)
2. IDW imputation for missing values; stations with >5% missing data are excluded
3. Chronological 60/20/20 train/val/test split (no shuffling — split always precedes labelling and normalisation to prevent leakage)
4. Binary extreme-event labelling at 95th-percentile thresholds computed on the training set only
5. Z-score normalisation with statistics computed on the training set only
6. Adjacency matrix construction from lat/lon coordinates (inverse-distance weighting with a configurable distance threshold)

Results are cached to `preprocessing/cache/` with hash-based invalidation — the pipeline only reruns if the raw data or config changes.

**Training layer** — fully implemented:
- Shared `Trainer` class: 10-seed outer loop, per-epoch train/val passes, gradient clipping, early stopping, checkpoint saving
- MSE loss (regression) and weighted BCE (classification, class weights computed from training labels at ~5:95 imbalance)
- CosineAnnealingWarmRestarts LR scheduler
- `seed_everything()` for fully reproducible runs (Python + NumPy + PyTorch CPU + CUDA)
- MLflow tracking (local SQLite backend, logs hyperparams, per-epoch losses, final metrics)
- Rich terminal display with nested progress bars

**Evaluation suite** — fully implemented:
- Regression: MAE, RMSE, MSE, MAPE, stratified MAE, peak intensity error, NRMSE-per-horizon curve, VPT
- Classification: Precision, Recall, F1, AUC-ROC, confusion matrix
- Computational: parameter count (M), FLOPs (G), inference time (ms/sample)
- Statistical: Wilcoxon signed-rank test across technique families

**Model adapters** — baseline workstream fully implemented:
- AGCRN adapter and model wrapper (including a fix for uninitialized weights in the upstream repo that caused NaN training — see below)
- CLCRN adapter and model wrapper (including a NumPy 2.x compatibility patch for the upstream repo)

Continuous-time, lightweight, and diffusion adapters exist as stubs — they import cleanly and raise `NotImplementedError` from `build_model()`. The shared ODE integrator (`ode_integrator.py`) that MTGODE and ClimODE both need is also implemented.

**CLI and Makefile** — fully implemented. See the Makefile section below.

**Test suite** — 49 tests covering preprocessing, evaluation, and all adapters. Run with `make test`.

---

## What's outstanding

| Item | Owner | Notes |
|---|---|---|
| MTGODE adapter | Praise | `mtgode_adapter.py` is a stub |
| ClimODE adapter | Praise | `climode_adapter.py` is a stub |
| STLGRU adapter | Zoleka | `stlgru_adapter.py` is a stub |
| Lite-STGNN adapter | Zoleka | `litestgnn_adapter.py` is a stub |
| DiffSTG adapter | Génevieve | `diffstg_adapter.py` is a stub |
| USTD adapter | Génevieve | `ustd_adapter.py` is a stub |
| `denoising.py` | Génevieve | Shared denoising module for DiffSTG + USTD |
| SAWS data | team | Not yet received from SAWS |
| HPC GPU access | team | Email eresearch@uct.ac.za before 2 June 2026 |

---

## Setup

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git (with submodule support)
- For HPC container builds: Singularity/Apptainer on Linux or WSL2

### Clone and create the environment

```bash
git clone --recurse-submodules git@github.com:praise-jaravani/WeatherFLOW.git
cd WeatherFLOW

conda env create -f environment.yml
conda activate weatherflow
```

The `--recurse-submodules` flag clones all eight model repositories at their pinned commit hashes.

### Verify everything is wired up

```bash
make verify
```

This runs `scripts/verify_setup.py` and prints a status table. Everything should be green before you start implementing.

---

## Makefile — the commands you'll actually use

All day-to-day operations go through the Makefile so you don't have to remember CLI flags.

```bash
# Run a full experiment (10 seeds, all logging on)
make run MODEL=agcrn DATASET=era5 TASK=regression

# Quick smoke test — 50 nodes, 1 seed, 2 epochs, no MLflow
# Use this to check your adapter works before committing anything
make smoke MODEL=agcrn DATASET=era5 TASK=regression

# Run preprocessing only (no training)
make preprocess DATASET=era5

# Run both baselines back-to-back
make baselines DATASET=era5 TASK=regression

# Print a comparison table from all outputs in outputs/
make compare

# Launch the MLflow UI at http://localhost:5000
make ui

# Run the test suite (fast, skips ODE integration tests)
make test

# Run the full test suite including slow ODE tests
make test-all
```

You can override any config value inline, for example:

```bash
make run MODEL=agcrn DATASET=era5 TASK=regression RUNS=1 \
    --config.training.epochs 5 \
    --config.training.learning_rate 0.0005
```

---

## MLflow UI

MLflow records every run automatically — hyperparameters, per-epoch losses, final metrics. To view results in a browser:

```bash
make ui
# then open http://localhost:5000
```

You'll see experiments organised by `model/dataset/task`. You can compare loss curves across seeds, filter by metric, and export tables. The tracking database lives at `tracking/mlflow.db` (gitignored — it's local to your machine).

---

## Smoke testing your adapter

Before submitting a real training run to HPC, always run the smoke test locally first. It subsamples the graph to 50 nodes, runs 1 seed for 2 epochs, and skips MLflow. It finishes in a few minutes and will catch shape errors, NaN losses, and import failures before you waste GPU hours:

```bash
make smoke MODEL=<your-model> DATASET=era5 TASK=regression
```

A passing smoke test prints real (non-NaN) MAE and RMSE values. If you see `nan`, something is wrong with the adapter or the data flow. Check `outputs/<run>/run_42/train.log` for the per-epoch breakdown.

---

## Containerisation and HPC

All serious training runs happen on UCT HPC using an A100 GPU. The pipeline runs inside a Singularity container because HPC clusters don't support Docker (it requires root access that multi-user systems can't give you). Singularity is the standard HPC container runtime.

### How the container works

`container.def` at the project root is the source of truth. It's built into `container.sif` (several GB, gitignored) once and uploaded to HPC. All three of us reference the same `.sif` from our SBATCH scripts.

The base image is an NVIDIA NGC PyTorch container specifically validated for the A100's CUDA architecture (`sm_80`). This matters — generic PyTorch Docker images often don't include `sm_80` and will fail silently or error at runtime.

### Building the container (Linux or WSL2 only)

```bash
make build-container
```

This runs `singularity build --fakeroot container.sif container.def`. Only needs to happen once, or whenever `requirements.txt` changes.

### Uploading to HPC

```bash
make upload-container HPC_USER=<your-uct-username>
# or both steps at once:
make sync-hpc HPC_USER=<your-uct-username>
```

### Submitting jobs on HPC

```bash
# Any model/dataset/task combination
sbatch scripts/jobs/run_experiment.sh  # edit MODEL/DATASET/TASK at the top of the script

# Submit all Phase 1 regression experiments at once
bash scripts/jobs/run_all_phase1.sh
```

**GPU access:** By default your HPC account doesn't have access to the A100 partition. You need to email `eresearch@uct.ac.za` with Dr. Moodley CC'd and request GPU access. Do this before 2 June 2026 — it can take a few days to process.

---

## Things worth knowing before you write adapter code

**The adapter contract.** Every model must implement four methods from `models/base_adapter.py`: `build_model()`, `forward()`, `predict()`, and `get_param_count()`. The trainer calls `forward()` during training and `predict()` during validation and test — these can differ (e.g., a diffusion model might use a shortened denoising chain in `forward()` for speed but the full chain in `predict()` for quality).

**Don't touch the cloned repos.** All integration happens in `*_adapter.py` and `*_model.py`. If the upstream code has a bug or incompatibility, patch it in the model wrapper, not in the repo itself — AGCRN and CLCRN both needed patches and we handled them this way.

**The preprocessing cache.** The pipeline hashes the raw data files and the config. If neither has changed, it loads the cached tensors directly. Use `--no-cache` or `make preprocess --no-cache` to force a rerun. The cache lives in `preprocessing/cache/` which is gitignored.

**NaN in training.** ERA5 total precipitation is extremely right-skewed — z-scores reach 57σ after normalisation. The pipeline clips all normalised features to ±10σ at DataLoader build time (`clip_sigma=10.0` in the preprocessing config). If you're getting NaN losses, this is the first thing to check. Also check for uninitialized weights in your model — `torch.FloatTensor(shape)` allocates raw uninitialized memory and can produce NaN from the very first forward pass. Use `torch.zeros()`, `torch.randn()`, or `nn.init.*` instead.

**Seeds.** Each of the 10 runs uses seeds 42 through 51. `seed_everything()` sets Python, NumPy, PyTorch CPU, and PyTorch CUDA simultaneously. Results must be reproducible given the same seed, config, and data.

**Outputs.** Each run creates a timestamped folder under `outputs/`. Inside: one subfolder per seed with `checkpoint.pt`, `history.json`, and `train.log`, plus a top-level `summary.json` (mean ± std across seeds) and a `plots/` folder with loss curves and metric plots. These are gitignored.

---

## Project structure (brief)

```
WeatherFLOW/
├── cli.py                      ← single entry point for all experiments
├── Makefile                    ← all day-to-day commands
├── experiments/configs/        ← one YAML config per model
├── preprocessing/              ← data pipeline (parse, impute, split, label, normalise, graph)
├── models/
│   ├── base_adapter.py         ← abstract interface all adapters implement
│   ├── shared_modules/         ← graph utilities shared across model families
│   ├── baselines/              ← AGCRN, CLCRN + their cloned repos
│   ├── continuous_time/        ← MTGODE, ClimODE + shared ODE integrator (Praise)
│   ├── lightweight/            ← STLGRU, Lite-STGNN (Zoleka)
│   └── diffusion/              ← DiffSTG, USTD + shared denoising module (Génevieve)
├── training/                   ← trainer, losses, scheduler, seed
├── evaluation/                 ← regression, classification, computational, stats
├── scripts/                    ← download_era5, verify_setup, run_baselines, compare_results
│   └── jobs/                   ← SBATCH scripts for HPC
├── tests/                      ← test suite (49 tests)
├── container.def               ← Singularity container definition (source of truth for HPC env)
└── ARCHITECTURE.md             ← full design documentation
```

`ARCHITECTURE.md` has the complete picture — component responsibilities, data flow, every design decision and its rationale. Worth reading before you start implementing.
