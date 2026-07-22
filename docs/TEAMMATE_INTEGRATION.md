# WeatherFLOW — Teammate Integration Guide

**Who this is for:** teammates integrating a new model into the WeatherFLOW benchmark. Explains what the shared framework does for you, what the dataset looks like, what your adapter needs to implement, and what the hyperparameter contract is.

**Team scope**

| Teammate | Models | Status |
|----------|--------|--------|
| **Praise** | AGCRN, CLCRN, MTGODE, ClimODE | ✅ Phase 1 complete (40/40 seeds), Phase 2 in progress (19/40 so far) |
| **Zoleka** | STLGRU, Lite-STGNN | ✅ Adapters landed and smoke-tested at N=100; production runs pending |
| **Génevieve** | DiffSTG, USTD | Maintained in Génevieve's own forks — integrate via `genevieve/` scratch dir |

---

## 1. Dataset

We use **ERA5 reanalysis for South Africa only**, downloaded via the Copernicus Climate Data Store (CDS) API. Same dataset for all models — nobody re-downloads.

| Attribute | Value |
|---|---|
| Source | Copernicus C3S (CDS API, `reanalysis-era5-single-levels`, product_type `reanalysis`) |
| Region | South Africa (N=-22°, W=15°, S=-35°, E=33°) |
| Temporal coverage | 2000-01 through 2026-05, hourly (317 months) |
| Native resolution | 0.25° → 3,869 grid points |
| **Resolution we use** | 0.5° (lat_stride=2, lon_stride=2) → **999 grid points** |
| Split | 60/20/20 chronological — train / val / test |
| Location on HPC | `/scratch/jrvpra001/weatherflow/WeatherFLOW/data/era5/raw/` |

### 1.1 Variables — downloaded vs computed

| Variable | Symbol | Units | Source | Description |
|----------|--------|-------|--------|-------------|
| 2-m temperature | `t2m` | K | **Downloaded** | Instantaneous air temperature at 2 m |
| 2-m dewpoint | `d2m` | K | **Downloaded** | Instantaneous dewpoint at 2 m |
| Total precipitation | `tp` | m | **Downloaded** | Accumulated (hourly) precip |
| 10-m u wind | `u10` | m/s | **Downloaded** | Zonal wind component at 10 m |
| 10-m v wind | `v10` | m/s | **Downloaded** | Meridional wind component at 10 m |
| Surface pressure | `sp` | Pa | **Downloaded** | Instantaneous surface pressure |
| 10-m wind speed | `ws10` | m/s | **Computed** | `√(u10² + v10²)` |
| Relative humidity | `rh` | % | **Computed** | Magnus formula from `t2m`, `d2m`, `sp` |

- **Downloaded**: 6 variables via CDS API. Each monthly file is a ZIP containing two inner NetCDF4 files — instantaneous vars in one, accumulated `tp` in the other. Handled transparently by [preprocessing/era5_parser.py](../preprocessing/era5_parser.py).
- **Computed**: 2 derived variables built in [preprocessing/derive.py](../preprocessing/derive.py). `ws10` from wind components (physical); `rh` from temperature/dewpoint/pressure using the Magnus formula (standard meteorology).
- **Total after derivation**: **8 variables** — this is `d=8` you see in tensor shapes. The model is trained to predict **all 8** jointly.

### 1.1a Alignment with Gaibie et al. (2024)

Our HPO for AGCRN and CLCRN comes from Gaibie et al. (2024). Their paper benchmarks on **4 primary target variables**: `t2m` (2m temperature), `rh` (relative humidity), `ws10` (10m wind speed), `sp` (surface pressure). Two things about our variable set to be honest about in the paper:

- **Training input/output = d=8**, not d=4. We feed all 8 (including `tp` and pre-derived `ws10`/`rh`) rather than the raw 5-ish Gaibie fed. Pre-deriving `ws10`/`rh` means the model doesn't have to internally reconstruct them each forward pass, and adding `tp` costs nothing. This is a "same feature set expressed differently" divergence, not a methodological break.
- **Headline metrics filter to Gaibie's 4**. `evaluation/regression.py:REPORT_VARS = ["t2m", "rh", "ws10", "sp"]` — the paper's headline tables report on these 4 primary targets. The other 4 (`d2m`, `tp`, `u10`, `v10`) are reported in the full 8-variable table for completeness / appendix.

**Paper phrasing you can use:** *"The model is trained to jointly predict all 8 variables in the feature vector; headline results are reported on the 4 primary target variables of Gaibie et al. (2024), with the full 8-variable table in Appendix X."*

### 1.2 Phase-2 label variables

For classification (Phase 2), labels are computed for a **subset of 6 variables**: `[t2m, d2m, tp, ws10, rh, sp]`. `u10` and `v10` are excluded — extremes on those are captured by `ws10`, so keeping them would double-count the same wind extreme. This is called `d_label = 6`.

Extreme threshold: **95th percentile per (station, variable), computed on the training split only.** Precipitation uses a wet-hour override (ETCCDI convention) — full details in [results/analysis/WeatherFLOW_Phase2_Plan.md](../results/analysis/WeatherFLOW_Phase2_Plan.md).

---

## 2. Shared experimental framework

The framework does everything except run your model's forward pass. Your adapter plugs in one place; the rest is inherited.

```
    ┌─────────────────────────────────────────────────────────┐
    │                        cli.py                            │
    │            (single entry point, seed loop)               │
    └──────────┬──────────────────────────────────────────────┘
               │
               ▼
    ┌─────────────────────────────────────────────────────────┐
    │           preprocessing/pipeline.py                      │
    │  ── raw ERA5 → chronological split → labels →           │
    │     normalise (train-only stats) → build adjacency →    │
    │     sliding-window DataLoaders                          │
    │  Cache hit? Skip everything and load from disk.         │
    └──────────┬──────────────────────────────────────────────┘
               │
               ▼
    ┌─────────────────────────────────────────────────────────┐
    │              training/trainer.py                         │
    │  For each seed:                                          │
    │    build model via adapter.build_model(config, adj)      │
    │    for each epoch:                                       │
    │       train_epoch()  → adapter.forward(x, adj) → loss    │
    │       val_epoch()    → adapter.predict(x, adj) → loss    │
    │       save best-val checkpoint; check patience           │
    │    load best checkpoint                                  │
    │    test_eval → adapter.predict(x, adj) → compute_all()   │
    └──────────┬──────────────────────────────────────────────┘
               │
               ▼
    ┌─────────────────────────────────────────────────────────┐
    │        evaluation/{regression,classification}.py         │
    │  Standardised metrics — MAE/RMSE/NRMSE/VPT for           │
    │  regression, F1/AUC-ROC/PR-AUC/CSI for classification.   │
    │  Writes test_metrics.json + per_station.npz.             │
    └─────────────────────────────────────────────────────────┘
```

**What you get for free** by writing an adapter:
- Deterministic seeding across NumPy, PyTorch (CPU + CUDA), and Python
- Preprocessed data with cached splits, train-only normalisation, adjacency matrix, labels
- Sliding-window DataLoaders with correct batch shape `(B, T, N, d)`
- Optimizer (Adam), scheduler (CosineAnnealingWarmRestarts), gradient clipping (max_norm=5.0)
- Best-val checkpointing with early stopping (patience=15)
- Rich console progress bars
- Standardised test evaluation, per-station arrays, MLflow logging
- Static metrics: parameter count, FLOPs (via `fvcore`), inference latency

**What your adapter provides**:
- `build_model(config, adj) → nn.Module`
- `forward(x, adj) → predictions` (training)
- `predict(x, adj) → predictions` (evaluation, wrapped in `no_grad()`)
- `get_param_count() → int`

See §5 for the full contract.

---

## 3. Hyperparameters

### 3.1 Shared across all models (do not change without discussion)

These are fixed for the benchmark — changing them would break comparability across models.

| Setting | Value | Rationale |
|---------|-------|-----------|
| `data.input_len` | **12** | 12-hour input window; matches Gaibie et al. (2024) HPO optimum for AGCRN |
| `data.horizon` | **12** | 12-hour forecast horizon; within Gaibie et al. tested range |
| `preprocessing.train_ratio` | 0.6 | Chronological split — 60% train, 20% val, 20% test |
| `preprocessing.val_ratio` | 0.2 | |
| `preprocessing.lat_stride` | 2 | 0.5° subsampling from native 0.25° |
| `preprocessing.lon_stride` | 2 | |
| `preprocessing.distance_threshold_km` | 300 | For graph adjacency (Gaussian-weighted, thresholded) |
| `preprocessing.sigma_km` | 100 | Bandwidth for Gaussian edge weights |
| `training.gradient_clip_max_norm` | 5.0 | Prevents blow-ups from noisy ODE / diffusion gradients |
| `training.early_stopping_patience` | 15 | Applies to all models |
| `training.scheduler.type` | `CosineAnnealingWarmRestarts` | With `T_0=10, T_mult=2, eta_min=1e-6` |
| `evaluation.vpt_epsilon` | 0.3 | NRMSE threshold for Valid Prediction Time (Platt et al. convention) |
| Seeds | 42–51 | 10 seeds per model for Wilcoxon significance testing |

### 3.2 Per-model (each teammate owns these)

Take these from an external HPO source (Gaibie et al. 2024 has HPO for AGCRN/CLCRN; upstream paper defaults for others). Cite the source in the config comment.

| Model | learning_rate | batch_size | epochs (cap) | hidden_dim | Notes |
|-------|--------------:|-----------:|-------------:|-----------:|-------|
| AGCRN | 0.01 | 64 | 100 | 64 | Gaibie et al. (2024) HPO |
| CLCRN | 0.001 | 32 | 100 (regression), 60 (Phase-2 classification) | 64 | Gaibie et al. (2024) HPO; classification cap after seed 44 timeout |
| MTGODE | 0.001 | 32 | 100 | 40 | Upstream paper defaults; seed 49 exception at cap=50 |
| ClimODE | 0.0005 | 16 | 30 | 64 | Adaptive ODE solver → 38 min/epoch at N=999 forces low cap |
| STLGRU | 0.001 | 64 | 100 | 32 | Larger batch OK due to small model |
| Lite-STGNN | 0.001 | 64 | 100 | 32 | 6.9K params — extremely light |

**Convention**: `epochs` is the maximum cap; early stopping typically fires much sooner (Phase-1 CLCRN early-stops at ep 45).

### 3.3 SLURM budget (per production run, per seed)

| Setting | Value | Rationale |
|---------|-------|-----------|
| `--mem` | 256G | Phase 1 test-eval peaks at ~141 GB (predictions + denormalise copies); 256G is the proven envelope |
| `--time` | 24:00:00 | Hard HPC cap on `l40sugrd`; cannot be extended |
| `--gres` | `gpu:l40s:1` | L40S GPU |
| Partition | `l40s` | GPU partition |
| Account | `l40sugrd` | 5-job concurrent limit → drip-feed via `hpc_watcher.sh` |

CPU-only utility jobs (baselines, audits) go to `--partition=ada --account=compsci` — no GPU, no QOS cap.

---

## 4. Ways to integrate

You have three paths, listed in order of ease. Pick the one that matches your model's structure.

**Path A — vendored submodule + adapter (recommended, matches all existing models).** Add the upstream paper repo as a `git submodule` under `models/<family>/<REPO_NAME>/`, then write a thin adapter that imports from it. Keeps the upstream code pristine — you can bump submodule pins to track upstream, and your adapter is the ONLY file you write. All 6 current models follow this pattern.

**Path B — reimplementation (if upstream doesn't fit graph/station format).** If the upstream code assumes gridded data (e.g. ClimODE was designed for 2D H×W grids) or has hard-coded assumptions that don't survive a graph adaptation, write a clean reimplementation under `models/<family>/<name>_model.py`. Cite the paper and document the adaptation. Praise's [climode_model.py](../models/continuous_time/climode_model.py) is the reference example.

**Path C — external fork (Génevieve's current path).** Maintain your model repo separately. When you're ready to pop it in, follow Path A or B. In the meantime, use `genevieve/` as your scratch space in the main repo for integration scripts, notebooks, and drafts.

---

## 5. Adapter integration guide

Every adapter satisfies the [BaseAdapter](../models/base_adapter.py) contract — four methods, ~70 lines total.

### 5.1 File layout

```
models/<family>/
├── <MODEL_REPO>/         ← git submodule of upstream paper repo (Path A)
├── <name>_model.py       ← wraps the model class from the submodule (or reimpl for Path B)
└── <name>_adapter.py     ← implements BaseAdapter
```

For `family`: `baselines`, `continuous_time`, `lightweight`, or `diffusion`.

### 5.2 The contract

Every adapter subclasses [BaseAdapter](../models/base_adapter.py) and implements:

```python
class BaseAdapter(ABC):
    def build_model(self, config, adj) -> nn.Module:      # instantiate & configure
    def forward(self, x, adj) -> torch.Tensor:            # training forward (grad enabled)
    def predict(self, x, adj) -> torch.Tensor:            # eval forward (no grad, eval mode)
    def get_param_count(self) -> int:                     # for reporting
```

**Tensor conventions the trainer will pass and expect:**

| Direction | Shape | Meaning |
|-----------|-------|---------|
| Input `x` | `(B, T=12, N=999, d=8)` | z-scored, clipped to ±10σ |
| Input `adj` | `(N, N)` | normalised adjacency (Gaussian-weighted, thresholded at 300 km) |
| Output (regression) | `(B, H=12, N, d=8)` | z-score space; trainer denormalises before metrics |
| Output (classification) | `(B, H=12, N, d=8)` | LOGITS (pre-sigmoid) for all 8 variables; trainer subsets to `d_label=6` and applies sigmoid |

**Key contract details:**
- If your upstream model uses a different axis order (e.g. `(B, d, N, T)` for MTGODE), permute inside the adapter.
- If your upstream is single-variable, set the output to `H × d` channels and reshape — see [mtgode_model.py](../models/continuous_time/mtgode_model.py) `out_dim = H * d`.
- The `adj` argument is passed for interface consistency; if your model learns its own graph (like AGCRN's node embeddings), ignore it.
- **Same forward path for regression and classification** — don't fork on task. The trainer applies BCE loss on the raw logits for classification; MSE for regression.

### 5.3 Reference adapter (AGCRN, 70 lines)

```python
# models/baselines/agcrn_adapter.py
import torch, torch.nn as nn
from typing import Dict, Any
from models.base_adapter import BaseAdapter
from models.baselines.agcrn_model import AGCRNModel


class AGCRNAdapter(BaseAdapter):

    def __init__(self):
        self.model: AGCRNModel | None = None

    def build_model(self, config, graph):
        N = graph.shape[0]
        model_cfg = config["model"]
        model_cfg["input_dim"]  = config["meta"]["n_vars"]
        model_cfg["output_dim"] = config["meta"]["n_vars"]
        model_cfg["horizon"]    = config["data"]["horizon"]
        self.model = AGCRNModel(model_cfg, num_nodes=N)
        return self.model

    def forward(self, x, adj):
        return self.model(x)

    def predict(self, x, adj):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def get_param_count(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
```

That's the whole thing. If your model is more complex (e.g. ClimODE has ODE solver tolerances that differ between train and eval), put that logic inside `forward` / `predict` — see [climode_adapter.py](../models/continuous_time/climode_adapter.py).

### 5.4 Integration checklist

1. **Add upstream repo as submodule** (Path A):
   ```bash
   git submodule add https://github.com/<AUTHOR>/<REPO> models/<family>/<REPO_NAME>
   ```

2. **Write the model wrapper** — thin file at `models/<family>/<name>_model.py` that imports from the submodule and handles axis permutation / output reshaping.

3. **Write the adapter** — copy the AGCRN template above; adjust the model import + any tensor-order tweaks.

4. **Register in `cli.py`** — add one line to `ADAPTER_REGISTRY`:
   ```python
   "<name>": "models.<family>.<name>_adapter.<Name>Adapter",
   ```

5. **Write a config YAML** — copy `experiments/configs/agcrn.yaml`, tweak `model:` fields and hyperparams. Do NOT change the `data:`, `preprocessing:`, or `evaluation:` blocks (those are shared).

6. **Smoke test locally**:
   ```bash
   python cli.py --model <name> --task regression --dataset era5 \
                 --runs 1 --seed-start 42 --max-nodes 50 --mlflow-off
   ```
   This runs 1 seed at N=50 nodes with reduced epochs. Should complete in a few minutes on a laptop.

7. **GPU smoke test on HPC** (30 min):
   ```bash
   sbatch scripts/jobs/smoke.sh <name>   # runs 1 epoch at N=100 on l40s
   ```
   **Do not skip this step.** History: we lost ~3 days + ~95 GPU-hours by jumping straight from CPU smoke to production. See `memory/feedback_hpc_smoke_test.md`.

8. **Production submission** — use the shared drip-feed watcher (see [scripts/jobs/train_array.sh](../scripts/jobs/train_array.sh) and [hpc_watcher.sh](../hpc_watcher.sh)).

9. **Post-run sync** — outputs land in `outputs/`. Once your matrix completes, add per-seed run dirs to `results/phase{1,2}_*/runs/` and generate a per-model summary.

### 5.5 Common gotchas

- **Device-attribute bug**: some upstream models store tensors as plain Python attributes rather than `register_buffer`, so `.to(cuda)` skips them silently. CLCRN's [clcrn_adapter.py](../models/baselines/clcrn_adapter.py) has a `_move_aux_tensors` walk that handles this. If you see mysterious device-mismatch errors on GPU, this is why.
- **Single-variable upstream models** typically expect `(B, T, N)` inputs. Reshape to loop over the `d` axis, or set `out_dim = H × d` and reshape at the end.
- **Never modify the submodule contents.** All fixes go in your adapter. That way upstream can be bumped without conflict.
- **Do not add a `ClassificationHead`.** Phase 2 keeps the same forward path — model outputs `d=8` logits, the trainer subsets to `d_label=6` at the trailing axis and applies sigmoid inside `evaluation.classification.compute_all`. Your adapter should be identical for regression and classification.

---

## 6. Zoleka's models — completed adapters

Both STLGRU and Lite-STGNN adapters are landed on `main` and smoke-tested at N=100. Production runs haven't been submitted yet.

### 6.1 STLGRU

- Upstream repo: [github.com/Kishor-Bhaumik/STLGRU](https://github.com/Kishor-Bhaumik/STLGRU)
- Adapter: [models/lightweight/stlgru_adapter.py](../models/lightweight/stlgru_adapter.py)
- Config: [experiments/configs/stlgru.yaml](../experiments/configs/stlgru.yaml)
- Model wrapper: [models/lightweight/stlgru_model.py](../models/lightweight/stlgru_model.py)
- **Multi-variable handling**: upstream STLGRU is single-channel. Wrapper sets `out_length = horizon × n_vars`, reshapes to `(B, H, N, d)`.
- **Adjacency**: registered as a buffer on `STLGRUModel`, follows `.to(device)` automatically — no `_move_aux_tensors` needed.
- Params (smoke): 13.4K — extremely lightweight
- Status: **adapter complete, smoke-tested, production run pending**

### 6.2 Lite-STGNN

- Upstream repo: [github.com/HTMoges/Lite-STGNN](https://github.com/HTMoges/Lite-STGNN)
- Adapter: [models/lightweight/litestgnn_adapter.py](../models/lightweight/litestgnn_adapter.py)
- Config: [experiments/configs/litestgnn.yaml](../experiments/configs/litestgnn.yaml)
- Model wrapper: [models/lightweight/litestgnn_model.py](../models/lightweight/litestgnn_model.py)
- **Adjacency**: Lite-STGNN learns its own low-rank TopK adjacency from node embeddings. The static adjacency from preprocessing is ignored (only its shape determines N).
- Params (smoke): 6.9K — even smaller than STLGRU
- Status: **adapter complete, smoke-tested, production run pending**

Both adapters share Zoleka's design pattern: simple, deterministic, no `forward`/`predict` divergence. Same computation path in both, with `predict()` wrapping in `no_grad()`. When Zoleka runs production, she should use the same `train_array.sh` + drip-feed watcher pattern Praise uses (see [scripts/jobs/](../scripts/jobs/)).

### 6.3 Zoleka's smoke-test findings (from RESEARCH_FINDINGS)

STLGRU and Lite-STGNN both showed **positive VPT on multiple variables at smoke-scale (N=100)** — unlike the Praise-owned baselines which had VPT=0 across the board at this scale. Interesting side-finding worth checking at production N=999. Full context in [results/analysis/RESEARCH_FINDINGS.md](../results/analysis/RESEARCH_FINDINGS.md) §STLGRU and §Lite-STGNN.

---

## 7. Génevieve's models — external forks

DiffSTG and USTD are no longer in this git tree. Génevieve maintains them in her own forks. When she's ready to integrate:

- Add each as a git submodule under `models/diffusion/<REPO>/`
- Write adapters at `models/diffusion/{diffstg,ustd}_adapter.py` following the AGCRN template in §5.3
- Register in [cli.py](../cli.py)'s `ADAPTER_REGISTRY` (there are commented-out template lines showing where they go)
- Drop configs at `experiments/configs/{diffstg,ustd}.yaml`

**Scratch space**: [genevieve/](../genevieve/) is Génevieve's folder in the main repo for integration scripts, notebooks, drafts, and anything else she wants tracked with the team. See [genevieve/README.md](../genevieve/README.md).

**Diffusion-specific integration notes** (from prior integration attempts):
- Diffusion models train on a noise-prediction objective, but `predict()` must run the full denoising chain. **This is exactly what the `forward`/`predict` split in BaseAdapter is for** — put the noise-prediction step in `forward`, the full sample-generation loop in `predict`.
- Diffusion models are memory-hungry. Budget `--mem=256G` (same as Praise's models) minimum; may need higher for USTD.
- USTD's upstream code has device-management issues that need adapter-side handling — expect to write a `_move_aux_tensors`-style helper.

---

## 8. Quick reference — what lives where

| Thing | Path |
|-------|------|
| Adapter contract | [models/base_adapter.py](../models/base_adapter.py) |
| CLI entry point + adapter registry | [cli.py](../cli.py) |
| Preprocessing pipeline | [preprocessing/pipeline.py](../preprocessing/pipeline.py) |
| Training loop | [training/trainer.py](../training/trainer.py) |
| Regression metrics | [evaluation/regression.py](../evaluation/regression.py) |
| Classification metrics | [evaluation/classification.py](../evaluation/classification.py) |
| Config templates | [experiments/configs/](../experiments/configs/) |
| SLURM job scripts | [scripts/jobs/](../scripts/jobs/) |
| Baseline reference forecasts | [results/baselines/](../results/baselines/) |
| Results archive (post-analysis) | [results/README.md](../results/README.md) |
| Génevieve's scratch dir | [genevieve/](../genevieve/) |
| Living project findings | [results/analysis/RESEARCH_FINDINGS.md](../results/analysis/RESEARCH_FINDINGS.md) |
| HPC state + lessons | [results/analysis/HANDOFF_STATUS.md](../results/analysis/HANDOFF_STATUS.md) |

## 9. When you're stuck

- Verify setup script: `python scripts/verify_setup.py` — checks imports, configs, dirs, submodules
- Smoke test recipe: §5.4 step 6 above
- The complete team memory of hard-won lessons: `memory/` (Claude-managed, but human-readable)
- Ping Praise. Faster than debugging alone — most integration issues have been hit before.
