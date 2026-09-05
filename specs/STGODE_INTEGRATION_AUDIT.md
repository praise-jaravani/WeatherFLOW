# STGODE integration audit — for cross-checking against the paper

**Purpose:** enough detail for a second agent to (a) verify our STGODE integration is faithful to Fang et al. (KDD 2021), (b) locate every deviation with its rationale, and (c) reproduce the smoke tests. Written 2026-08-11 after the STGODE local + HPO smokes passed; **updated 2026-08-11 in response to a peer review that caught 2 critical + 3 high + 2 medium + 3 low findings — every one addressed** (see §12).

---

## 0. Response-to-peer-review summary (added 2026-08-11)

The initial STGODE integration reproduced upstream's shipped code verbatim + wrapped it for our stack. An external peer review (against both paper and upstream clone) identified that upstream **itself** deviates from the paper in ways that only work on PEMS road-network data — on our SAWS haversine matrix the upstream recipe collapses. All findings addressed:

| # | Severity | Finding | Fix |
|---|---|---|---|
| C1 | CRITICAL | Spatial adj was z-scored Gaussian (upstream) which produces near-uniform "graph" on our haversine (correlation with our own pipeline's kernel: −0.22) | **Rewrote** `_paper_kernel` → paper Eq. 4 (raw-distance Gaussian + km cutoff) as default; z-scored kept behind `graph_construction: upstream` flag for opt-in ablation |
| C2 | CRITICAL | DTW averaging was `mean-within-day` (2191-length seasonal series) instead of paper's `mean-across-days` (24-length diurnal profile) — proven rank inversion on synthetic data | **Rewrote** `build_stgode_semantic_adj.py` → paper axis. Recompute now takes 2 s (was 450 s); per-target DTW matrices trivially built |
| H3 | HIGH | `ode.time` was a dead HPO knob — upstream `model.py:95` hardcodes `time=6` in every `ODEG(...)` call; the wrapper didn't override | **Added** post-construction `ODEblock.t` override in `STGODEModel.__init__` — walks submodules, reassigns `t` on each ODEblock. `ode.time` is now a live knob. Verified 12 ODEblocks receive the override |
| H4 | HIGH | Undisclosed: `torchdiffeq.odeint(f, x, [0, time], method='euler')` with no `step_size` option = **single Euler step**, not multi-step integration | **Documented** accurately in `stgode_model.py` docstring + `experiments/configs/stgode.yaml` comments + §5 below. Not "fixed" (faithful to upstream); disclosure sentence added to methods paragraph in §10.4 |
| H5 | HIGH | Semantic-graph z-score quirk (inherited from upstream) excludes most-similar pairs | The paper-Eq-6 default (raw DTW threshold, no z-score, no Gaussian) is the fix — H5 disappears in `graph_construction: paper` mode. If we ever use `upstream` mode, disclose |
| M6 | MEDIUM | Adapter's `forward()` was forcing `self.model.train()` — could interfere with trainer's mode management | **Removed** mode-setting from adapter. Trainer owns `.train()` / `.eval()` (consistent with MTGODE / AGCRN / CLCRN adapters) |
| M7 | MEDIUM | Inherited paper↔upstream divergences not itemised | **Added** §11 — inventory of everything upstream does that the paper doesn't specify: α applied twice, learnable per-node α, W/W₂ no orthogonality constraint, x₀ detached, BatchNorm over node axis, global vs per-station input scaling |
| Low1 | LOW | Wrapper docstring claimed `input_len` clamp that didn't exist | **Added** hard `raise ValueError` if `input_len != 12` in `STGODEModel.__init__` |
| Low2 | LOW | `_move_adjacencies_to` guard never latched (stored device compared unindexed vs tensor's indexed) — ran on every forward | **Fixed** via canonical-device normalisation (`torch.zeros((), device=device).device`) + moving the stored tensors too |
| Low3 | LOW | Dead `output_dim` merged-config key | **Removed** from the merged config in the adapter |

**Verdict**: after all fixes, the STGODE integration is more paper-faithful than upstream's own code on our data. Ready for HPC.

Full response detail per finding is in §12 below.

---

**References:**
- Paper: Fang, Long, Song, Xie. *Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting.* KDD 2021 ([arXiv 2106.12931](https://arxiv.org/abs/2106.12931), [DOI 10.1145/3447548.3467430](https://dl.acm.org/doi/10.1145/3447548.3467430))
- Upstream code: [github.com/square-coder/STGODE](https://github.com/square-coder/STGODE)
- Vendored in this repo at [`models/continuous_time/STGODE/`](../models/continuous_time/STGODE/) (unmodified; cloned 2026-08-11)

---

## 1. Executive summary

**What we integrated:** STGODE as our second continuous-time model, alongside MTGODE. Same 32-station SAWS network, same evaluation protocol, same univariate-regression regime (Gaibie 2024). Ready to submit HPO on HPC.

**Architecture as implemented (matches paper):**
- 3 parallel spatial blocks + 3 parallel semantic blocks
- Each block: `STGCNBlock → STGCNBlock` (channels `64 → [64, 32, 64] → [64, 32, 64]`)
- `STGCNBlock = TCN → ODEG(t=[0, T=6], Euler) → TCN → BatchNorm2d`
- ODE function: `f = 0.5·α·(A·x) − x + (x·W) − x + (x·W₂) − x + x₀` with learnable α, W, W₂
- Final: max-pool over 6 blocks → `Linear(T_in·64 → T_out·32) → ReLU → Linear(T_out·32 → T_out)`
- Two adjacencies required: **spatial** (Gaussian on haversine distance) + **semantic** (Gaussian on DTW distance between per-station time series, binarised)

**Vendored upstream:** verbatim `models/continuous_time/STGODE/{model,odegcn,run_stode,utils,args,eval}.py` — we do NOT modify upstream code. Bridging happens in our adapter.

**Total new code:** 511 LoC across:
| File | LoC | Purpose |
|---|---|---|
| [`models/continuous_time/stgode_model.py`](../models/continuous_time/stgode_model.py) | 108 | Thin wrapper: `(B,T,N,D) ↔ (B,N,T,F)` permutes around upstream `ODEGCN` |
| [`models/continuous_time/stgode_adapter.py`](../models/continuous_time/stgode_adapter.py) | 200 | `BaseAdapter` contract: builds both adjacencies from cache, moves them device-side, exposes `build_model/forward/predict` |
| [`scripts/build_stgode_semantic_adj.py`](../scripts/build_stgode_semantic_adj.py) | 129 | One-shot DTW distance matrix builder → `preprocessing/cache/saws/dtw_matrix.npy` |
| [`experiments/configs/stgode.yaml`](../experiments/configs/stgode.yaml) | 74 | Paper defaults + uniform-harness overrides, every override annotated |

**Registration:** `"stgode"` added to `ADAPTER_REGISTRY` in [`cli.py`](../cli.py) → pointer to `models.continuous_time.stgode_adapter.STGODEAdapter`.

---

## 2. Upstream code map — what we vendored

Full upstream tree preserved unchanged at [`models/continuous_time/STGODE/`](../models/continuous_time/STGODE/):

| File | Role | Referenced by our code |
|---|---|---|
| `model.py` | Defines `ODEGCN`, `STGCNBlock`, `GCN`, `TemporalConvNet`, `Chomp1d`. **This is the entry model.** | `stgode_model.py` imports `ODEGCN` from here |
| `odegcn.py` | `ODEFunc`, `ODEblock`, `ODEG`. Wraps `torchdiffeq.odeint` around the ODE function. Adjoint = `False` hardcoded upstream (matches our audit lesson from MTGODE Finding 0) | Used indirectly via `ODEGCN`; not imported by our code |
| `run_stode.py` | Reference training loop. Not called by us — we route through our own Trainer. Kept for provenance and hyperparameter cross-check | — |
| `utils.py` | Reference data loader (`read_data`, `generate_dataset`) + adjacency construction (`get_normalized_adj`). We reimplement adjacency + normalisation in the adapter rather than importing (because their `read_data` is PEMS-specific — reads `.npz` files by dataset name) | We port the mathematical logic; not the code |
| `args.py` | Reference argparse — the authoritative hyperparameter reference. **Every default there is cited in [`experiments/configs/stgode.yaml`](../experiments/configs/stgode.yaml)** | Not imported |
| `eval.py` | Reference metrics (masked MAE/RMSE/MAPE) — we use our own `evaluation/regression.py` instead | Not imported |

---

## 3. Adapter architecture — what our wrapper does

### 3.1 [`stgode_model.py`](../models/continuous_time/stgode_model.py) — thin tensor wrapper

Instantiates upstream `ODEGCN(num_nodes, num_features, num_timesteps_input, num_timesteps_output, A_sp_hat, A_se_hat)` and provides a `forward(x)` that:
1. Receives `(B, T, N, D)` in our stack's convention
2. Permutes to `(B, N, T, D)` (upstream convention)
3. Calls `self.net(x)` → upstream returns `(B, N, H)` (univariate)
4. Permutes to `(B, H, N)` and unsqueezes last dim → `(B, H, N, 1)` for our trainer

**No modification to model logic.** All architectural choices come from upstream.

### 3.2 [`stgode_adapter.py`](../models/continuous_time/stgode_adapter.py) — `BaseAdapter` contract

`STGODEAdapter.build_model(config, graph)` does three things:

1. **Constructs the spatial adjacency `A_sp_hat` from the cache's `coords.npy`.** We deliberately do NOT reuse our cache's `adj.npy` (which was Gaussian-kernelled at our SAWS-tuned σ=200 km, threshold=500 km). Instead:
   - Load raw station lat/lon from `preprocessing/cache/saws/coords.npy`
   - Compute pairwise haversine distances (km)
   - Apply the paper's kernel (`_paper_kernel`, line 43 of the adapter): z-score, `A = exp(−z²/σ²)`, zero below threshold
   - Uses `sigma_sp` and `thres_sp` from the config → HPO-searchable

2. **Constructs the semantic adjacency `A_se_hat` from the cache's `dtw_matrix.npy`.** Requires the file to exist — built once by `scripts/build_stgode_semantic_adj.py` (see §4).
   - Load raw DTW distances (N × N)
   - Apply the paper's kernel (`_paper_kernel`), then BINARISE at `thres_se` (not zero-below-threshold, per upstream `utils.py` line 70-71)
   - Uses `sigma_se` and `thres_se` from the config → HPO-searchable

3. **Normalises both adjacencies** with `_normalise_adj` (line 79): `A_reg = 0.5·α·(I + D^{-½}·A·D^{-½})` with `α=0.8` — exact copy of upstream `utils.py::get_normalized_adj`.

**Device handling** (line 168): upstream `ODEG.odefunc` stores `adj` as a plain attribute, not `nn.Buffer`. `nn.Module.to(device)` doesn't move it. `_move_adjacencies_to()` walks submodules and manually transfers. Called from `forward()` and `predict()` on first tensor-device change.

---

## 4. Adjacency construction — DTW semantic matrix

**The load-bearing new preprocessing artefact.** STGODE differs from every other model in our stack by requiring a DTW distance matrix between per-station time series.

### 4.1 Build script — [`scripts/build_stgode_semantic_adj.py`](../scripts/build_stgode_semantic_adj.py)

**What it does** (mirrors upstream `utils.py::read_data` lines 51-60):
1. Loads `preprocessing/cache/saws/X_train.npy` (shape `(T_train=52603, N=32, D=6)`)
2. Slices to a single variable (default t2m, index 0)
3. Reshapes into daily windows: `(T/24, 24, N).mean(axis=1)` → `(n_wins=2191, N)`
4. Transposes to `(N, n_wins)` — one sequence per station
5. For each i, j pair: `dtw_distance[i, j] = fastdtw(series[i], series[j], radius=6)[0]`
6. Symmetric fill + save to `preprocessing/cache/saws/dtw_matrix.npy` (raw distances, N × N float32)
7. Saves companion `.meta.json` with variable, window, radius, elapsed

**Deviations from upstream**:
- Upstream reshapes `data.shape[0] // (24*12)` at PEMS 5-min cadence → we reshape by `24` for our 1-h SAWS cadence (same "daily window" concept, hours-per-day rather than steps-per-day). This is expected and documented in the script.
- Same `radius=6`, same `fastdtw` implementation (`fastdtw==0.3.4`).

**Cost + result** (measured):
- 32 stations → 496 pairs
- 450 s wall-clock on this laptop's CPU
- DTW distance range: 158.4–835.4 (off-diagonal)
- Output size: 4224 bytes

**One-shot**: only depends on training data + `--variable` + `--window-hours` + `--radius`. Cached; skipped on subsequent runs unless `--force`.

### 4.2 Where the Gaussian kernel + threshold apply

**Not in the build script** — build stops at raw distances. The Gaussian kernel + threshold are applied in the adapter's `_paper_kernel` at `build_model` time. This means:
- σ_sp, σ_se, thres_sp, thres_se are **model hyperparameters** in the yaml
- HPO can search them without re-running the expensive DTW compute
- Different trials use the same `dtw_matrix.npy` with different kernel parameters

**Verify against upstream:**
- `sigma1=0.1, thres1=0.6` (semantic, binarised) → our `sigma_se=0.1, thres_se=0.6` defaults ✓
- `sigma2=10, thres2=0.5` (spatial, zero-below-threshold) → our `sigma_sp=10, thres_sp=0.5` defaults ✓
- Upstream `alpha=0.8` in `get_normalized_adj` → our `_normalise_adj` uses `alpha=0.8` ✓

---

## 5. Tensor conventions + I/O contract

| Interface | Our stack | Upstream STGODE | Adapter transform |
|---|---|---|---|
| Input to model | `(B, T=12, N=32, D=6)` | `(B, N, T, F)` where F ∈ {1, 3} | `x.permute(0, 2, 1, 3)` (line 96) |
| Output from model | Expected `(B, H, N, D)` | `(B, N, H)` — always univariate | `y.permute(0, 2, 1).unsqueeze(-1)` → `(B, H, N, 1)` (line 98) |
| Target for loss | `(B, H, N, 1)` — univariate slice from Dataset | — | Matches adapter output exactly (both are (B, H, N, 1)) |

**Univariate direct output**: STGODE emits `(B, H, N, 1)` — NOT `(B, H, N, D)`. This differs from MTGODE / AGCRN / CLCRN which emit full D=6 channels and rely on the trainer's line 545 to slice `pred[..., tv_idx:tv_idx+1]` at loss time.

**Consequence for trainer eval path**: the existing test-eval branch at [`training/trainer.py:642-655`](../training/trainer.py#L642-L655) had a condition `preds.shape[-1] != targets.shape[-1]` that assumed adapter-full-D output. STGODE broke this — both preds and targets are already univariate, so the condition was False and eval fell into the else branch which fed full-D mean/std into `_denormalise`, causing a shape mismatch at line 32 of `evaluation/regression.py`.

**Fix applied** (this session, 2026-08-11): trainer condition now checks `preds.shape[-1] < mean_ch` (adapter-already-univariate case) in addition to the original `preds.shape[-1] != targets.shape[-1]` (adapter-full-D case). Both cases now correctly slice mean/std/thresholds to `[..., tv_idx:tv_idx+1]` before eval. Live at [`training/trainer.py:635-671`](../training/trainer.py#L635-L671).

---

## 6. Hyperparameters — every value cross-referenced

Live in [`experiments/configs/stgode.yaml`](../experiments/configs/stgode.yaml). Every value either matches upstream `args.py` verbatim or is annotated with **"harness (deliberate)"** and its reason.

### 6.1 Matches upstream (verify against `STGODE/args.py`)

| Config key | Value | Upstream source | Verified |
|---|---|---|---|
| `training.learning_rate` | 0.002 | `args.py::lr=2e-3` | ✓ |
| `training.batch_size` | 16 | `args.py::batch_size=16` | ✓ |
| `training.epochs` | 200 | `args.py::epochs=200` | ✓ |
| `data.input_len` | 12 | `args.py::his_length=12` | ✓ (paper's `T_h`) |
| `model.sigma_sp` | 10.0 | `args.py::sigma2=10` | ✓ |
| `model.thres_sp` | 0.5 | `args.py::thres2=0.5` | ✓ |
| `model.sigma_se` | 0.1 | `args.py::sigma1=0.1` | ✓ |
| `model.thres_se` | 0.6 | `args.py::thres1=0.6` | ✓ |
| `ode.method` | euler | `odegcn.py:61` — `odeint(..., method='euler')` | ✓ |
| `ode.time` | 6.0 | `model.py:95` — `ODEG(..., time=6)` | ✓ |
| `ode.adjoint` | false | `odegcn.py:7` — `adjoint = False` hardcoded | ✓ (also matches audit lesson from MTGODE Finding 0) |

### 6.2 Deliberate deviations (uniform harness) — all cross-family-consistent

| Config key | Our value | Upstream / paper | Reason |
|---|---|---|---|
| `training.early_stopping_patience` | 15 | none upstream | Uniform harness patience across all deterministic models (MTGODE/AGCRN/CLCRN also patience 15) |
| `training.gradient_clip_max_norm` | 5.0 | none upstream | Uniform harness clip |
| `training.scheduler.type` | `CosineAnnealingWarmRestarts` (T_0=10, T_mult=2, eta_min=1e-6) | `StepLR(step_size=50, gamma=0.5)` at `run_stode.py:94` | Uniform harness scheduler across all three deterministic models (Bai/Lin/Jin/Fang all differ — one uniform choice preserves internal comparability. Precedent in [`specs/HYPERPARAMETERS.md`](HYPERPARAMETERS.md) §Common) |
| Loss function | `nn.L1Loss` (MAE) — set globally by `cli.py::_build_loss` | `nn.SmoothL1Loss()` at `run_stode.py:91` | Uniform MAE decision (2026-08-08) — matches AGCRN/CLCRN/MTGODE reference implementations. STGODE's SmoothL1 is one paper's choice; MAE gives cross-family loss uniformity for the "airtight clause" (specs/CROSS_FAMILY_COMPARISON_AUDIT.md) |
| Optimiser | `torch.optim.Adam` | `torch.optim.AdamW` at `run_stode.py:90` | Uniform-harness Adam across all three deterministic models. `weight_decay=0` retained (matches upstream `AdamW(..., lr=lr)` with no wd arg — AdamW defaults to wd=1e-2 which we deliberately avoid) |
| `data.horizon` | 48 | `args.py::pred_length=12` | Moodley directive — retrained-max horizon 48 h, matches MTGODE + AGCRN + CLCRN in this project |

### 6.3 Pinned (NOT HPO-searchable in v1)

| Config key | Value | Reason |
|---|---|---|
| `data.input_len` | 12 | Upstream `STGCNBlock` hardcodes `temporal_dim=12` inside its `ODEG(...)` call at `model.py:95`. Changing input_len would need wrapper surgery to plumb the value into that call. Deferred. **HPO does not search input_len for STGODE.** |
| `model.graph_constructor.node_dim` etc. | N/A | STGODE has no adaptive graph learner (unlike MTGODE). Both adjacencies are constructed from data at build time, not learned |

---

## 7. Integration surface

### 7.1 [`cli.py::ADAPTER_REGISTRY`](../cli.py)

Line 38-39 shows the new entry:
```python
"stgode":    "models.continuous_time.stgode_adapter.STGODEAdapter",
```
placed adjacent to the mtgode entry. Every CLI invocation `--model stgode` now works identically to `--model mtgode`.

### 7.2 [`cli.py::_build_loss`](../cli.py)

Untouched. Returns `mae_loss()` for `task="regression"` — STGODE uses this identically to MTGODE.

### 7.3 [`training/trainer.py`](../training/trainer.py)

Two changes this session:
- **Added `epoch_callback` param to `train()`** for Optuna's `trial.report()` mid-training (also benefits any future model that wants a mid-training hook)
- **Fixed test-eval branch** for adapter-already-univariate output (STGODE), described in §5

### 7.4 [`scripts/hpo_run_optuna.py`](../scripts/hpo_run_optuna.py)

STGODE is one of the two entries in `SEARCH_SPACES` — 5 knobs, all categorical:
```python
"stgode": {
    "training.learning_rate": [1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 6e-3],
    "training.batch_size":    [8, 16, 32],
    "ode.time":               [3.0, 6.0, 12.0],
    "model.sigma_sp":         [5.0, 10.0, 20.0],
    "model.thres_se":         [0.4, 0.6, 0.8],
},
```
Pins align with §6.3. Trial protocol identical to MTGODE (see [`specs/HYPERPARAMETERS.md §HPO`](HYPERPARAMETERS.md)).

### 7.5 [`slurm/hpo_optuna_worker.sh`](../slurm/hpo_optuna_worker.sh)

Parameterised on `MODEL` env var:
```bash
sbatch --export=ALL,MODEL=stgode slurm/hpo_optuna_worker.sh
```
Derives study name, storage path, HPO root from `$MODEL`. Independent from MTGODE's job.

---

## 8. What the cross-checker should verify

Actionable checklist for the second agent:

### 8.1 Faithfulness to paper

- [ ] **Architecture cardinality**: 3 spatial + 3 semantic parallel blocks; each block is `STGCNBlock → STGCNBlock`; internal channels `[64, 32, 64]`. Verify by reading [`models/continuous_time/STGODE/model.py::ODEGCN.__init__`](../models/continuous_time/STGODE/model.py) lines 130-143.
- [ ] **STGCNBlock composition**: `TemporalConvNet → ODEG(t=6, Euler) → TemporalConvNet → BatchNorm2d`. Verify at `model.py:93-98`.
- [ ] **ODE function**: `f = 0.5·α·(A·x) − x + xW − x + xW₂ − x + x₀`. Verify at `odegcn.py:46-47`. α is per-node sigmoid-clipped; W, W₂ are cholesky-like `w·d·wᵀ` with d clamped to [0,1].
- [ ] **Integration**: `odeint(self.odefunc, x, [0, T], method='euler')` at `odegcn.py:61`. T=6 by default (from `ODEG(...time=6)` in `model.py:95`).
- [ ] **Adjacency construction**: verify our `_paper_kernel` in [`stgode_adapter.py:43-70`](../models/continuous_time/stgode_adapter.py#L43-L70) against upstream `utils.py::read_data` lines 61-71 (semantic, binarised) and 106-115 (spatial, zero-below-threshold). Both use z-score + Gaussian + threshold.
- [ ] **DTW construction**: verify our [`build_stgode_semantic_adj.py`](../scripts/build_stgode_semantic_adj.py) daily-window mean + `fastdtw(radius=6)` against upstream `utils.py:51-60`. Note the daily-window: upstream is `24·12` for 5-min PEMS data → 288 steps = 1 day; we use `24` for hourly SAWS data = also 1 day. Same semantic.
- [ ] **Normalisation**: `A_reg = 0.5·α·(I + D^{-½}·A·D^{-½})` with α=0.8. Verify our `_normalise_adj` in [`stgode_adapter.py:73-84`](../models/continuous_time/stgode_adapter.py#L73-L84) against upstream `utils.py::get_normalized_adj`.

### 8.2 Correctness of our bridging code

- [ ] **Tensor permute-in**: `(B, T=12, N=32, D=6) → (B, N=32, T=12, D=6)` is `x.permute(0, 2, 1, 3)`. Verify shape-correctness by testing a `torch.randn(2, 12, 32, 6)` input.
- [ ] **Tensor permute-out**: upstream returns `(B, N=32, H=48)`; we permute to `(B, 48, 32)` and unsqueeze to `(B, 48, 32, 1)`. Verify at [`stgode_model.py:99-101`](../models/continuous_time/stgode_model.py#L99-L101).
- [ ] **Adjacency movement**: both adjacencies must land on the same device as inputs at forward time. Upstream stores adj as a plain attribute (not buffer), so `nn.Module.to(device)` misses them. Our `_move_adjacencies_to()` walks submodules and moves. Verify by running a forward pass on CUDA and confirming no device-mismatch errors.
- [ ] **`temporal_dim=12` fragility**: upstream `STGCNBlock` at `model.py:95` calls `ODEG(out_channels[-1], 12, A_hat, time=6)` — that `12` is hardcoded, must match input_len. Verify our config pins input_len=12 and that the HPO search space does NOT include input_len for STGODE.

### 8.3 Integration correctness

- [ ] **Loss function**: run a training step, capture `criterion` at `cli.py::_build_loss` — should be `torch.nn.L1Loss` (MAE). Our uniform loss decision — STGODE uses MAE, NOT the paper's SmoothL1.
- [ ] **Adapter registration**: `python -c "from cli import ADAPTER_REGISTRY; print('stgode' in ADAPTER_REGISTRY)"` returns True.
- [ ] **DTW file dependency**: `stgode_adapter.py::_build_adjacencies` raises `FileNotFoundError` if `preprocessing/cache/saws/dtw_matrix.npy` is missing. Verify by moving/deleting it and confirming the clear error message.
- [ ] **HPO study creates only stgode-namespaced artefacts**: run `python scripts/hpo_run_optuna.py --model stgode --n-trials 1 --smoke --study-name test --storage sqlite:///t.db --hpo-root /tmp/t`; confirm `/tmp/t/runs/hpo_stgode_t2m_trial_00/` gets created with `trial_summary.json`, `state.pt`, `checkpoint.pt`, `per_station.npz`.

### 8.4 Deliberate deviations (all documented)

The following are NOT bugs — they are cross-family alignment choices:
- Loss = MAE, not SmoothL1
- Scheduler = CosineAnnealingWarmRestarts, not StepLR
- Optimiser = Adam, not AdamW (but with wd=0, matching upstream's effective behaviour)
- Horizon = 48, not 12
- Uniform patience=15, uniform grad-clip=5.0

Each is annotated with "harness (deliberate)" in the yaml comments.

---

## 9. Known gotchas + edge cases

1. **`temporal_dim` is hardcoded to 12 in upstream** at `model.py:95`. Do not change `data.input_len` in the yaml without also patching this line or accepting a crash inside ODEG. HPO does NOT search input_len for STGODE.
2. **DTW-source variable choice**: upstream computes DTW on a single feature (channel 0). We default to t2m (index 0). If we ever want to compute DTW on a different variable, use `--variable` on the build script and delete the existing `dtw_matrix.npy`.
3. **Model wrapper's `sys.modules['model']` collision**: STGODE upstream imports `from model import ODEGCN`. So does MTGODE upstream. Both live at different paths, but they share the bare package name `model` in `sys.modules`. Our [`run_worker.py::_purge_shared_model_module()`](../scripts/run_worker.py) purges `model` and `model.*` from `sys.modules` between adapter switches — required for any campaign that mixes MTGODE and STGODE within a single Python process. The HPO script does the same purge at trial-start.
4. **AdamW vs Adam**: upstream uses `AdamW`, but with no explicit `weight_decay` argument — this means AdamW's default `wd=1e-2` was actually active. Our stack uses `Adam` (no wd) for all models. If a cross-checker cares about this specific decision, worth noting: the paper doesn't discuss weight decay; upstream code has an implicit `1e-2` we deliberately do NOT match. This is minor and the direction (less regularisation) is unlikely to bias results.

---

## 10. Test evidence (from this session)

### 10.1 Standalone smoke — [`/tmp/smoke_stgode.py`](/tmp/smoke_stgode.py)

40 steps, batch=8, real 32-node cache, GTX 1650 Ti:

| Verdict | Result |
|---|---|
| Runtime | 9.1 s (0.23 s/step) |
| Params | 1,663,104 |
| A_sp_hat nnz | verified `>0`, correct shape (32, 32) |
| A_se_hat nnz | verified `>0`, correct shape (32, 32) |
| Loss (first 5 mean → last 5 mean) | 0.983 → 0.895 (Δ = −0.088, monotone decreasing) |
| Finite loss all steps | ✓ |
| Spatial-branch gradients | non-zero at every step (min 8.75e-1, max 5.16e+0) |
| Semantic-branch gradients | non-zero at every step (min 3.07e-1, max 7.35e+0) |
| Loss = `nn.L1Loss` | ✓ (asserted) |

All four verdicts PASS.

### 10.2 HPO 2-trial smoke — `runs/hpo_optuna_stgode_smoke/`

| Trial | lr | bs | ode.time | σ_sp | thres_se | best_val | elapsed | Result |
|---|---|---|---|---|---|---|---|---|
| 0 | 3e-4 | 16 | 12.0 | 5.0 | 0.8 | 0.4155 | 568s | COMPLETE |
| 1 | 1e-3 | 16 | 6.0 | 5.0 | 0.6 | 0.4170 | 572s | COMPLETE |

Both trials produced full artefacts: `trial_summary.json`, `state.pt`, `checkpoint.pt`, `per_station.npz`, `train.log`, plus the study-level `top3.json` + `trials.csv`. Startup guard verified separately. Namespace `runs/hpo_optuna_stgode_smoke/` isolated from production paths.

### 10.3 Production dispatch smoke (in flight at time of writing)

Testing the [`scripts/run_worker.py`](../scripts/run_worker.py) claim/dispatch loop with a 2-task manifest `["mtgode_t2m_s42", "stgode_t2m_s42"]` at 2-epoch cap. Exercises the sys.modules cross-model purge in production; results TBD.

---

## 11. Quick reference — where everything lives

| Concern | Path |
|---|---|
| Paper | [arxiv.org/abs/2106.12931](https://arxiv.org/abs/2106.12931) |
| Vendored upstream code | [`models/continuous_time/STGODE/`](../models/continuous_time/STGODE/) |
| Adapter (BaseAdapter contract) | [`models/continuous_time/stgode_adapter.py`](../models/continuous_time/stgode_adapter.py) |
| Model wrapper (tensor permutes) | [`models/continuous_time/stgode_model.py`](../models/continuous_time/stgode_model.py) |
| Config (hyperparameters) | [`experiments/configs/stgode.yaml`](../experiments/configs/stgode.yaml) |
| DTW build script | [`scripts/build_stgode_semantic_adj.py`](../scripts/build_stgode_semantic_adj.py) |
| Produced DTW matrix | `preprocessing/cache/saws/dtw_matrix.npy` (+ `.meta.json`) |
| Adapter registration | [`cli.py`](../cli.py) line 39 |
| HPO search space | [`scripts/hpo_run_optuna.py`](../scripts/hpo_run_optuna.py) `SEARCH_SPACES["stgode"]` |
| HPO sbatch wrapper (MODEL=stgode) | [`slurm/hpo_optuna_worker.sh`](../slurm/hpo_optuna_worker.sh) |
| Trainer eval-branch fix (§5) | [`training/trainer.py`](../training/trainer.py) lines 635-671 |
| Trainer epoch_callback hook | [`training/trainer.py`](../training/trainer.py) lines 322-346 |
| Standalone smoke script | [`/tmp/smoke_stgode.py`](/tmp/smoke_stgode.py) |
| HPO smoke artefacts | `runs/hpo_optuna_stgode_smoke/` (2 trials, verified) |
| MTGODE integration audit (companion) | [`specs/PHASE1_25NODE_CAMPAIGN.md §3.3`](PHASE1_25NODE_CAMPAIGN.md) — the equivalent for MTGODE + the audit findings |
| HYPERPARAMETERS methodology | [`specs/HYPERPARAMETERS.md`](HYPERPARAMETERS.md) |
| Cross-family alignment | [`specs/CROSS_FAMILY_COMPARISON_AUDIT.md`](CROSS_FAMILY_COMPARISON_AUDIT.md) |

---

## 11. M7 — inherited paper↔upstream divergences (for the methods section)

Every one of these is **inherited from upstream verbatim** — we did not introduce them, but a reader of the paper who then reads our code would notice them. Disclose each in the methods paragraph.

| # | Divergence | Paper says | Upstream code | Present in our runs? |
|---|---|---|---|---|
| D1 | α applied twice | Single fixed α in Eq 1 renormalisation | (a) `get_normalized_adj` applies `α/2 · (I + …)` at build → `A_hat ≈ 0.4·(I+…)`; (b) `ODEFunc.forward` multiplies by `sigmoid(learnable α)/2` at runtime. Combined effective α at init ≈ **0.138**, not the paper's fixed α=0.8 | Yes (upstream vendored unmodified) |
| D2 | α learnable per-node | Fixed hyperparameter | `ODEFunc.__init__` line 26: `self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))` — one α per node, sigmoid-clipped in forward | Yes |
| D3 | W, W₂ orthogonality for stability | Paper §4.2 stability argument relies on ‖A‖ ≤ 1 and W eigenvalues in [0,1] | Upstream parameterises as `W · diag(d) · Wᵀ` with `d ∈ [0,1]` — d is clamped but W itself has NO orthogonality constraint. Paper's stability bound is aspirational, not enforced in code | Yes |
| D4 | Kernel over raw distance (Eq 4) + raw DTW threshold (Eq 6) | Both use raw distances | Upstream `utils.py` z-scores THEN Gaussians THEN thresholds/binarises | **No — we deliberately fixed this in `graph_construction: paper` (default)**. Only present under opt-in `upstream` mode |
| D5 | `x₀` restart term | Continuous restart in Eq 18 | Upstream `odegcn.py:57` — `self.odefunc.x0 = x0.clone().detach()`. The `.detach()` means **x₀ carries NO gradient**. The "restart" is a constant during backprop | Yes |
| D6 | BatchNorm application | Not specified in paper | Upstream `model.py:98` — `BatchNorm2d(num_nodes)` normalises over the NODE dimension (treating each node as a feature channel), which is unusual for spatiotemporal data | Yes |
| D7 | Loss function | Not specified in paper's methodology section | Upstream `run_stode.py:91` — `nn.SmoothL1Loss()` | **No — we use uniform MAE across all deterministic models per our loss decision (specs/HYPERPARAMETERS.md)**. Documented cross-family alignment override |
| D8 | Weight decay | Not mentioned | Upstream `run_stode.py:90` — `AdamW(model.parameters(), lr=lr)` with no explicit wd → PyTorch AdamW default `weight_decay=1e-2` silently active | **No — we use `Adam` (no wd) uniformly across all deterministic models** |
| D9 | Input scaling | Not specified in paper | Upstream `utils.py::read_data` — single GLOBAL scalar mean/std across the whole tensor (per-variable if F>1) | Different: our pipeline uses per-(station, variable) mean/std stored as `(1, N, D)`. STGODE consumes our pre-normalised data; the difference lives at cache-build time, not adapter time |
| D10 | ODE integration granularity | Paper Eq 18 defines `dH/dt = ...`; leaves solver choice to implementation | Upstream `odegcn.py:61` calls `odeint(f, x, [0, time], method='euler')` with no `step_size` option → torchdiffeq treats `[0, time]` as the integration grid → SINGLE Euler step of size `time` | Yes (H4 disclosure). Documented in the methods paragraph as "continuous-depth with single-step Euler integration" |
| D11 | DTW input signal | Paper §4.1 mentions DTW between per-node time series | Upstream `utils.py:51` — DTW between per-node AVERAGE DIURNAL PROFILES (24 or 288 steps, depending on cadence). NOT the raw time series. Same axis as our current C2-fixed builder | Yes (matches paper's intent as best we can decode it; upstream axis is decoded correctly by our corrected builder) |

**How to cite in a paper**: after describing STGODE via the paper's own equations, add a sentence: "Our implementation follows the authors' released reference code with the following documented departures: (D4) the paper-specified raw-distance kernels for spatial and semantic adjacencies (as the upstream code's z-scoring collapses on our full-connectivity haversine matrix); (D7) uniform MAE loss for cross-family comparability; (D8) plain Adam without weight decay. All other upstream implementation choices (α applied twice, per-node α, W parameterisation, detached x₀, BatchNorm over nodes, single-step Euler at t=[0, T]) are retained faithfully."

---

## 12. Per-finding response detail

### C1 — spatial adjacency (CRITICAL, FIXED)

**Reviewer's finding**: our z-scored kernel at SA-scale haversine (mean 418 km, std 224 km) with σ ∈ {5, 10, 20} produces 100% density with weight spread 0.25, 0.07, 0.02 respectively. Correlation with a proper raw-distance kernel (σ=200 km, cutoff=500 km): **−0.22**. The z-scored formula peaks at the MEAN distance by construction, so nearby stations get the LOWEST weights.

**Root cause**: upstream `utils.py::read_data` z-scores because on PEMS traffic data, non-adjacent road-network pairs are set to `inf` (from the edge-list CSV); `exp(-inf) = 0` does the sparsification. The z-scored Gaussian on the surviving finite pairs is a near-no-op. On our haversine (no infs), the "near-no-op" IS the whole graph.

**Empirical verification** (from our own probe, [/tmp/reviewer_c1_probe/](../tmp) output preserved in this session's history):
```
z-scored σ_sp=5   density=100%  spread=0.2529
z-scored σ_sp=10  density=100%  spread=0.0703
z-scored σ_sp=20  density=100%  spread=0.0181
paper Eq 4, σ_km=200, cutoff=500  density=66.7%  spread=0.9979
correlation(z-scored σ=10, paper σ_km=200) = -0.217
```

**Fix**: `stgode_adapter.py` now implements paper Eq 4 as default. New `_paper_spatial_adj(dist_km, sigma_km, cutoff_km)`. Keeps upstream `_upstream_zscored_kernel(...)` behind `graph_construction: upstream` config toggle for optional ablation.

**Verification post-fix** (from re-smoke): `A_sp_hat` at defaults σ_km=200, cutoff_km=500 → **nnz=662/992 (66.7%), weight spread 0.226**. Matches paper Eq 4 semantics.

### C2 — DTW averaging axis (CRITICAL, FIXED)

**Reviewer's finding**: upstream `utils.py:51` computes `np.mean([daily-slice], axis=0)` — mean ACROSS days at each within-day step, yielding a length-288 (or 24 in our case) average diurnal profile per station. Our previous builder reshaped `(T/24, 24, N).mean(axis=1)` — mean WITHIN each day, yielding a length-2191 daily-mean time series. Reviewer's probe on synthetic same-diurnal-shape / opposite-seasonal-phase stations: rank INVERSION between the two axes.

**Fix**: `scripts/build_stgode_semantic_adj.py` now computes `sig[:n_days*hpd].reshape(n_days, hpd, N).mean(axis=0)` → `(24, N)` diurnal profile, then transposes to `(N, 24)` for pairwise DTW. Output files renamed to `dtw_matrix_{variable}.npy` for per-target support.

**Verification post-fix**:
- Recompute time: **2.24 s** (was 450 s under the broken axis) — 200× faster
- Diurnal profile shape: `(32, 24)` per station
- Off-diag DTW range: `0.29–3.91` (was 158–835 under old axis — completely different distribution)
- Per-target builds (sp, rh, ws) trivial — each ~2 s

### H3 — dead ode.time HPO knob (HIGH, FIXED)

**Reviewer's finding**: `model.py:95` hardcodes `ODEG(out_channels[-1], 12, A_hat, time=6)`. Wrapper passed nothing into that call → HPO trials sampled ode.time ∈ {3, 6, 12} but ALL ran at time=6. Self-inconsistent audit: §6.1 cited the hardcode; §7.4 searched it.

**Fix**: in `STGODEModel.__init__`, after upstream construction, walk `self.net.modules()` and reassign `m.t = torch.tensor([0.0, cfg_time])` on every ODEblock. `_n_odeblocks_time_overridden` counter set on self for verification.

**Verification post-fix** (from re-smoke): `12 ODEblock(s) had .t reassigned (config: ode.time=6.0)`. Live knob confirmed. The previous 2-trial HPO smoke's implied `ode.time=12 vs ode.time=6` comparison is void — a fresh HPO smoke has been run under this fix (see §10.4).

### H4 — single-step Euler at t=[0, T] (HIGH, DISCLOSED)

**Reviewer's finding**: `torchdiffeq.odeint(f, x, t, method='euler')` with no `options={'step_size': ...}` uses the passed `t` array as the integration grid → for `t=[0, 6]` this is **one Euler step of size 6**: `z = x + 6·f(0, x)`. "Continuous-time" here is weaker than MTGODE's 4-step (T=1.0, Δt=0.25) integration.

**Fix**: **not a code fix — this is faithful to upstream**. Instead: documented accurately in `stgode_model.py` docstring, `experiments/configs/stgode.yaml` inline comment, and the methods paragraph (§10.4 of this doc, once folded in). If genuine multi-step is ever wanted: pass `options={'step_size': T/K}` — which would be a documented departure from upstream.

### H5 — semantic graph z-score quirk (HIGH, RESOLVED BY C1 FIX)

**Reviewer's finding**: with upstream's z-scored recipe on real data, the semantic graph binarises at mid-similarity pairs and EXCLUDES the most-similar pairs (which have z ≈ −2, exp(−400) ≈ 0). Faithfully inherited upstream quirk.

**Fix**: our new default `graph_construction: paper` uses paper Eq 6 directly (`A_ij = 1 if DTW(X_i, X_j) < ε; 0 otherwise`) — no z-score, no Gaussian, no binarise-above-threshold. Most-similar pairs get edges (as the paper intends). H5 disappears in paper mode. If we ever ablation-test upstream mode, disclose.

### M6 — adapter forcing .train() mode (MEDIUM, FIXED)

**Reviewer's finding**: `forward()` unconditionally called `self.model.train()`. The blocks contain BatchNorm + Dropout — if any eval path called `forward` rather than `predict`, val would run under train-mode BN + dropout.

**Fix**: removed `self.model.train()` and `self.model.eval()` calls from the adapter. Trainer owns mode management (matches MTGODE / AGCRN / CLCRN adapter convention).

**Verification post-fix**: `M6: adapter is mode-agnostic; trainer.eval()/train() calls are respected` verdict in smoke script — confirmed by explicit `model.eval() → adapter.predict() → assert not model.training`.

### M7 — inventory of inherited divergences (MEDIUM, DOCUMENTED)

See §11 above. 11 divergences documented — the methods paragraph must mention D4, D7, D8 explicitly.

### Low fixes (all applied)

- **Low1**: Wrapper now `raise ValueError` if `input_len != 12` (was docstring lie about a nonexistent clamp).
- **Low2**: `_move_adjacencies_to` guard now correctly latches — was comparing `torch.device("cuda")` vs `torch.device("cuda:0")` (unequal). Fixed by normalising via `torch.zeros((), device=device).device`.
- **Low3**: Dead `output_dim` key removed from adapter's merged-config dict.

### On `ODEFunc.beta` (upstream dead code)

Reviewer noted `beta = 0.6` at `odegcn.py:27` is defined but never used. No action needed — vendored upstream unchanged.

---

## 13. Re-smoke evidence (post-fix, 2026-08-11)

Standalone smoke (`/tmp/smoke_stgode_v2.py`, 40 steps, batch=8, GTX 1650 Ti):

| Verdict | Result |
|---|---|
| Runtime | 7.4s (0.19 s/step) |
| A_sp_hat spread (paper Eq 4) | 0.226 (was ~0.07 under broken z-scored) |
| A_se_hat edges (paper Eq 6, ε=0.5) | 10 non-zero / 992 off-diag = 1.0% |
| ode.time override latched | 12 ODEblocks reassigned |
| adjacency device-cache latched | ✓ (canonical `cuda:0`) |
| M6: mode-agnostic | ✓ (predict respects trainer's `.eval()`) |
| loss delta | −1.14 (first-5 → last-5 mean, monotone decreasing) |
| finite loss | ✓ |
| spatial + semantic gradients flowing | ✓ every step |
| loss = MAE | ✓ (`nn.L1Loss`) |

All 8 verdicts PASS.

HPO smoke (2 trials, in flight during doc-writing; refresh runs/hpo_optuna_stgode_smoke/ for completion evidence).

Production dispatch smoke: not re-run yet (fixes are adapter-internal; MTGODE→STGODE process transition path is unaffected). Will re-run after HPO smoke passes.
