# Phase-1 SAWS Regression Campaign — 25-Station Network (N=25, 2013-2020)

**Purpose:** self-contained specification of what was actually implemented and shipped in the Phase-1 SAWS regression campaign that ran 2026-08-04 → 2026-08-07 and is shelved at [`results_saws_25node_phase1/`](../results_saws_25node_phase1/). Written so a paper-authoring agent can describe the data, models, training procedure, and infrastructure **without re-exploring the codebase**.

**Scope of this document:** implementation and methodology — *not* results interpretation. Numbers appear only where necessary to characterise the setup (e.g., sample counts, missingness thresholds, hyperparameter values).

**Critical caveat for the paper:** the codebase has moved on since the 25-node campaign. The main SAWS parser, MTGODE model wrapper, loss function, and several config files were changed on 2026-08-08 (MTGODE audit) and 2026-08-10 (HPO winner). **All file references below describe the state of the code at the time the 25-node campaign RAN**, with the current filenames + deltas noted explicitly. §11 (Known deviations) enumerates every one; §10 (File-paths reference) lists both then-current and now-current paths.

---

## 1. Executive summary

| | |
|---|---|
| **Scope** | 3 models × 4 targets × 10 seeds = **120 tasks**, each trained end-to-end and evaluated on a held-out chronological test window |
| **Models** | AGCRN (Bai et al., 2020), CLCRN (Lin et al., 2022), MTGODE (Jin et al., 2022) |
| **Targets** | temperature `t2m`, surface pressure `sp`, relative humidity `rh`, wind speed `ws` (Gaibie 2024 target set) |
| **Seeds** | 42–51 (project convention) |
| **Task type** | Univariate regression — model outputs all 6 SAWS variables; loss is computed only on the target variable's channel (Gaibie's protocol) |
| **Input / horizon** | input_len = 12 h, horizon = 48 h. Reported at h ∈ {3, 6, 9, 12, 24, 36, 48} by truncation of the direct 48-step forecast |
| **Training loss** | **MSE** (`nn.MSELoss`). Every `summary.json` in `results_saws_25node_phase1/runs/*/` carries `"loss_type": "MSELoss"` as ground truth. Not MAE — the uniform-MAE decision came 2026-08-08, after this campaign completed |
| **Optimiser / scheduler** | Adam, CosineAnnealingWarmRestarts (T_0=10, T_mult=2, eta_min=1e-6), grad-clip max_norm=5.0, early stopping patience 15 |
| **Data source** | SAWS Lekuba workbooks (`data/saws/extracted/data1.xlsx` + `data2.xls`), 2013-01-01 to 2020-12-31 |
| **Network** | 25 stations selected by ≤ 10% worst-feature missingness (Davidson 2022 criterion) |
| **Wall-clock** | Aug 4 evening → Aug 7 morning; ~72 h calendar, ~24 h GPU-time from first R via 6 concurrent trainings (K=3 × 2 lineages) |
| **Result** | 120 / 120 completions, zero lost seeds despite 1 hardware fault + 3 code bugs surfaced mid-flight |

---

## 2. Data

### 2.1 Raw source

Two long-form SAWS workbooks (Lekuba delivery), parsed by `preprocessing/saws_parser.py` **as of campaign time** (now renamed to [`preprocessing/saws_parser_25node_lekuba.py`](../preprocessing/saws_parser_25node_lekuba.py) — the file at that path is the parser that ran):

| File | Sheets | Columns | Notes |
|---|---|---|---|
| `data/saws/extracted/data1.xlsx` | 21 (one per station-group) | 8 (with header row): `StasName, Temperature, Rain, Humidity, WindDirection, WindSpeed, Pressure, DateT` | Each sheet is header + long-form (row per hour per station) |
| `data/saws/extracted/data2.xls` | 14 | 12 (header only on Sheet1): `CompNo, StasName, Lat, Lon, DateT, hour, T, R, H, WD, WS, P` | Sheets 2-14 have no header — parser applies Sheet1 schema by column position |

**Canonical rename at parse time** (lines 40-48 of the Lekuba parser):

| Raw | Canonical |
|---|---|
| Temperature | `t2m` |
| Rain | `tp` |
| Humidity | `rh` |
| WindDirection | `wd` |
| WindSpeed | `ws` |
| Pressure | `sp` |

**Time window (canonical hourly grid, parser line ~112):** 2013-01-01 00:00 → 2020-12-31 23:00 = **70,128 hours**.

### 2.2 Station selection (25 stations)

**Criterion (Davidson 2022):** each station retained iff its worst per-variable missingness inside the canonical window is ≤ 10 %. Parser constant `MISSINGNESS_THRESHOLD = 0.10` (line 53).

**Audit procedure:**
1. Filter each workbook to the canonical 2013–2020 window
2. Per-station: reindex to hourly grid, compute per-variable `missing_pct`
3. Compute `max_missing = max_over_variables(missing_pct)`
4. Retain if `max_missing ≤ 0.10`

**Explicitly dropped** (parser lines 57-61): PORT ELIZABETH - WK (data1: 1 month), LADYSMITH (data2: WindDir 13.78 %), WARDEN - HERITAGE (data2: 16.98 %).

**Final 25 stations (alphabetical, worst-feature missingness in parentheses):**

ALEXANDERBAAI (0.99 %), BEAUFORT-WES (0.37 %), BLOEMFONTEIN WO (1.93 %), CALVINIA WO (1.13 %), CAPE TOWN WO (0.65 %), CRADOCK-MUN (0.50 %), EAST LONDON WO (2.64 %), GARIEP DAM (3.13 %), GEORGE WO (0.33 %), GREYTOWN (0.76 %), KIMBERLEY WO (2.80 %), KING SHAKA INTERNATIONAL AIRPORT WO (2.61 %), KNELLPOORTDAM (3.38 %), KOKSTAD (1.18 %), KROONSTAD (1.01 %), MOOI RIVER (8.17 %), PORT ELIZABETH AWOS (1.08 %), PRINS ALBERT - SWARTRIVIER (4.69 %), QUEENSTOWN (5.21 %), REDELINGSHUYS-AWS (0.75 %), SUTHERLAND (4.07 %), UMTHATHA WO (6.43 %), UPINGTON WO (1.44 %), VAN ZYLSRUS (9.57 %), WORCESTER-AWS (0.52 %).

**Audit artefact:** `data/saws/audit/parse_report.csv` (per-station per-variable missingness).

### 2.3 Preprocessing pipeline

Orchestrated by [`preprocessing/pipeline.py::run_pipeline`](../preprocessing/pipeline.py). Steps 1-6:

**Step 1 — Parse.** Returns `(X, timestamps, coords, var_names)` where `X.shape = (70128, 25, 6)` float32 with NaNs at missing cells.

**Step 2 — Impute** ([`preprocessing/impute.py`](../preprocessing/impute.py)):
- `exclude_sparse_nodes(X, coords, exclusion_threshold=0.10)` drops any residual sparse nodes (no-op at N=25 — parser already filtered)
- `idw_impute(X, coords)` — spatial IDW per (variable, time):
  - Haversine great-circle distance matrix (R=6371 km, [`preprocessing/graph.py`](../preprocessing/graph.py) lines 39-51)
  - For each node i, `k=5` nearest neighbours (self excluded via inf), weights `w = 1/dist²` (`power=2.0`)
  - `X[t, i, var] = Σ(w_j × val_j) / Σ(w_j)` over non-NaN neighbours at time t
  - **Temporal fallback:** residual NaN (spatial IDW fails when all k neighbours are missing simultaneously) → per-(station, variable) forward-fill then backward-fill along time axis

**Step 3 — Chronological 60/20/20 split** ([`preprocessing/split.py::chronological_split`](../preprocessing/split.py)):

| Split | Hours | % | Date range |
|---|---|---|---|
| Train | 42,076 | 60 | 2013-01-01 00:00 → 2017-10-20 03:00 |
| Val | 14,025 | 20 | 2017-10-20 04:00 → 2019-05-27 12:00 |
| Test | 14,027 | 20 | 2019-05-27 13:00 → 2020-12-31 23:00 |

**Step 4 — Extreme-event labels** ([`preprocessing/labels.py`](../preprocessing/labels.py)) — used for stratified evaluation, not for training in Phase-1:
- Thresholds computed on training only: `thresholds = percentile(X_train, 95, axis=0)`, shape (N=25, D=6)
- Binary label: `label = (X > thresholds).astype(int8)`
- **`tp` wet-hour override** (labels.py lines 57-99): where dry-hour fraction > 90 % AND wet hours ≥ 20 at a station, the naive 95th-percentile is overridden by the 95th percentile of *wet* hours only (dry hour = `tp < 1e-4 m/h`). **At the 25 stations, `tp_wet_trigger.npy` = all True** — every station had > 90 % dry hours and used the wet-hour rule
- Cached as `thresholds.npy` (25, 6) + `tp_wet_trigger.npy` (25,)

**Step 5 — Z-score normalisation** ([`preprocessing/normalise.py::normalise_all_splits`](../preprocessing/normalise.py)):
- Per-(station, variable) mean/std computed on train only; broadcast-safe shape (1, 25, 6)
- `X_norm = (X - mean) / std`; `std = 1` where 0 (avoid division by zero)
- Applied to train, val, test with train's stats

**Step 6 — Graph adjacency** ([`preprocessing/graph.py::build_adjacency_matrix`](../preprocessing/graph.py)):
- Haversine distance matrix (25, 25) km
- Gaussian kernel: `A_ij = exp(-D_ij² / (2σ²))` with `σ = sigma_km = 200`
- Hard sparsity cutoff: `A_ij = 0` where `D_ij ≥ distance_threshold_km = 500`
- Zero diagonal, then row-normalise (`A_ij / row_sum`)
- **Resulting structure at N=25:** 25×25, 256 non-zero entries (59 % sparse), min weight 0.0103 / max 0.6939. Nearest-neighbour distances 60.2–449.1 km (all within 500 km cutoff). Furthest disconnected pair 1422 km (ALEXANDERBAAI ↔ KING SHAKA).

**Post-processing — outlier clipping:** after normalisation, `X = clip(X, -clip_sigma, +clip_sigma)` with `clip_sigma = 10.0`. Reason (pipeline.py comment): "ERA5 total precipitation (tp) is extremely right-skewed — raw z-scores reach 57σ, causing NaN in model activations after a few gradient steps. 10σ retains all physically meaningful variance while preventing instability." SAWS `tp` is even more skewed than ERA5 (`saws_lekuba.yaml` comment line 32).

### 2.4 Sliding-window Dataset ([`preprocessing/dataset.py::STGNNDataset`](../preprocessing/dataset.py))

- Each sample: `(x, y)` where `x = X[t : t+input_len]` shape (12, 25, 6) and `y = X[t+input_len : t+input_len+horizon]` shape (48, 25, 6)
- Sample count `n_samples = T - input_len - horizon + 1`
- Batch layout: DataLoader yields `(B, T, N, D)` for x and `(B, H, N, D)` for y
- **Univariate slicing** (dataset.py line 117): if `target_variable_index` is set in config (which it was for every 25-node run, target ∈ {t2m, sp, rh, ws}), `y` is sliced to a single channel `y = y[..., idx:idx+1]` shape (B, H, N, 1). The model still outputs all 6 channels; the trainer slices `pred` to match at loss time (see §4.1)

### 2.5 Sample counts

| Split | Hours | Sliding-window samples (input=12, horizon=48) |
|---|---|---|
| Train | 42,076 | 42,017 |
| Val | 14,025 | 13,966 |
| Test | 14,027 | 13,968 |

### 2.6 Preprocessing cache

Written under `preprocessing/cache/saws/` as npy + meta.json. Cache key includes filename stems of raw workbooks + preprocessing config values (input_len, horizon, split ratios, distance/sigma thresholds, derive_variables flag). The `compute_data_hash` function was later hardened (2026-08-08) to apply dataset-aware defaults so equivalent-but-differently-expressed configs hash the same; the 25-node cache was generated under the pre-fix hash logic, but it produces identical output arrays (only the hash marker itself was different).

---

## 3. Model implementations

Three models were trained. All three consume the same input tensor `(B, 12, 25, 6)` and emit `(B, 48, 25, 6)`; univariate slicing to `(B, 48, 25, 1)` happens at loss time.

### 3.1 AGCRN (Bai et al., 2020)

**Files:**
- Adapter: [`models/baselines/agcrn_adapter.py`](../models/baselines/agcrn_adapter.py)
- Wrapper: [`models/baselines/agcrn_model.py`](../models/baselines/agcrn_model.py)
- Vendored upstream: [`models/baselines/AGCRN/`](../models/baselines/AGCRN/)

**Architecture** (upstream `AGCRN/model/AGCRN.py` + `AGCRNCell.py` + `AGCN.py`):
- **Adaptive graph via node embeddings** — learned `(N, embed_dim=10)` tensor; adjacency at each layer computed as `softmax(ReLU(E · Eᵀ))` (`AGCN.py` line 15). No use of the pipeline's static adjacency (adapter explicitly documents this at agcrn_model.py:9-10).
- **AVWGCN (Adaptive Vertex-Wise Graph Convolution)** — Chebyshev polynomial bases up to order `cheb_k=2` combined via einsum with a per-node weights pool of shape `(embed_dim, cheb_k, in_dim, out_dim)` (`AGCN.py` lines 16-25).
- **Recurrent core:** two-layer stack of `AGCRNCell` (GRU-style: `z, r` from a gate AVWGCN; `h_new = r*h + (1-r)*tanh(update AVWGCN)`), unrolled over the input's time dimension.
- **Output head:** final encoder hidden `(B, 1, N, hidden_dim)` → `Conv2d(1, horizon×output_dim, kernel=(1, hidden_dim))` → reshape to `(B, horizon, N, D)`.

**Adapter quirk (`agcrn_model.py` lines 52-57):** upstream `AVWGCN` uses `nn.Parameter(torch.FloatTensor(...))` which leaves uninitialised memory (NaN risk). The adapter re-initialises every `weights_pool` with `kaiming_uniform_` and every `bias_pool` with `zeros_` before any forward pass — this is absent in the official AGCRN repo.

**Hyperparameters as used in the 25-node campaign** ([`experiments/configs/agcrn.yaml`](../experiments/configs/agcrn.yaml) as locked 2026-08-04):

| Key | Value | Source |
|---|---|---|
| `model.hidden_dim` (→ upstream `rnn_units`) | 64 | Gaibie 2024 Table 1 (HPO-optimal on 45-station SAWS) |
| `model.num_layers` | 2 | Gaibie 2024 Table 1 |
| `model.embed_dim` | 10 | Bai 2020 PEMSD4 default |
| `model.cheb_k` | 2 | Bai 2020 default |
| `training.learning_rate` | 0.01 | Gaibie 2024 Table 1 |
| `training.batch_size` | 64 | Gaibie 2024 Table 1 |
| `training.epochs` | 40 | Gaibie 2024 Table 1 |
| `data.input_len` | 12 | Gaibie's HPO optimum from {12, 24, 48, 72, 168} |
| `data.horizon` | 48 | Extension beyond Gaibie's 24 h (Moodley directive) |

**Approximate parameter count:** ~650k–800k trainable parameters at the 25-node config values.

**Deviations from Bai et al. (2020):** scheduler (they use constant-lr Adam; we use CosineAnnealingWarmRestarts uniformly across all three models — disclosed as harness choice); `embed_dim=10` is Bai's PEMSD4 default (307-node dataset), inherited without re-tuning.

### 3.2 CLCRN (Lin et al., 2022) — recurrent variant

**Files:**
- Adapter: [`models/baselines/clcrn_adapter.py`](../models/baselines/clcrn_adapter.py)
- Wrapper: [`models/baselines/clcrn_model.py`](../models/baselines/clcrn_model.py)
- Vendored upstream: [`models/baselines/CLCRN/`](../models/baselines/CLCRN/)

**Variant selection** (`clcrn_model.py:79`): `from model.clcnn.recurrent.seq2seq_model import CLCRNModel as _CLCRN` — the **recurrent** variant. Upstream also ships an attention variant (`model/clcnn/attention/`) whose `hidden_units` parameter has no analogue in the recurrent code path we import (see 3.5 in HYPERPARAMETERS.md for the three Gaibie'24 CLCRN rows that don't apply).

**Architecture:**
- **Spherical k-NN graph** built from station lat/lon via `KernelGenerator` (upstream `experiments/dataloader.py`). For each node, k=25 nearest neighbours by great-circle distance; local tangent-plane frames + per-edge angular density weights + geodesic lengths derived.
  - Adapter coordinate quirk (`clcrn_adapter.py:57-58`): pipeline stores `(lat, lon)` but CLCRN expects `(lon, lat)` — adapter swaps columns.
- **Non-buffer auxiliary tensors** (`clcrn_adapter.py:25-36`): upstream stores geometry as plain attributes, not `register_buffer()`. `nn.Module.to(device)` skips them. Adapter's `_move_aux_tensors()` walks all submodules and manually transfers Tensor attributes to device before every forward.
- **CLConv (Conditional Local Convolution):** MLP-modulated spatial convolution over the k-NN neighbourhood, anisotropic in angle and geodesic distance.
- **Encoder–Decoder recurrent** (upstream `recurrent/encoder.py`, `decoder.py`): encoder consumes 12 input steps via CLConv → GRU-style cell; decoder autoregressively generates 48 output steps. Curriculum learning disabled (`use_curriculum_learning=False`).

**Key adapter mapping (`clcrn_model.py:86-100`):** our config's `hidden_dim` → upstream's `rnn_units`. Verified against upstream at `models/baselines/CLCRN/model/clcnn/recurrent/seq2seq.py`.

**Hyperparameters as used in the 25-node campaign** ([`experiments/configs/clcrn.yaml`](../experiments/configs/clcrn.yaml) as locked 2026-08-04):

| Key | Value | Source |
|---|---|---|
| `model.hidden_dim` → `rnn_units` | **32** | Gaibie 2024 Table 1 (corrected 2026-08-04 from earlier 64) |
| `model.num_layers` | 2 | Gaibie 2024 Table 1 |
| `model.embed_dim` | 16 | Lin 2022 `seq2seq.py:15` default |
| `model.k_neighbors` | 25 | Lin 2022 `dataloader.py:67` default |
| `model.lck_structure` | [4, 8] | Lin 2022 `seq2seq.py:14` default |
| `model.max_view` | 2 | Lin 2022 `seq2seq.py:6` default |
| `training.learning_rate` | **0.05** | Gaibie 2024 Table 1 (aggressive; at boundary of her {0.0003, …, 0.05} search) |
| `training.batch_size` | 32 | Gaibie 2024 Table 1 |
| `training.epochs` | 40 | Gaibie 2024 Table 1 |
| `data.input_len`, `data.horizon` | 12, 48 | Uniform across all three models |

**Approximate parameter count:** ~25k–40k trainable parameters (dominated by encoder + decoder GRU cells at `rnn_units=32`).

**Note on `k_neighbors=25` at N=25:** the graph is effectively fully connected (each node's 25 nearest includes itself + all others). At larger N this would become sparse and would need re-tuning; flagged in `specs/HYPERPARAMETERS.md`.

**Known stability flag for `lr=0.05`:** the 50× ratio to our earlier 0.001 is documented in HYPERPARAMETERS.md §CLCRN under a "high-lr stability" note. `gradient_clip_max_norm=5.0` is the safety net. The value survived campaign-scale training.

### 3.3 MTGODE (Jin et al., 2022) — **PRE-AUDIT VERSION**

**Files:**
- Adapter: [`models/continuous_time/mtgode_adapter.py`](../models/continuous_time/mtgode_adapter.py)
- Wrapper: [`models/continuous_time/mtgode_model.py`](../models/continuous_time/mtgode_model.py) *(file exists but contents were rewritten on 2026-08-08 to fix Findings 0, 2, 5 — see §11)*
- Vendored upstream: [`models/continuous_time/MTGODE/`](../models/continuous_time/MTGODE/) (model.py, layer.py)

**The 25-node MTGODE runs were compromised by three implementation bugs** discovered and fixed on 2026-08-08 (after the campaign ended). The bugs are load-bearing enough that any paper reporting these results must disclose them explicitly (see §11). What follows describes the **as-run** state.

**Architecture** (upstream `MTGODE/model.py` + `layer.py`):
- **Coupled ODEs.** Outer temporal ODE (`ODEBlock`, `model.py:138`) integrates the full spatio-temporal state via `torchdiffeq.odeint` (or `odeint_adjoint`) using method_1=`euler`, T=1.0, Δt=0.25 (4 steps). Inner spatial ODEs (inside STBlock) integrate CGP (Continuous Graph Propagation) with method_2=`euler`, T=1.0, Δt=0.25.
- **STBlock (`layer.py:44-94`):** dilated inception temporal blocks with kernels [2, 3, 6, 7], tanh-sigmoid gating (lines 71-75), then two CGP layers — one with the forward adjacency, one with the transpose (bidirectional; line 82). Receptive-field padding (line 84).
- **CGP (`layer.py:412-438`):** `CGPFunc.forward(t, x)` computes `f = 0.5 · α · (Â_norm·x − x)` (line 370). Paper Eq. 6 is `dH/dt = (Â − I)H`, which requires α = 2.0 to recover the full vector field.
- **Graph constructor (`layer.py:168-223`):** two node embeddings `emb1, emb2 : (N, node_dim)`, two linear transforms `lin1, lin2 : (node_dim, node_dim)`, antisymmetric outer product, tanh, top-k sparsification (k per node), returned as `(N, N)`.

**Hyperparameters as USED IN THE 25-NODE CAMPAIGN** (pre-audit configuration):

| Key | Value in 25-node runs | Provenance / status |
|---|---|---|
| `model.hidden_dim` | 64 | Jin §5.2 conv_channels — **also silently used as node_dim** (audit Finding 2) |
| `model.dropout` | 0.3 | Jin §5.2 |
| `model.graph_constructor.k` | **10** | 40 % density heuristic at N=25 (scaled from MTGNN's k=20 at N=207 ≈ 10 % density) |
| `model.graph_constructor.alpha` (tanhα) | 3 | Jin / MTGNN default |
| `model.graph_constructor.node_dim` | **not exposed — silently = hidden_dim = 64** | Paper §5.2 pins node_dim = 40. **Audit Finding 2.** |
| `ode.method_1`, `method_2` | euler / euler | Jin §5.2 (METR-LA) |
| `ode.time_1`, `step_size_1` | 1.0, 0.25 | Jin §5.2 |
| `ode.time_2`, `step_size_2` | 1.0, 0.25 | Jin §5.2 |
| `ode.cgp_alpha` | **hardcoded 1.0** (not in config) | Paper Eq. 6 requires 2.0 — code halved the vector field. **Audit Finding 5.** |
| `ode.adjoint` | **true** | Chosen for memory efficiency, but under `odeint_adjoint` the graph learner's parameters (attached via `setGraph()`) are outside `func.parameters()` and receive **grad = None** every backward step. The adaptive adjacency remained at random init for the entire 25-node campaign. **Audit Finding 0.** |
| `training.learning_rate` | 0.001 | Jin §5.2 (Adam, lr = 1e-3) |
| `training.batch_size` | 64 | Jin §5.2 |
| `training.epochs` | 100 (early-stopping cap; not 200) | Cap in place at the time of the 25-node run |
| `data.input_len`, `data.horizon` | 12, 48 | Uniform |

**Adapter's `_purge_shared_model_module()` (`run_worker.py` lines 74-84):** each vendored upstream (AGCRN, CLCRN, MTGODE) imports a bare-name `model` package via `sys.path` insertion. Within a single Python process, once one adapter's `model` package is in `sys.modules`, a subsequent adapter hits the wrong one. This function purges `sys.modules['model']` and any `model.*` entries + restores `sys.path` before every adapter build. **Note:** this fix was deployed *during* the campaign after 4 procs died on link 1 with `ModuleNotFoundError` (see §7 bug 4); for links 2+ every task ran with the purge active.

**Approximate parameter count:** ~100k–150k trainable parameters at the 25-node config (dominated by `end_conv` when `H × D = 288` output channels).

### 3.4 Model output convention

All three adapters return `(B, H, N, D) = (B, 48, 25, 6)` in normalised z-space. The trainer slices to `(B, 48, 25, 1)` at loss time for the univariate target (see §4.1).

---

## 4. Training harness

Implemented in [`training/trainer.py`](../training/trainer.py) (`Trainer` class). One instance per task; task-specific output dir under `runs/<task_id>/`.

### 4.1 Per-epoch loop (`Trainer._train_epoch`, lines 384-426)

```python
for x, y in self.train_loader:
    x, y = x.to(device), y.to(device)
    self.optimiser.zero_grad()
    pred = self.adapter.forward(x, self.adj)               # (B, H, N, D)

    # Univariate slice: model outputs all D channels; loss only on target channel
    tv_idx = getattr(self.train_loader.dataset, "target_variable_index", None)
    if tv_idx is not None and pred.shape[-1] != y.shape[-1]:
        pred = pred[..., tv_idx:tv_idx + 1]                # (B, H, N, 1)

    loss = self.loss_fn(pred, y)                           # nn.MSELoss (25-node) — see §4.3
    loss.backward()
    clip_gradients(self.adapter.model, max_norm=5.0)       # returns pre-clip norm for logging
    self.optimiser.step()
```

### 4.2 Optimiser + scheduler + grad clipping

- **Optimiser** (`Trainer.__init__` lines 161-164): `torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)`
- **Scheduler** ([`training/scheduler.py::build_scheduler`](../training/scheduler.py)): `CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)` — **uniform across all three models**. Not from any of the model papers (they all use constant-lr Adam); disclosed as a harness choice for internal comparability.
- **Gradient clip** (`training/scheduler.py::clip_gradients`): `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)` after every backward; returns the pre-clip norm which is logged per epoch.

### 4.3 Loss — **MSE for the 25-node campaign**

Constructed via `cli.py::_build_loss(task="regression", ...)` → **`mse_loss()`** = `nn.MSELoss()` (unweighted, no masking — the pipeline's IDW imputation fills all NaN before training, so no missing-value mask is needed).

**Ground-truth artefact:** every `results_saws_25node_phase1/runs/*/summary.json` carries `"loss_type": "MSELoss"`. This is the definitive record of what ran.

The uniform-MAE loss switch (`nn.L1Loss`) landed on 2026-08-08 as part of the post-audit methodology alignment (matches MTGODE §4.4, AGCRN & CLCRN repo defaults). **It does not apply to the 25-node campaign** and any paper section on the 25-node results must say "MSE".

### 4.4 Early stopping + divergence guard

- **Early stopping:** patience = 15 (uniform). Fires when `patience_counter ≥ patience`. Checkpoint (`checkpoint.pt`) saved only when `val_loss < best_val_loss`.
- **Divergence guard** (`training/trainer.py:277-296`): a `TrainingDiverged` exception is raised if `train_loss` or `grad_norm` is non-finite for 3 *consecutive* epochs. A `FAILED_DIVERGED` marker file is written under `runs/<task_id>/`, the task is **not** retried, and the worker moves on.

### 4.5 Seed management

- **Base seed** set at task start via [`training/seed.py::seed_everything(seed)`](../training/seed.py), covering torch (CPU + CUDA), numpy, python random, CUBLAS workspace.
- **Per-epoch re-seed** (`Trainer._seed_epoch`, lines 693-709): deterministic `s = SHA256(f"{base_seed}:{epoch}") & 0xFFFF_FFFF`, re-seeded before each epoch's forward pass. Ensures a resumed run at epoch N produces identical batches to an uninterrupted run at epoch N.

### 4.6 Checkpointing

- **`state.pt`** (atomic write via `os.replace`): full training state — `epoch`, model + optimizer + scheduler state_dicts, `best_val_loss`, `patience_counter`, `diverged_streak`, `history`, `seed`, and RNG state for torch/CUDA/numpy/python.
- **`checkpoint.pt`**: best-val model weights only, used at test-eval time.
- **Resume:** on task start, if `state.pt` exists, the trainer loads it, restores RNG state, and appends `"# RESUMED from epoch {N}"` to `train.log`.

### 4.7 Graceful stop on SIGUSR1

Trainer accepts a `stop_check` callback. In production this is set by [`scripts/run_worker.py`](../scripts/run_worker.py) to poll a `_STOP_REQUESTED` flag set by a SIGUSR1 handler (lines 103-113). SLURM's `#SBATCH --signal=B:USR1@1200` delivers SIGUSR1 20 minutes before walltime; the trainer checkpoints and exits cleanly at the next epoch boundary. `worker.sh` re-locates the Python child of the apptainer starter via `pgrep -P $starter_pid -f python3` and signals it directly (apptainer's own starter does *not* forward signals — probed 2026-08-05).

### 4.8 Test evaluation (`Trainer._evaluate_test`, lines 561-668)

After early stopping fires, the trainer loads `checkpoint.pt` (best-val weights) and runs the full test loader in `no_grad()` mode:
1. Concatenate all `pred`, `target` batches → `(samples, H, N, D)`
2. Univariate slice both `pred` and `target` to (samples, H, N, 1) if `target_variable_index` set
3. Build extreme mask from thresholds (§4.9)
4. Call [`evaluation.regression.compute_all(...)`](../evaluation/regression.py) with `mean`, `std`, `thresholds`, `var_names`, `vpt_epsilon=0.3`
5. Record `inference_ms_per_sample` via [`evaluation/timing.py`](../evaluation/timing.py)
6. Write `test_metrics.json` and `per_station.npz`

### 4.9 Metrics (physical units)

Everything is denormalised before reporting via `(pred × std + mean)`. `test_metrics.json` fields for each target variable:

| Metric | Definition |
|---|---|
| `mae` | Mean absolute error, pooled over (samples, H, N) |
| `rmse` | Root mean squared error |
| `mse` | Mean squared error |
| `smape` | Symmetric MAPE (stable at near-zero targets) |
| `mae_extreme` | MAE conditioned on target > 95th-percentile threshold (per station, per variable) |
| `mae_nonextreme` | MAE on the non-extreme complement |
| `peak_intensity_error` | Error at the peak value within each 48-h forecast window |
| `nrmse_per_horizon` | Length-48 array of per-horizon NRMSE in z-space (normalised by training std) |
| `vpt` | Valid Prediction Time: first horizon index at which `nrmse_per_horizon ≥ 0.3`. If never crosses, returns H. |
| `mae_pooled_norm` | Pooled MAE in z-space (internal monitoring) |
| `inference_ms_per_sample` | Per-sample wall-clock at inference |
| `units` | Always `"physical"` for 25-node runs |
| `vpt_epsilon` | 0.3 (Pathak 2018 convention: fraction of climatology std) |

**Per-station stratification** (`per_station.npz`): shape `(N=25, D=1)` for the univariate regime; contains `mae`, `rmse`, `mae_extreme`, `mae_nonextreme`, `peak_intensity_error` per station.

**Rollout evaluation** (`evaluation/rollout.py`): NOT run for the 25-node campaign — the block-autoregressive rollout requires `input_len == horizon`, which was violated here (12 vs 48). The direct 48-step forecast is the sole evaluation path.

### 4.10 Baselines (seed-independent, computed once)

[`scripts/compute_baselines.py`](../scripts/compute_baselines.py) reads the preprocessed cache and writes into `results_saws_25node_phase1/baselines/`:

- **Persistence** (`regression-persistence.json`): ŷ(t+h) = y(t + input_len − 1) for h = 1…48 (last observed step repeated across the horizon)
- **Climatology** (`regression-climatology.json`): ŷ(t+h) = training mean per (station, variable), constant across time and horizon

Both compute the *same metric set* as the trained models (§4.9) for direct comparison in physical units on the exact same test split. Command:
```bash
python scripts/compute_baselines.py --dataset saws --skip-classification
```

---

## 5. Infrastructure — HPC parallel worker system

Full operator's guide at [`specs/HPC_Parallel_Worker_System.md`](HPC_Parallel_Worker_System.md). Summary of what shipped for the 25-node campaign:

### 5.1 Concurrency model

- Slurm partition `l40s`, account `l40sugrd`; hard cap `--time=24:00:00` (association MaxWall)
- `MaxTRESPU=gres/gpu:l40s=2` → max 2 concurrent GPU jobs per user; `MaxSubmitPU=5` submit slots
- **2 lineages** submitted initially; each self-perpetuates
- **K = 3 co-tenant processes per GPU** (48 GB L40S / ~2-3 GB per model at N=25 → fits ~15× over; K=3 measured throughput multiplier ~1.42 vs K=1)
- Effective concurrent trainings: **6** (2 lineages × 3 procs)
- Thread caps per process: `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=2`

### 5.2 Manifest + claim system

- **Manifest** (`runs/manifest.json`): ordered list of 120 task IDs, format `{model}_{target}_s{seed}`
- **Ordering** ([`orchestration/manifest.py::build_manifest`](../orchestration/manifest.py)): round-robin by seed (outer loop), slow-first models per round (`clcrn, mtgode, agcrn`), then targets (`t2m, sp, rh, ws`). Design intent: first 12 tasks form a complete 3×4 preliminary matrix at seed 42; subsequent rounds increment N uniformly → partial results are always usable.
- **Model timing from pilot 1181304 (L40S):** CLCRN ~234 s/epoch (autoregressive at H=48), MTGODE ~44 s/epoch, AGCRN ~27 s/epoch — dictates slow-first ordering
- **Atomic claim** (`orchestration/claim.py::try_claim`): `mkdir claims/<task_id>/` is atomic on POSIX. Exactly one worker succeeds per task. Meta.json inside records `jobid, pid, host, ts`.
- **Sweep** (`stale_claim_sweep`): before each new claim, read `runs/live_jobids.txt` and release claim dirs whose jobid is not in the live set (auto-recovery from process death).

### 5.3 Chain design (`slurm/worker.sh`)

Each `worker.sh` invocation:
1. Probes `squeue` in PATH (fail-loud); writes initial `runs/live_jobids.txt` snapshot; starts background 5-min refresher (per-lineage tmp file + `mv` for atomicity)
2. Runs `python scripts/manifest_incomplete.py`; if any task lacks both `summary.json` and `FAILED_DIVERGED`, submits successor: `sbatch --dependency=afterany:$SLURM_JOB_ID slurm/worker.sh`. `afterany` means the successor becomes eligible when this link exits for *any* reason (walltime, crash, cancel) — this is the crash-recovery mechanism.
3. Launches K = 3 `singularity exec --nv` processes in parallel, staggered 20 s apart (avoids simultaneous cache-load + CUDA-context spikes), each running `python3 -u scripts/run_worker.py --manifest ... --claims-dir ... --runs-dir ... --wallclock-margin 1500`
4. On SIGUSR1 (delivered 20 min before walltime by `#SBATCH --signal=B:USR1@1200`), the bash trap locates each child's python PID via `pgrep -P $starter_pid -f python3` and signals it directly (apptainer starter does not forward signals — probed 2026-08-05)
5. Re-wait loop until all children reap; kill background refresher; exit 0

**Chain terminates** when either `MAX_LINKS = 10` reached or `manifest_incomplete.py` returns exit 1 (nothing left to do).

### 5.4 Container

`container.def` builds `container.sif` (~11 GB, lives in `/scratch/jrvpra001/weatherflow/`); base image `nvcr.io/nvidia/pytorch:24.01-py3` (validated for L40S sm_89). `--nv` binding mandatory for GPU visibility inside the container.

**Runtime bind**: `singularity exec --nv --bind /scratch/jrvpra001:/scratch/jrvpra001 --env PYTHONPATH=/scratch/jrvpra001/pypkgs $CONTAINER python3 -u scripts/run_worker.py ...`

### 5.5 Per-task command (as run inside the container)

The effective 25-node-campaign command per process was:
```bash
python3 -u scripts/run_worker.py \
    --manifest /scratch/.../WeatherFLOW/runs/manifest.json \
    --claims-dir /scratch/.../WeatherFLOW/runs/claims \
    --runs-dir  /scratch/.../WeatherFLOW/runs \
    --wallclock-margin 1500
```

`run_worker.py` then loops: `next_incomplete_task → parse_task_id → load_config(model) → run_pipeline (cache hit) → _build_loss → load_adapter (with sys.modules purge) → seed_everything → adapter.build_model → Trainer(...) → trainer.train(stop_check=...) → write summary.json`.

---

## 6. Results tree — [`results_saws_25node_phase1/`](../results_saws_25node_phase1/)

```
results_saws_25node_phase1/
├── README.md                          # campaign summary
├── sync_from_hpc.sh                   # one-shot rsync from /scratch/.../runs → local
│
├── runs/                              # 120 task directories, one per (model, target, seed)
│   └── <task_id>/                     # e.g. mtgode_t2m_s42
│       ├── summary.json               # completion marker; loss_type, stopped_epoch, histories, test_metrics
│       ├── test_metrics.json          # denormalised metrics (see §4.9)
│       ├── per_station.npz            # (N=25, D=1) per-station stratification arrays
│       ├── state.pt                   # full training state (resumable)
│       ├── checkpoint.pt              # best-val model weights only
│       └── train.log                  # per-epoch train/val/grad_norm; RESUMED markers
│
├── phase1_regression/                 # aggregated per-model summaries
│   ├── agcrn_regression_summary.json
│   ├── clcrn_regression_summary.json
│   ├── mtgode_regression_summary.json
│   └── per_station_across_seeds.json  # geographic aggregation across seeds
│
├── baselines/                         # seed-independent
│   ├── regression-persistence.json
│   └── regression-climatology.json
│
├── analysis/                          # findings, calibration notes (kept empty at shelf time)
├── archive/                           # retired sub-experiments (empty at shelf time)
└── plots/
    ├── SUMMARY.md                     # auto-generated markdown: model×target MAE ± std vs baselines
    ├── mae_bar.png                    # per-model per-target MAE with baseline lines
    └── val_convergence.png            # per-model val-loss curves across seeds
```

### 6.1 `summary.json` schema (per task)

```json
{
  "task_id": "mtgode_t2m_s42",
  "model": "mtgode",
  "target": "t2m",
  "seed": 42,
  "stopped_epoch": 18,
  "train_loss": [ ... per-epoch ... ],
  "val_loss":   [ ... per-epoch ... ],
  "grad_norm":  [ ... per-epoch ... ],
  "test_metrics": { <as §4.9> },
  "loss_type": "MSELoss",
  "elapsed_seconds": 2093.02,
  "completed_ts": 1786003345.0
}
```

`loss_type` is the definitive artefact for what training objective ran. Presence of `summary.json` = task complete; no other check needed.

### 6.2 Aggregation (`scripts/analyze_saws25.py`)

Walks `runs/*/summary.json`, groups by model, then by target, and emits per-`(model, target)` aggregate:
- `n_seeds`, `seeds`, `stopped_epoch{mean, min, max}`, `best_val_loss{mean, std}`
- `test_metrics.<metric>{mean, std, median, n, per_seed}` with Bessel-corrected std (`ddof=1`)

Also produces `per_station_across_seeds.json` (geographic aggregation) and the two plots in `plots/`. Idempotent — safe to re-run whenever new seeds land.

---

## 7. In-flight bugs during the campaign (survived)

Documented in [`specs/HPC_Parallel_Worker_System.md`](HPC_Parallel_Worker_System.md) §10; summarised here because a paper's methods/limitations section may need to reference them:

| # | When | Symptom | Root cause | Fix |
|---|---|---|---|---|
| 1 | 2026-08-05 14:08 | 2 jobs died in 8 s with CUDA `uncorrectable ECC error` on srvrocgpu014 | Hardware fault on that node's GPU | `--exclude=srvrocgpu014` in `worker.sh` |
| 2 | 2026-08-05 evening | 2 fresh workers died in <10 s at `set -u` | Observability echo referenced `$SLURM_TIMELIMIT` (not a real SLURM env var) | Replaced with literal `24h(sbatch-fixed)` string |
| 3 | 2026-08-05 | 36 h `--time=` submissions rejected | Association-level `MaxWall=24h` (separate from QOS/partition caps) | Locked to `--time=24:00:00` per `sacctmgr` audit |
| 4 | 2026-08-06 10:03 | 4 procs died on MTGODE→AGCRN transition with `ModuleNotFoundError` | Each vendor's `model/` package collides in `sys.modules` inside one Python process | `_purge_shared_model_module()` in `run_worker.py` before every adapter build; deployed link 2+ |
| 5 | 2026-08-06 | Silent claim leak: sweep spared all claims, 82 dead-jobid claims accumulated | `squeue` not in singularity container's PATH → `subprocess.check_output` returned `_SQUEUE_UNKNOWN` sentinel; sweep interpreted as "unknown, be safe, spare all" | Wrapper (compute node, squeue-available) writes `runs/live_jobids.txt` every 5 min via atomic tmp+mv; container reads file with staleness guard; fail-loud probe on wrapper startup. Manual head-node sweep at 14:03 SAST recovered the leak |
| 6 | ongoing | 1 CLCRN task (`clcrn_rh_s45`) stuck-at-init | Seed sensitivity at Gaibie's aggressive `lr=0.05` (grad-clip max_norm=5.0 was the safety net; this seed still diverged) | Reported honestly in the results without a `FAILED_DIVERGED` marker |

**Lesson codified as the "environment-probe rail" (`specs/HPC_Parallel_Worker_System.md` §7):** any external command or capability the code depends on must be probed in its actual execution context and fail loud if absent. Retroactively applied to apptainer signal forwarding, association MaxWall, and container-squeue.

---

## 8. Known deviations & caveats for the paper's methods / limitations section

The paper-writing agent must include these when describing the 25-node campaign:

### 8.1 MTGODE audit findings — the 25-node MTGODE runs are compromised

Three implementation issues in the MTGODE wrapper were found and fixed on **2026-08-08, after the 25-node campaign ended**. They apply to every MTGODE task in `results_saws_25node_phase1/runs/mtgode_*` (40 tasks: 4 targets × 10 seeds):

- **Finding 0 — adjoint mode severed the graph learner.** `ode.adjoint=true` at 25-node meant `torchdiffeq.odeint_adjoint` was used. Its backward pass only differentiates parameters listed in `func.parameters()`. The graph learner (`gc.emb1/emb2/lin1/lin2`) is attached to the ODEBlock via `setGraph()` but lives in `STBlock`, nested inside `ODEFunc.stnet` — outside `func.parameters()`. Every backward step: `gc.*.weight.grad is None`. Adam skipped these parameters. **The adaptive adjacency stayed at random initialisation for the entire 25-node MTGODE campaign.** Each seed produces a different frozen random graph.
- **Finding 2 — node_dim silently = hidden_dim.** The graph-learner embedding dimension is a distinct paper hyperparameter (Jin §5.2: `node_dim = 40`, via MTGNN inheritance). At 25-node the wrapper read `node_dim = int(cfg.get("hidden_dim", 40))` which conflated it with `conv_channels = 64`. All MTGODE runs used `node_dim = 64`, giving the graph learner 60 % more parameters than the paper's design but at incorrect proportions.
- **Finding 5 — CGP coupling α = 1.0 instead of 2.0.** CGPFunc computes `f = 0.5 · α · (Âx − x)`. Paper Eq. 6 `dH/dt = (Â − I)H` requires α = 2.0. The wrapper passed α = 1.0 hardcoded. Effect: the spatial vector field was halved, equivalent to integrating CGP to T = 0.5 instead of the configured T = 1.0.

**Recommended paper disclosure:** "The 25-station SAWS phase-1 MTGODE results were trained under three implementation issues subsequently identified in an external audit (2026-08-08): (i) `torchdiffeq.odeint_adjoint` structurally excluded the adaptive-graph learner from gradient flow, leaving the adjacency at random initialisation; (ii) the graph-learner embedding dimension was conflated with the convolutional hidden width; (iii) the Continuous Graph Propagation coefficient was hardcoded at half its paper value. These results are archived and reported under the label `MTGODE (pre-audit)`; corrected runs at N = 32 are reported separately and not pooled."

AGCRN and CLCRN have no known audit findings and their 25-node results are valid under the reported configuration.

### 8.2 Training loss: MSE (not MAE)

The 25-node campaign used `nn.MSELoss`. The uniform-MAE switch (matching MTGODE §4.4, AGCRN's `nn.L1Loss` default, CLCRN's `masked_mae_loss` default) landed 2026-08-08. Provenance is in every `summary.json` (`"loss_type": "MSELoss"`).

If the 25-node results appear alongside N = 32 results in the paper, they must be labelled as a separate condition with the training loss stated explicitly. Cross-condition comparisons should note the loss difference (MAE-trained models are advantaged on MAE tables, MSE-trained on RMSE tables — the "double penalty" story).

### 8.3 Scheduler is a harness choice, not from any paper

`CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)` uniform across all three models. Bai, Lin, and Jin all use constant-lr Adam. Kept uniformly for internal comparability. Disclose in methods.

### 8.4 Train / val / test window is compact

60 / 20 / 20 chronological split of 70,128 hours gives a **~1.6-year test window (May 2019 → Dec 2020)**. Report the exact test dates so any seasonal bias is visible. Note that COVID-lockdown SAWS operations in 2020 sit inside the test window — flag this for the reviewer.

### 8.5 Univariate regime (all three models)

Each experiment predicts one of {t2m, sp, rh, ws}. Model outputs all 6 SAWS channels; loss is computed on the target channel only via `pred[..., tv_idx:tv_idx+1]` in the trainer. `wd` (wind direction) and `tp` (precipitation) are input features but never predicted (per Gaibie's protocol).

### 8.6 Rollout evaluation is dormant for these results

Block-autoregressive rollout requires `input_len == horizon`. The 25-node runs used 12 / 48 asymmetric shape, so rollout was never invoked; only direct 48-step forecasts were evaluated. Paper wording: "direct multi-step forecasts; block-autoregressive rollout not applicable at this input/horizon shape."

### 8.7 The 25-station selection is a *proper subset* of what SAWS provides

The Lekuba delivery (2013–2020) is a smaller network than the later 60-node campaign delivery. Overlap between the 25-node set and the eventual 32-node set (from the 60-node delivery, restricted to 2016–2025) is partial. Any cross-campaign comparison must note both the network change (25 → 32 stations) and the temporal window change (2013–2020 → 2016–2025).

### 8.8 Baseline configuration

Persistence and climatology are the only two baselines. No deep-learning baseline outside the three-model set. Skill-score analysis (MSSS) is documented in `specs/HYPERPARAMETERS.md` as available via `scripts/compute_msss.py` but was **not** part of the 25-node deliverable at shelf time.

---

## 9. Configuration files (as they existed at 25-node campaign time)

The files below are the *reference* for what the campaign ran. Note: `experiments/configs/{agcrn,clcrn,mtgode}.yaml` have been updated since (audit fixes + HPO winner values); see the "As of" comment blocks inside each file for the 2026-08-04 lock timestamp, and see [`specs/HYPERPARAMETERS.md`](HYPERPARAMETERS.md) for the authoritative history.

| Config | Purpose |
|---|---|
| [`experiments/configs/agcrn.yaml`](../experiments/configs/agcrn.yaml) | AGCRN hyperparameters (Gaibie 2024 Table 1) |
| [`experiments/configs/clcrn.yaml`](../experiments/configs/clcrn.yaml) | CLCRN hyperparameters (Gaibie 2024 Table 1 + Lin 2022 defaults) |
| [`experiments/configs/mtgode.yaml`](../experiments/configs/mtgode.yaml) | MTGODE hyperparameters (Jin 2022 §5.2, pre-audit values) |
| [`experiments/configs/saws_lekuba.yaml`](../experiments/configs/saws_lekuba.yaml) | Reference config for SAWS-specific data + preprocessing keys |

**Preprocessing knobs (SAWS branch, `preprocessing/pipeline.py`):**

| Key | Value | Source |
|---|---|---|
| `exclusion_threshold` | 0.10 | Davidson 2022 §3.2 |
| `distance_threshold_km` | 500.0 | SAWS-tuned (ERA5 default 300 leaves ALEXANDERBAAI isolated at N=25) |
| `sigma_km` | 200.0 | SAWS-tuned Gaussian kernel bandwidth |
| `train_ratio` / `val_ratio` | 0.6 / 0.2 | Davidson-standard chronological 60/20/20 |
| `clip_sigma` | 10.0 | Prevents `tp` numerical instability (57σ outliers pre-clip) |

---

## 10. File-paths reference (then vs now)

Files whose content has changed since the 25-node campaign — the *now* version is what's on HEAD, but the *then* version is what ran:

| Logical role | Then (25-node) | Now (post-audit) | Delta |
|---|---|---|---|
| SAWS parser | `preprocessing/saws_parser.py` (Lekuba long-form parser) | [`preprocessing/saws_parser_25node_lekuba.py`](../preprocessing/saws_parser_25node_lekuba.py) (renamed) | The current `preprocessing/saws_parser.py` is the new N=32 block-format parser (2026-08-08); the file at that path is *not* what ran at 25-node. The Lekuba parser is preserved verbatim under the new name |
| MTGODE wrapper | `models/continuous_time/mtgode_model.py` (pre-audit) | [`models/continuous_time/mtgode_model.py`](../models/continuous_time/mtgode_model.py) (post-audit) | Same path, content rewritten 2026-08-08 to fix Findings 0/2/5. Docstring in the current file names the three findings; git history has the pre-fix state |
| Loss builder | `training/losses.py::mse_loss` was called | [`training/losses.py`](../training/losses.py) | `mae_loss()` added, `mse_loss()` demoted to "reference metric only" |
| MTGODE config | `experiments/configs/mtgode.yaml` (2026-08-04 lock) | [`experiments/configs/mtgode.yaml`](../experiments/configs/mtgode.yaml) (2026-08-10 HPO winner) | `adjoint: true → false`, `cgp_alpha: (missing) → 2.0`, `node_dim: (missing) → 40`, `k: 10 → 20`, `learning_rate: 0.001 → 0.003`, `batch_size: 64 → 128`. See file's own update-block comments |
| Cache | `preprocessing/cache/saws/` (25-node arrays) | Same path (rebuilt at N=32) | The current cache is N=32, 2016–2025. The 25-node cache is not preserved on-disk locally; it can be regenerated by running the Lekuba parser + pipeline against the 25-node config. The npy artefacts inside `results_saws_25node_phase1/runs/*` (state.pt, checkpoint.pt, per_station.npz) are preserved and self-describing |

**Unchanged (safe to reference as-is):**

- [`preprocessing/impute.py`](../preprocessing/impute.py), [`preprocessing/split.py`](../preprocessing/split.py), [`preprocessing/labels.py`](../preprocessing/labels.py), [`preprocessing/normalise.py`](../preprocessing/normalise.py), [`preprocessing/graph.py`](../preprocessing/graph.py), [`preprocessing/dataset.py`](../preprocessing/dataset.py)
- [`training/trainer.py`](../training/trainer.py), [`training/scheduler.py`](../training/scheduler.py), [`training/seed.py`](../training/seed.py)
- [`evaluation/regression.py`](../evaluation/regression.py), [`evaluation/vpt_extended.py`](../evaluation/vpt_extended.py)
- [`models/baselines/agcrn_adapter.py`](../models/baselines/agcrn_adapter.py), [`models/baselines/agcrn_model.py`](../models/baselines/agcrn_model.py), [`models/baselines/AGCRN/`](../models/baselines/AGCRN/)
- [`models/baselines/clcrn_adapter.py`](../models/baselines/clcrn_adapter.py), [`models/baselines/clcrn_model.py`](../models/baselines/clcrn_model.py), [`models/baselines/CLCRN/`](../models/baselines/CLCRN/)
- [`orchestration/manifest.py`](../orchestration/manifest.py), [`orchestration/claim.py`](../orchestration/claim.py)
- [`slurm/worker.sh`](../slurm/worker.sh)
- [`scripts/run_worker.py`](../scripts/run_worker.py) *(the `--hpo-configs-dir` flag was added 2026-08-08 for HPO; the default path — no flag — is what ran at 25-node)*
- [`scripts/compute_baselines.py`](../scripts/compute_baselines.py), [`scripts/analyze_saws25.py`](../scripts/analyze_saws25.py), [`scripts/manifest_incomplete.py`](../scripts/manifest_incomplete.py), [`scripts/build_manifest.py`](../scripts/build_manifest.py)

---

## 11. Provenance & references

- **Data:** SAWS Lekuba delivery (2013–2020), two workbooks (`data1.xlsx`, `data2.xls`); station-selection audit at `data/saws/audit/parse_report.csv`
- **AGCRN:** Bai, L. et al. (2020). *Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting*. NeurIPS 2020.
- **CLCRN:** Lin, H. et al. (2022). *Conditional Local Convolution for Spatio-temporal Meteorological Forecasting*. AAAI 2022.
- **MTGODE:** Jin, M. et al. (2022). *Multivariate Time Series Forecasting with Multi-scale Temporal Graph ODE*.
- **Gaibie protocol:** Gaibie, F. (2024). Honours thesis, UCT. Provided the HPO-optimal AGCRN + CLCRN hyperparameter values used verbatim in the 25-node campaign; provided the univariate-per-target regime and target set {t2m, sp, rh, ws}
- **Davidson & Moodley (2022):** SAWS-station missingness threshold (≤ 10 %) and 60/20/20 chronological split convention. Note: they used MSE loss (the only reference in our chain that does).
- **Pathak et al. (2018):** VPT convention (`ε = 0.3`)
- **Companion specs in this repo:**
  - [`specs/HPC_Parallel_Worker_System.md`](HPC_Parallel_Worker_System.md) — operator's guide to the parallel worker system
  - [`specs/HYPERPARAMETERS.md`](HYPERPARAMETERS.md) — the authoritative hyperparameter provenance document with full history
  - [`specs/Run_Plan_Spec.md`](Run_Plan_Spec.md) — the original design spec for the 120-task run plan
  - [`specs/HORIZON_EXTENSION.md`](HORIZON_EXTENSION.md) — rationale for horizon=48 extension beyond Gaibie's 24 h
