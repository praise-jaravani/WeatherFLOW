# ClimODE faithful reimplementation — component checklist

Mapping Verma et al. (ICLR 2024) ClimODE → upstream code
(`models/continuous_time/ClimODE/`) → our integration point.

**Status: RESUMED at B.2 after DECISION (2026-07-28).** The mission
brief's STOP-AND-ASK was invoked at end-of-B.1; decision came back:

- **Modified Option 2**: native-recipe training via a ClimODE-private
  dataloader (calendar-aligned batching, ~15 training years, at **1h
  resolution**), evaluated under the standard WeatherFLOW sliding-window
  contract. Shared harness contract = data + splits + evaluation, not
  training-time batching. This forks no teammate code path and
  invalidates no Phase-1 seeds.
- **1h resolution** (train and eval): resolution is task, not method.
  ClimODE is continuous-time and architecturally indifferent to sampling
  interval — hourly queries are architecturally native.
- **Velocity estimation**: verified to be Case (a) — precomputed Eq. 10
  least-squares fits, not co-optimized parameters (see below). We
  implement the identical estimator per-window online.
- **Boundary padding**: verified — regional path inherits circular
  longitude. Correction to reflect-on-both-axes required.

See `docs/REPORT.md` for the decision record and PATCHES.md (once
vendored) for the running deviation ledger.

## Upstream license

MIT License, Copyright (c) 2023 Aalto-QuML. Vendorable under attribution.

## Which upstream class is the reference for us

The upstream repo has four model classes in `model_function.py`:

- `Climate_encoder_free_uncertain` — global (5.625° WeatherBench)
- `ClimODE_encoder_free_uncertain` — global variant
- `ClimODE_uncertain_region` — **regional** (bounded lat-lon box)
- `Climate_encoder_free_uncertain_monthly` — climate (monthly averages)

**Reference class for our SA integration: `ClimODE_uncertain_region`.**
It's designed exactly for our use case (bounded regional H×W with the
regional variant `Self_attn_conv_reg` of the attention module using
reflect+circular boundary padding).

## Component-by-component checklist

Legend: ✅ present upstream & compatible with our data · ⚠️ present but
mismatches our harness · ❌ blocking mismatch · ➖ not yet examined.

### Architecture

| Paper component | Upstream location | Our integration point | Status |
|---|---|---|---|
| Second-order neural continuity equation (Eq. 5) | `ClimODE_uncertain_region.pde` (transport `adv1` + compression `adv2`) | new `climode_faithful_model.py` — call upstream `pde` | ✅ |
| Flow velocity ˙v = f(u, ∇u, v, ψ) (Eq. 4) | `vel_f = Climate_ResNet_2D(...)` in `__init__` | (as above) | ✅ |
| Convolution network `f_conv` (ResNet, 3×3 convs) | `Climate_ResNet_2D` using `ResidualBlock` from `model_utils.py` | (as above) | ✅ |
| Boundary padding for regional case | `boundarypad` (reflect on lat, **circular** on lon) | need to verify: SA is not circular in longitude — see Deviations below | ⚠️ |
| Attention network `f_att` (KQV) | `Self_attn_conv_reg` in `model_utils.py` (regional variant) | (as above) | ✅ |
| Learnable γ combining conv + att (Eq. 6) | `self.gamma = nn.Parameter(...)` | (as above) | ✅ |
| Spatiotemporal embeddings ψ (Eqs. 7–9) | in `pde()`: day/season embeddings + latlon `pos_feats` | (as above) | ✅ |
| Land–sea mask + orography (constants ψ(c)) | loaded from `constants_5.625deg.nc` (WeatherBench) | need to compute/download for SA at 0.5° — see Deviations | ⚠️ |
| Gaussian emission network g (Eq. 11) | `noise_net = Climate_ResNet_2D(...)` and `noise_net_contrib` method | (as above) | ✅ |
| ODE solver | `torchdiffeq.odeint(pde_rhs, ..., method=self.method, atol, rtol)` | (as above) | ✅ |
| Method-of-lines discretisation (Eq. 5) | implicit in `torch.gradient(ds, dim=2/3)` on H×W grid | ✅ works on our (H=27, W=37) grid (B0 gate passed) | ✅ |

### Data pipeline

| Paper component | Upstream location | Our integration point | Status |
|---|---|---|---|
| Regular H×W grid | `Train_data.shape[3], Train_data.shape[4]` = H, W | our cache: 27 × 37 (B0 test passes) | ✅ |
| Regular temporal grid (uniform Δt) | 6h increments (`data.resample('6H')`) | ours: **1h increments** — see Deviations | ⚠️ |
| Constants (lsm, oro) | `constants_5.625deg.nc` | not in our cache; needs sourcing from ERA5 constant fields at 0.5° | ⚠️ |
| Data layout | `(day_of_year_index, num_years, K, H, W)` — YEAR-INDEXED | **our data is sliding-window `(sample_idx, T, N, d)` — not year-indexed** | ❌ **BLOCKER** |
| Batching | batch = same day-of-year across all `num_years` years | our batch = random sliding-window samples | ❌ **BLOCKER** |
| K variables | K=5 (`z`, `t`, `t2m`, `u10`, `v10`) | K=8 (t2m, d2m, tp, u10, v10, sp, ws10, rh) | ⚠️ trivial config change |

### Initial-velocity inference (Eq. 10) — verified Case (a)

**Verified 2026-07-28**: the initial velocities are **Case (a) — precomputed
fits, not co-optimized parameters**. Evidence:

- `optimize_vel` (utils.py:382) creates a **fresh** `Optim_velocity(num, H, W)`
  per batch, fits it with its own `optim.Adam(lr=2)` for 200 steps, returns
  best-loss `v_x, v_y`.
- `fit_velocity` (utils.py:411) saves the fitted velocities to `.npy` (plain
  arrays, no gradient info).
- `train_region.py:99` creates the main optimizer with `model.parameters()`
  ONLY — the `Optim_velocity` module is not in the optimizer's param groups
  during main training.
- `train_region.py:113,133` loads the `.npy` velocities as plain tensors and
  passes them into `model.update_param([past_sample, ...])` where they act as
  fixed inputs to the ODE, not learnable parameters.

**Deviation class per PATCHES.md ledger: "indexing/caching change; estimator
preserved."** We implement the identical estimator (Adam-fit of the Eq. 10
loss with RBF prior) per-window online rather than precomputing per (year,
day-of-year). The mathematical form of Eq. 10, the number of past frames
used, the RBF prior on velocity smoothness, and the natural-cubic-spline
temporal derivative are all preserved.

| Paper component | Upstream location | Our integration point | Status |
|---|---|---|---|
| Penalised least-squares solve for v(x, t₀) using past frames | `optimize_vel` (utils.py:382) — 200-step Adam(lr=2) fit of MSE loss + RBF prior | new `climode_faithful/initial_velocity.py` — same estimator, per-sample-online rather than per-year-cached | ✅ (indexing change, math preserved) |
| Optim_velocity module (v_x, v_y as `nn.Parameter`) | `model_function.py:12` — used inside `optimize_vel` as an intermediate optimization variable, discarded after fit | keep as-is; still an intermediate | ✅ |
| Gaussian RBF prior for velocity smoothness (α term in Eq. 10) | `get_gauss_kernel_region((H, W), lat, lon, region)` — RBF over the H×W grid, precomputed and cached to `kernel_<region>.npy` | precompute once at model init for our (27, 37) SA grid, cache in-memory | ✅ |
| Δu numerical approximation from past frames | `get_delta_u(past_sample, past_time)` in `utils.py:330` — natural cubic spline through 3 past frames, take derivative at endpoint | use same estimator on our 12h input window (12 past frames — more than the 3 they use, we take the last 3 or all 12) | ✅ (config choice, document in PATCHES.md) |

### Training

| Paper component | Upstream location | Our integration point | Status |
|---|---|---|---|
| Objective = Gaussian NLL (Eq. 12) | `nll(mean, std, truth, lat, var_coeff)` in `utils.py:166` | **new `training/losses.py::climode_nll` or per-model loss branch in trainer** — per Praise's confirmed decision | ✅ (design settled) |
| Variance prior with cosine-annealed λ_σ | `var_coeff = 2 * scheduler.get_last_lr()[0]` (first epoch: 0.001) | replicate scheduler behaviour | ✅ |
| Optimizer / L2 reg / grad clip | `AdamW(lr=0.0005)`, L2 λ=0.001 added to loss, no clip | close to our existing recipes; expose in config | ✅ |
| Scheduler | `CosineAnnealingLR(optimizer, T_max=300)` (their default niters=300) | our shared `CosineAnnealingWarmRestarts` is a superset — likely fine | ✅ |

### Evaluation

| Paper component | Upstream location | Our integration point | Status |
|---|---|---|---|
| Point-forecast metric on μ | evaluation_region.py returns `mean` | route μ through our shared `evaluation.regression.compute_all` | ✅ (per Praise's decision) |
| Uncertainty σ | stored, unused in headline table | store σ for later extremes analysis | ✅ |

## Deviations already identified (some blocking, some trivial)

Ordered from "would need STOP-AND-ASK" to "trivial-and-allowed per brief".

### ✅ Resolved: model-private training dataloader (calendar-aligned, 1h)

**Reframed by decision 2026-07-28**: the shared harness contract is
data + splits + evaluation — **not training-time batching**. Each model
already brings its own optimizer/scheduler/objective, so a ClimODE-private
training dataloader that constructs calendar-aligned batches from the
same underlying train split is model-recipe, not a harness fork.

Implementation:
- Training data source: the same `X_train` array in
  `preprocessing/cache/era5/` (no re-split, no re-normalization).
- Batching: a training batch is a set of samples at the same clock-time
  drawn across the training years (~15-way for our 2000–2015 train
  window, if we treat any consecutive calendar year as an epoch tile).
  This preserves the year-axis structure upstream's optimization relies
  on, at zero cost to the shared cache.
- Sample selection: an epoch iterates over a sampled schedule of
  calendar start-times; stride chosen so epoch size is comparable to
  other models' (documented in PATCHES.md).
- Test-time: standard WeatherFLOW sliding-window contract via our
  shared trainer. Model returns μ; μ goes through
  `evaluation.regression.compute_all` unchanged.

### ✅ Resolved: 1h resolution (task adaptation, not method change)

**Decision 2026-07-28**: train and evaluate at **1h resolution** under
the standard 12h → 12h contract. Rationale: resolution is task, not
method — every model in the matrix is retrained at task resolution;
ClimODE is continuous-time and architecturally indifferent to sampling
interval. Their 6h protocol was a task choice (WeatherBench 5.625°/6h),
not an architectural requirement.

Concrete changes:
- ODE time coordinate rescaled from `t = 0.01 * timestep` (where their
  `timestep` counts 6h units) to `t = 0.01 * hourly_timestep` (ours
  counts hourly units). Semantically the same "0.01 solver-unit per
  data timestep" scaling; just the data-timestep length changes.
- ODE tolerances kept at upstream defaults on first pass; re-tune if
  the smoke reveals numerical issues.
- Solver: upstream default (Euler for regional) as the starting point;
  reproduction precedent (github.com/tornikeo/climode-reproducibility)
  used Euler at global scale on RTX 4090 24 GB — approved fallback if
  wall-clock threatens.

Disclosed in PATCHES.md as: "Task resolution: 1h vs their 6h protocol
(preserves the Phase-1 comparison; ClimODE is continuous-time and
sampling-interval-agnostic)."

### ⚠️ Boundary padding — reflect on both axes (regional path correction)

**Verified 2026-07-28**: upstream's **regional** path also inherits
`boundarypad` (reflect-lat + circular-lon) via `Self_attn_conv_reg` and
`ResidualBlock` — see `model_utils.py:12,48,51,69,72,96`. Circular
longitude is correct for global forecasting but **wrong for our bounded
SA box** (15°E–33°E does not longitudinally wrap).

Correction: replace `boundarypad` with reflect-on-both-axes for our
regional variant. Same pad shape and same downstream conv machinery;
only the padding mode on the longitude axis changes from `circular`
→ `reflect`.

Disclosed in PATCHES.md as: "Boundary padding: reflect on both axes
(correction of a global-only assumption inapplicable to a bounded
regional box). Upstream's regional path inherits circular longitude
from the global implementation — verified in `model_utils.py`."

### ⚠️ Constants: land–sea mask + orography at 0.5° SA subgrid

Upstream ships `constants_5.625deg.nc` from WeatherBench. We need
either:
- Download ERA5 constant fields at 0.5° for our SA bounding box, or
- Use zeros / a synthetic mask (breaks faithfulness to a component the
  paper uses).

Trivial-and-allowed if we download the real ones. ⚠️ if we substitute.

### ⚠️ K = 8 variables vs upstream K = 5

Trivial-and-allowed per brief. `out_types=8` in the model constructor;
the model shape autoscales. Disclosed here as a data-interface
generalization, not silently changed.

### ➖ Not yet examined

- Exact per-tensor shapes of `past_samples`, `const_info`, `lat_map`, `lon_map` used by `update_param()` — will resolve once integration path is clear.
- Whether `torchdiffeq.odeint_adjoint` is compatible with the memory profile at (batch, K=8, H=27, W=37) — expected fine given the paper runs on RTX 4090 24 GB at (globe, K=5, H=32, W=64) which is ~2× our grid.
- What `spectral` mode does (`--spectral` arg in `train_region.py`); currently 0 in default runs.

## What needs a decision before B.2 can proceed

The initial-velocity substitution + year-indexed training paradigm
mismatch is real, blocking, and directly listed in the mission brief as
STOP-AND-ASK. See `docs/REPORT.md`.
