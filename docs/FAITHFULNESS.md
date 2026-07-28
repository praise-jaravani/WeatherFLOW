# ClimODE faithful reimplementation — component checklist

Mapping Verma et al. (ICLR 2024) ClimODE → upstream code
(`models/continuous_time/ClimODE/`) → our integration point.

⚠️ **Status: DRAFT — integration paused at B.1 pending a STOP-AND-ASK
decision.** See `docs/REPORT.md` under "Blocking discovery — training
paradigm mismatch" for the blocker and options. This file records what
was learned during the study phase so the decision can be made from
evidence, not intuition.

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

### Initial-velocity inference (Eq. 10)

| Paper component | Upstream location | Our integration point | Status |
|---|---|---|---|
| Penalised least-squares solve for v(x, t₀) using past frames | `fit_velocity(...)` in `utils.py:411` — precomputes per-year `.npy` cache | see Deviations — this is the biggest incompatibility | ❌ **BLOCKER** |
| Optim_velocity module | `Optim_velocity(nn.Module)` with `v_x, v_y` as `nn.Parameter(shape=(num_years, 1, 5, H, W))` — **year-indexed** | our sliding-window paradigm has no natural "year" axis at training time | ❌ **BLOCKER** |
| Gaussian RBF prior for velocity smoothness (α term in Eq. 10) | `kernel = get_gauss_kernel_region((H, W), lat, lon, region)` — RBF over the H×W grid | trivial to compute for our (27, 37) SA grid | ✅ |
| Δu numerical approximation from past 3 frames | `get_delta_u(past_sample, past_time)` in `utils.py` | our sliding-window input provides T=12 past frames — more than enough | ✅ (mechanically fine, but tied to the year-indexed structure) |

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

### ❌ Blocker: year-indexed training paradigm ↔ sliding-window paradigm

Upstream's `Train_loader` yields **per-day-of-year batches whose first
axis is `num_years`**. `Optim_velocity` has `num_years` in its parameter
shape. The whole training loop iterates day-of-year while implicitly
learning across years simultaneously.

Our WeatherFLOW harness yields sliding-window samples that can start at
any time — there is no "year" axis at training time. Attempting to feed
upstream's model with sliding-window batches would require:

1. Replacing `Optim_velocity`'s per-year parameters with a per-sample
   initial-velocity solve (a fresh least-squares problem for each batch
   element using its 12h input window).
2. Rewriting the training loop so each batch is a set of independent
   sliding-window samples rather than a `(num_years, ...)` block.

Per the mission brief this is a **STOP AND ASK** situation — it
"substitutes the initial-velocity procedure" and I would hesitate to
justify it in PATCHES.md as a "trivial-and-allowed" patch.

### ❌ Blocker: 6h ↔ 1h temporal resolution

Upstream integrates with `t = 0.01 * timestep`, where `timestep` counts
6-hour increments (1460/year). Our data is hourly (8760/year), and our
harness contract is T=12h input, H=12h horizon in 1h steps.

Two paths, both requiring a decision:

- Downsample our ERA5 to 6h to match upstream. **Breaks Phase-1 comparability with the other 3 models** — they'd all need re-benchmarking on 6h data, invalidating 40 completed seeds.
- Change upstream's `t = 0.01 * timestep` scaling and re-tune ODE tolerances for 1h steps. Empirically likely fine, but it's a substantive change to the numerics of the paper's ODE solver.

### ⚠️ Boundary padding assumes circular longitude

Upstream's `boundarypad` uses **reflect** on latitude and **circular**
on longitude. Circular longitude is correct for global forecasting but
wrong for a bounded regional box like SA — SA does not wrap around at
15°E ↔ 33°E. Fix candidates: reflect on both axes, or zero-pad, or
train-longer-with-reflect. Small architectural deviation, needs
documentation in PATCHES.md but not a STOP condition.

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
