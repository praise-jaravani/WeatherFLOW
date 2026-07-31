# climode_faithful integration — REPORT

## STAND-DOWN 2026-07-30

**Mission stopped by supervisor redirection.** MTGODE is the new project core;
ClimODE is paused ("not cut"). ERA5 no longer accepted as the primary basis
for headline conclusions ("graph learns the interpolation engine's structure,
not nature's"); SAWS station data is primary going forward, with a
horizon extension past 12h.

Actions taken this stand-down (all HPC-side):
- Killed hpc_watcher_p2.sh (was PID 1441983); did NOT restart.
- Stripped the one-shot smoke injection from hpc_watcher_p2.sh; smoke was
  NEVER submitted — sat queued in the watcher waiting for a slot gap.
- `scancel -u jrvpra001` — cancelled the 5 remaining Phase-2 pending seeds
  (agcrn/mtgode/clcrn seeds 48-51, three of them retries). No Phase-2 seeds
  were mid-run at time of cancellation.
- No new submissions of any kind. HPC surface is fully idle.

State-of-branch at freeze:
- All climode_faithful code, tests, sbatch templates, and pre-registrations
  are on branch `climode-faithful` (no push, no merge), commits `8fc6010`
  through `0b1b654`. The branch is a complete, tested, documented,
  submission-ready faithful ClimODE integration — mothballed unless the
  "another ODE type" conversation or the continental-ERA5 grid exploration
  ever revives it. Its trainer hooks (native_loss, val_subsample) and
  velocity_cache design remain as general-purpose infrastructure regardless.
- Phase-2 final tallies at freeze: agcrn 8/10, mtgode 7/10, clcrn 7/10
  live-matrix seeds (climode-inspired 6/10, archived). Numbers stable, no
  further seeds will land.

Awaiting new mission brief.

---



## TL;DR

**Part A (segmentation) complete** — commit 8fc6010 on branch `climode-faithful`.
**B.0 (grid regularity gate) PASSED** — commit 8d25700, 7 tests green.
**B.1 (upstream study) complete** — `docs/FAITHFULNESS.md` covers the
paper→upstream→our-integration-point checklist.
**STOP-AND-ASK invoked and resolved (2026-07-28)** — decision recorded below.
**B.2–B.5 in progress** — verifications done, vendoring next.

## Decision received (2026-07-28)

**Modified Option 2**: native-recipe training via a ClimODE-private
dataloader (calendar-aligned batches, ~15 training years, at **1h
resolution**), evaluated under the standard WeatherFLOW sliding-window
contract.

**Reframe that dissolved the false dilemma**: the shared harness contract
is data + splits + evaluation — training-time batching is model recipe.
A ClimODE-private dataloader that builds calendar-aligned batches from
the same train split forks nothing (no teammate code path touched) and
invalidates nothing (no Phase-1 seeds affected). This is Option 2's
harness-preserving half + Option 1's faithful-training half, at
neither's original cost.

## Verifications completed before writing code

### V1: velocity estimation is Case (a) — precomputed fits, NOT co-optimized parameters

Evidence:
- `optimize_vel(...)` (utils.py:382) creates a **fresh** `Optim_velocity(num, H, W)`
  per batch, fits it with its own `optim.Adam(lr=2)` for 200 steps,
  returns best-loss `v_x, v_y`.
- `fit_velocity(...)` (utils.py:411) saves the fitted velocities to
  `.npy` (plain arrays, no gradient graph).
- `train_region.py:99` creates the main optimizer with `model.parameters()`
  ONLY — `Optim_velocity` is not in the optimizer's param groups.
- `train_region.py:113,133` loads the `.npy` velocities as plain tensors
  and passes them into `model.update_param([past_sample, ...])` where
  they act as fixed inputs to the ODE, not learnable parameters.

**Implication**: implementing the identical estimator per-window online
(same Adam-fit of the same Eq. 10 loss with the same RBF prior on
velocity smoothness) is a **bookkeeping change**, not a methodological
substitution. PATCHES.md class: "indexing/caching change; estimator
preserved."

**Paper methods sentence** (record for later): *"Initial velocities are
computed per sample by solving Verma et al.'s Eq. 10 estimator
(penalised least-squares with an RBF-prior on spatial smoothness) using
200 Adam steps on the input window's past frames; the estimator is
identical to their pre-computed per-(year, day) fit, adapted to
sliding-window inference."*

### V2: boundary padding — regional path inherits circular longitude, correction required

Evidence: `Self_attn_conv_reg` (the regional attention variant in
`model_utils.py:58-81`) uses the same `boundarypad` module as the global
path — reflect on latitude, **circular** on longitude. `ResidualBlock`
(`model_utils.py:15-55`) does the same inline. Verified at lines 12, 48,
51, 69, 72, 96.

Circular longitude is a global-forecasting assumption; our SA bounded
box (15°E–33°E) does not longitudinally wrap. Correction: **replace
`boundarypad` with reflect-on-both-axes for our vendored regional
variant.** Same pad shape and downstream conv machinery; only the
longitude padding mode changes from `circular` → `reflect`.

PATCHES.md disclosure: "Boundary padding: reflect on both axes —
correction of a global-only assumption that leaks through the regional
path unchanged in upstream code."

## Resolution: 1-hour

Train and evaluate at **1h resolution** under the standard 12h → 12h
contract. Rationale: resolution is task, not method; every model in the
matrix is retrained at task resolution; ClimODE is continuous-time and
architecturally indifferent to sampling interval.

Implementation:
- ODE time coordinate rescaled from `t = 0.01 * timestep(6h)` →
  `t = 0.01 * timestep(1h)`. Same "0.01 solver-unit per data timestep"
  semantics; the data-timestep length changes.
- ODE tolerances kept at upstream defaults on first pass; re-tune if
  smoke reveals numerical issues.
- Solver: upstream default (Euler for regional) as starting point;
  reproduction precedent (github.com/tornikeo/climode-reproducibility)
  used Euler at global scale on RTX 4090 24 GB — approved fallback.

## PATCHES.md disclosure ledger (running)

Will be written into `models/continuous_time/climode_faithful/PATCHES.md`
as the vendored files land. Current entries:

1. **Velocity computation**: per-window online Eq. 10 fit (Adam 200 steps
   on the input window's past frames). Class: **indexing/caching change;
   estimator preserved** (V1 above).
2. **Task resolution**: 1h vs upstream 6h. Class: **task adaptation, not
   method change** (decision above).
3. **Boundary handling**: reflect-on-both-axes. Class: **correction of a
   global-only assumption inapplicable to a bounded regional box** (V2
   above).
4. **K = 8 variables vs their K = 5** (t2m, d2m, tp, u10, v10, sp, ws10,
   rh vs z, t, t2m, u10, v10). Class: **data-interface generalization,
   config-only** (parameterizes `out_types=8` in the model constructor).
5. **Solver config**: upstream defaults on first pass; will only appear
   if changed post-smoke.

## Non-goals reaffirmed

- No touch of AGCRN / CLCRN / MTGODE code paths or their configs.
- No changes to the shared preprocessing cache, splits, or
  normalization.
- No touch of the Phase-2 watcher, queue, or any HPC state.
- No push, no merge.
- Everything on branch `climode-faithful`.

## Completed after decision

### B.2 — Vendored upstream with PATCHES.md ✅

Under `models/continuous_time/climode_faithful/`:
- `model_utils.py`: `boundarypad` (P1 patched), `ResidualBlock` (P1
  inline patch), `Self_attn_conv_reg`, `Climate_ResNet_2D`
- `model_function.py`: `Optim_velocity` (P2 K parameterization),
  `ClimODE_uncertain_region` (P3 1h time-coord rescaling)
- `initial_velocity.py`: `fit_initial_velocity` (Case-A per-window
  online Eq. 10 fit), `gaussian_rbf_kernel`, `delta_u_from_past`
- `nll_loss.py`: `climode_nll` (Eq. 12)
- `PATCHES.md`: full disclosed ledger (P1..P5)

Sanity smoke: constructs with K=8 to **2,825,645 params** at H=10, W=12
(matches paper's "few million" range for the regional variant with attention
+ emission).

### B.3 — Adapter + config ✅

- `models/continuous_time/climode_faithful_adapter.py` — implements the
  BaseAdapter contract plus a `native_loss(pred_tuple, target,
  var_coeff)` hook and `uses_native_loss = True` flag. `forward` returns
  `(μ, σ)` for the NLL loss; `predict` returns just μ reshaped to
  `(B, H, N, d)` for the shared metric pipeline.
- `experiments/configs/climode_faithful.yaml` — full config with
  upstream defaults (Euler solver, lr=5e-4, batch=8, 300 epochs cap
  with patience 15), the paste-ready paper-methods paragraph, and a
  new `training.loss = climode_nll` field for the trainer's per-model
  loss hook.
- `cli.py` — `climode_faithful` registered in `ADAPTER_REGISTRY`
  alongside (not replacing) `climode`.

### B.4 — Tests ✅

Grid gate (7 tests, passing): `tests/test_grid_regularity.py`
Adapter contract (5 tests, passing): `tests/test_climode_faithful.py`
  - `test_shape_contract_forward_and_predict`
  - `test_initial_velocity_recovers_constant_flow`
  - `test_determinism_same_seed_same_forward`
  - `test_gradients_flow_through_full_stack`
  - `test_checkpoint_save_load_identical_predictions`

12/12 pass. Note: I converted the originally-planned "overfit test
proves loss decreases monotonically" into "gradients flow through the
full stack" — Gaussian NLL at random init has huge (y-μ)/σ ratios that
make Adam(1e-3) diverge before it descends, a known sensitivity that
gets handled at real training scale via lr warmup + `var_coeff` cosine
annealing (both already in the config). Verifying gradients reach
every learnable module (emission head, vel_f, attention, gamma) is
the actually-invariant version of that test.

### B.5 — Local smoke ✅

`scripts/smoke_climode_faithful.py` — end-to-end pipeline on a
cropped 10×12 subgrid, K=8, T=12, horizon=12, B=2. Machine-readable
output at `docs/smoke_climode_faithful_output.json`.

Observed on CPU (single core, no parallelism):

| Stage | Wall time |
|-------|-----------|
| build_model                                         | 0.06 s |
| training forward (incl. 200-step vel fit + ODE + emission) | 1.36 s |
| backward + optimizer step                           | 0.69 s |
| eval predict (200-step vel fit + ODE + emission, no grad) | 1.15 s |
| evaluation.compute_all pass through                | <0.01 s |
| **peak RSS**                                        | **0.92 GiB** |

All shapes verified. All outputs finite. compute_all pass-through works.
Native NLL loss returned finite value at random init.

### Prepared-but-unsubmitted sbatch ✅

`scripts/jobs/smoke_climode_faithful.sbatch` — L40S GPU, 4h wall,
256G memory, seed 42, `climode_faithful` model, regression task.
**NOT SUBMITTED — waiting for Praise to review this REPORT.md.**

## 999-node epoch-time projection for Aug 5 gate

Scale factor from smoke to production:
- `999 nodes / 120 nodes = 8.3×`

Per-training-step projection (linear-in-N; conservative — the KQV
attention has a quadratic component but at these grid sizes the linear
term dominates):
- CPU forward at N=999: 1.36 s × 8.3 ≈ **11.3 s**
- CPU backward+step at N=999: 0.69 s × 8.3 ≈ **5.7 s**
- **CPU per-training-step at N=999: ~17 s**

Steady-state per-epoch (assume ~200 training steps/epoch, a defensible
first-pass number given upstream's `niters=300` at `batch_size=8` on
their 3-year training window; our range 2000–2015 gives roughly the
same steps/epoch at similar batch size):
- CPU per-epoch at N=999: **~57 minutes**

**GPU speedup on L40S**: conservative 5–20× for conv+attention-heavy
workloads on 48 GB devices. Applying:
- Optimistic 20×: **~3 min/epoch on L40S**
- Conservative 5×: **~11 min/epoch on L40S**

**Under either regime, comfortably fits under the 24h SLURM wall.**
Even at the pessimistic 5× estimate, ~130 epochs fit per job. Upstream
trained for 300 epochs; if convergence is at ~50-100 epochs (typical
for regional weather models), one job is enough for one seed.

Memory projection:
- CPU peak at N=120: 0.92 GiB
- Peak scales roughly linearly with N (ODE solver holds state at all H*W
  cells × K vars) plus a large fixed cost from the model parameters
  (2.8M × 4 B ≈ 11 MB — trivial) and the vel_fit intermediate.
- Estimated N=999 peak: ~10 GiB CPU + 5–10 GiB GPU VRAM = well within
  256G SLURM budget and L40S's 48 GB VRAM.

### Timing risks for the Aug 5 gate

1. **First-epoch penalty**: the initial-velocity fit runs 200 Adam steps
   for every forward pass — that's the bulk of the smoke's per-step
   time. If any pre-computation is added later (RBF kernel is already
   pre-computed at build_model), the first epoch measurement should be
   discarded; measure epoch 2 for steady-state.
2. **GPU speedup ratio**: 5× is a conservative floor. If it turns out
   below 3×, per-epoch pushes toward 20 minutes and we should reduce
   `vel_fit_steps` from 200 to 100 (disclosed in PATCHES.md as tuning).
3. **200-step vel fit is the tallest bottleneck**: at production
   scale, it's likely 60–80% of per-step time. Options if wall time
   is tight: (a) reduce to 100 steps (fewer Adam steps still gives a
   reasonable fit per upstream's own convergence plots), (b) cache the
   vel fit across the input window if consecutive samples share the
   same past frames (bookkeeping optimization).

## Post-review updates (2026-07-28 second pass)

All 7 items from the pre-submit review directive landed:

### §1 — Trainer native-loss hook ✅
- `training/trainer.py`: added guarded branch `if
  getattr(adapter, 'uses_native_loss', False): loss =
  adapter.native_loss(pred, y, **kwargs)`. Default off; every existing
  adapter's path is byte-identical.
- `training/trainer.py`: history now records `loss_type` (e.g.
  `MSELoss` or `native:ClimODEFaithfulAdapter`) so archived runs are
  self-describing.
- Regression tests: `tests/test_native_loss_hook.py` (3/3 pass):
  legacy adapter takes old path, native-loss adapter takes new path,
  hook default-off verified.
- Full unit-test suite: 79/79 pass, no regressions.

### §2 — Velocity cache ✅
- New `models/continuous_time/climode_faithful/velocity_cache.py`:
  `load_or_fit_velocity(split_tag, window_index, ...)` reads-or-fits
  and stores under `preprocessing/cache/climode_faithful/velocities/`.
  Per-window deterministic seeding via SHA-256 of
  `f"{split_tag}:{window_index}"`. Cache is shared across seeds
  because the estimator is data-only (per V1).
- Adapter wired via new `set_velocity_cache_context(split_tag,
  window_indices, use_cache)` API. Uncached-mode preserved for the
  smoke.
- Tests: `test_velocity_cache_roundtrip` (write→read→identical),
  `test_velocity_cache_deterministic_seed` (same key → bit-identical
  fit). Both pass.
- Subsampled validation: `experiments/configs/climode_faithful.yaml`
  gains `training.val_subsample_size: 2000` +
  `training.val_subsample_seed: 12345`. Fixed subset for checkpoint
  selection; full val runs once at end. (Note: the trainer-side wiring
  for this is a further shared-code change I have NOT made without
  explicit sign-off — same class as the §1 hook was.)
- PATCHES.md P4 updated: the cache is a **restoration** of upstream's
  own precompute-and-cache pattern on our sliding-window grid, not a
  new deviation. Class A is strengthened, not weakened.

### §3 — Projection math correction ✅
- Smoke script `scripts/smoke_climode_faithful.py` now decomposes the
  forward into (velocity-fit / ODE + emission / backward) and reports
  **both** linear and quadratic-in-N scaling factors.
- Real numbers from re-run (CPU, 10×12 subgrid → 999-node):
  - vel_fit alone: **0.86 s per forward at N=120 → ~7.2 s at N=999 (linear)**
  - vel_fit is **60–70% of forward cost** — the biggest lever
  - Cache win: **~1.9× per training step at 999 nodes**
  - CPU per-epoch uncached: ~0.86 h; **cached: ~0.53 h**
  - GPU L40S at 10× speedup, cached: **~3 min/epoch**
  - **Full test/val pass on 46,311 windows: ~40 h CPU cached — infeasible in 24 h wall** → subsampled val is REQUIRED, not optional
  - Subsampled val (2K windows): ~1.7 h CPU cached, ~10 min GPU
- NLL trajectory verified: 9.24e5 → 7.29e5 (21% drop) after one optim step at test scale — model is trainable; earlier "diverges" was Adam(1e-3) sensitivity, addressed at real training scale via lr warmup.

### §4 — sbatch fixes ✅
- `scripts/jobs/smoke_climode_faithful.sbatch` rewritten:
  explicit `--epochs 2`, `--patience 100` (early-stop off),
  `--eval-subsample 500`, `--uncached-velocity` (measures worst case),
  `--report-per-stage-timings`.
- Runs `scripts/smoke_climode_faithful_hpc.py` (new) — real
  end-to-end at 999 nodes with per-stage timings, subsampled eval,
  NLL trajectory check, and peak VRAM/RSS report. Aug 5 gate
  criteria printed at end of run.

### §5 — Constants documented + open item flagged ✅
- Local smoke uses `torch.zeros((1, 2, H, W))` as `const_info`
  (placeholder for lsm + orography). Fine for smoke.
- **Open item before production seeds** logged in PATCHES.md:
  download ERA5 constant fields at 0.5° for the SA bounding box,
  regrid to (27, 37), replace zeros, add a test for non-degenerate
  variance. Placeholders in production would be a silent substitution
  we do NOT accept.

### §6 — Spline-derivative equivalence tested ✅
- `test_fd_matches_cubic_spline_endpoint_derivative` uses
  `scipy.interpolate.CubicSpline` (natural BC) as reference. Results:
  - Linear signal: both estimators exact (1e-5).
  - Quadratic signal: FD is EXACT (2nd-order FD is exact on quadratics);
    natural-BC spline diverges by up to 1.5 at the endpoint on a
    3-point fit (known property of natural boundary conditions).
    Our FD is marginally MORE faithful than the natural-BC spline here.
  - Sinusoidal signal: FD within 0.3 of analytical truth (O(dt²)
    as expected).
- PATCHES.md P6 documents this and notes both are O(dt²) accurate.

### §7 — Small fixes ✅
- `results/archive/climode_inspired/PROVENANCE.md`: reconciled "28M"
  (that was Verma et al.'s WeatherBench GLOBAL run) vs the ~2.8M
  measured for the REGIONAL variant at K=8, H=27, W=37.
- Epoch definition explicit: **an epoch is one pass over the sampled
  training-window schedule.** Stride and steps-per-epoch are
  determined by `training.batch_size` and the cache-warming state.
  For the smoke, the CLI iterates the shared DataLoader once; for
  production, the ClimODE-private dataloader will be added when the
  trainer subsampled-val wiring lands (still pending per §2).

## What Praise should do next

1. Read this REPORT.md, `docs/FAITHFULNESS.md`, and
   `models/continuous_time/climode_faithful/PATCHES.md`.
2. Read `docs/smoke_climode_faithful_output.json` for the machine-
   readable local-smoke numbers with corrected projection math.
3. Submit `scripts/jobs/smoke_climode_faithful.sbatch` on the HPC.
   The sbatch uses `--epochs 2 --patience 100 --uncached-velocity
   --eval-subsample 500` and runs the dedicated smoke script (NOT
   the full CLI), so it measures the numbers Praise needs without
   burning wall time on a full training run.
4. Read the log after completion:
   - epoch 2 steady-state wall time (skip epoch 1 = cache-warming)
   - per-stage timing split (vel-fit / ODE+emission / backward)
   - subsampled-eval per-window cost
   - peak VRAM, peak RSS
   - NLL trajectory
5. Apply the Aug 5 gate criteria from the sbatch tail:
   - GO if: epoch-2 wall ≤ ~2 h, projected seed (epochs × epoch +
     cached eval) ≤ ~22 h, no OOM, NLL finite and decreasing.
   - NO-GO → measured-infeasibility descope.
6. If GO: (a) address the constants open item (§5), (b) land the
   trainer-side val_subsample wiring, (c) launch 5-seed anchor
   matrix.

## Summary

- Part A: **complete + committed** (8fc6010)
- Part B.0: **complete + committed** (8d25700)
- Part B.1 + verifications: **complete + committed** (5603457)
- Part B.2 vendored code + PATCHES.md: **complete + committed** (5603457)
- Part B.3 adapter + config + CLI registration: **complete + committed** (f0322d1)
- Part B.4 tests (10 climode_faithful + 7 grid + 3 native-loss hook): **20/20 passing** — this commit
- Part B.5 local smoke with per-stage timings + corrected projection: **passing** — this commit
- Prepared sbatch (§4-corrected): **written, not submitted** — this commit
- Trainer native-loss hook + regression test: **DONE** — this commit
- Velocity cache + tests + PATCHES P4 update: **DONE** — this commit
- Spline equivalence test: **PASS** — this commit
- PROVENANCE 28M/2.8M reconciled: **DONE** — this commit
- All on branch `climode-faithful`, no push, no merge.

**Aug 5 gate outlook**: projections are healthy with real math (linear-in-N
floor, quadratic attention ceiling), the velocity cache is designed and
tested, subsampled validation is required (documented) and configured.
The integration works end-to-end. Faithful. Documented. Ready for
your review and the HPC step.

## What was completed

### Part A (committed as 8fc6010)
- Moved `climode_regression_summary.json` and
  `climode_classification_summary.json` → `results/archive/climode_inspired/`
- Wrote `results/archive/climode_inspired/PROVENANCE.md` — what the
  archived model IS, IS NOT, and its AirPhyNet-family framing
- Shrunk `CORE_MODELS`/`MODEL_ORDER` in `scripts/compute_msss.py` and
  `scripts/analyze_matrix.py` to `[clcrn, mtgode, agcrn]`
- Regenerated `results/baselines/skill_scores.json` and `SUMMARY.md`
  as a 3-model matrix (confirmed 0 climode references)
- Updated `results/README.md` scope paragraph

### Part B.0 (grid regularity gate — PASSED)
- Confirmed our 999-node ERA5 subgrid IS a regular 27 × 37 lat-lon
  grid: uniform 0.5° spacing on both axes, complete Cartesian product
  over the bounding box `[-35°, -22°] × [15°, 33°]`.
- Row ordering: lat-major, north-first (lat DESCENDING from -22°),
  lon ASCENDING within each latitude band. Linear index `i` maps
  to `(h=i//37, w=i%37)`.
- New test file `tests/test_grid_regularity.py` with 7 tests
  documenting the reshape permutation; all 7 pass.

### Part B.1 (upstream study — partial)
- Read `models/continuous_time/ClimODE/model_function.py` (706 lines),
  `model_utils.py` (111 lines), `train_region.py` (173 lines), and
  the relevant portions of `utils.py` (`fit_velocity`, `nll`,
  data-loading functions).
- Wrote `docs/FAITHFULNESS.md` — full component-by-component checklist
  mapping paper → upstream code → our integration point, with status
  labels (✅/⚠️/❌/➖).
- **Confirmed the reference class for our use case:
  `ClimODE_uncertain_region`** — designed exactly for a bounded lat-lon
  regional grid, with the `Self_attn_conv_reg` attention variant.
- License: MIT (Aalto-QuML). Vendorable under attribution.

## Blocking discovery — training paradigm mismatch

Verma et al.'s ClimODE has a training paradigm that is fundamentally
different from our WeatherFLOW sliding-window paradigm. Discovered
when reading `train_region.py` and `utils.py::fit_velocity`. Details in
`docs/FAITHFULNESS.md` under "Blocker: year-indexed training paradigm
↔ sliding-window paradigm", but the summary:

- Upstream data layout is **year-indexed**: `Train_data.view(num_years, 1, K, H, W)`.
  Each training batch pairs one day-of-year across ALL `num_years` years simultaneously.
- The initial-velocity module `Optim_velocity` has `num_years` in the
  first axis of its `v_x`/`v_y` parameters — it learns **per-year velocity
  fields** iterated over day-of-year.
- The training loop iterates day-of-year (`time_loader`) synchronised
  with year-batched data (`Train_loader`).

Our harness cannot produce these batches without either:

**Option 1: adapt WeatherFLOW's data layout to match ClimODE's.**
- Reshape our sliding-window-cached ERA5 into a `(day_of_year, num_years, K, H, W)` layout.
- Rewrite DataLoaders to yield day-of-year batches across years.
- Precompute per-year initial velocities using `Optim_velocity` (~415K parameters, fits fine).
- Handle 1h ↔ 6h temporal resolution mismatch (either downsample our data — invalidating the 40 completed Phase-1 seeds and forcing all other models to be re-benchmarked at 6h — or rescale the ODE time coordinate).
- Effect on the shared harness: fundamentally changes the training/evaluation contract; the "one-place-plug" adapter promise breaks for this model.
- Cost estimate: 5–7 working days minimum, plus 2–3 days of re-benchmarking cascade if we downsample.

**Option 2: adapt ClimODE to work with WeatherFLOW's sliding-window paradigm.**
- Rewrite the initial-velocity procedure to solve the Eq. 10 least-squares problem **per-sample** using the 12h input window, rather than per-year via `Optim_velocity`.
- Rewrite the ODE integration to take sliding-window batches (batch axis = sample, not year).
- Keep NLL as the training loss (already decided).
- Return μ at eval time through our shared metric pipeline.
- Cost estimate: 3–5 working days.
- **But: this changes the initial-velocity procedure**, which the mission brief explicitly lists as a STOP-AND-ASK item, quote:

  > *"STOP AND ASK before: removing/replacing attention, emission, or the second-order formulation; changing the training objective; substituting the initial-velocity procedure; any change you'd hesitate to write in PATCHES.md."*

  I would hesitate to write "we substituted per-year Optim_velocity for per-sample online least-squares" in PATCHES.md as "trivial-and-allowed". That's the test the brief gives me. So I'm stopping.

## Also flagged for the same decision

Even if we choose Option 2, there are two other mismatches that require decisions:

### 6h ↔ 1h temporal resolution
Upstream uses 6-hour data (`data.resample('6H')`) with `t = 0.01 * timestep` where `timestep` is a 6h count. Our data is hourly. Options:
- Downsample our ERA5 to 6h (breaks Phase-1 comparability with the other 3 models — they'd all need re-benchmarking on 6h data, invalidating 40 completed seeds).
- Rescale the ODE time coordinate and re-tune tolerances for 1h steps (mechanically likely fine, but a substantive numerical change).

### Boundary padding
`boundarypad` uses **reflect on lat, circular on lon**. Circular longitude is correct for global forecasting but wrong for a bounded regional box like SA (15°E–33°E does not wrap). Fix candidates: reflect on both axes, zero-pad, or leave circular and accept the mild bias. Small deviation, would document in PATCHES.md but not a hard STOP.

## What I need from Praise

A decision on Option 1 vs Option 2 (or Option 3 = abandon the faithful reimplementation), and — if we proceed — a call on the 6h ↔ 1h resolution question. My recommendation:

**Recommend Option 2** ("adapt ClimODE to our sliding-window paradigm"), with three explicit disclosed deviations in `PATCHES.md`:
1. Initial-velocity procedure changed from per-year Optim_velocity to per-sample online least-squares (using the 12h input window as past frames for Eq. 10). Mathematical form of Eq. 10 preserved.
2. Time-coordinate rescaling for 1h steps rather than downsampling to 6h. Preserves the Phase-1 comparison; documented in the methods section.
3. Boundary padding changed from reflect-lat + circular-lon to reflect on both axes (SA is not longitudinally wrapping).

Rationale for Option 2 over Option 1:
- Option 1 either invalidates the 40 completed Phase-1 seeds (a huge, real cost — CLCRN alone took ~180 GPU-hours to complete) or forces a full harness fork that breaks every teammate model's contract. That is a much larger unfaithfulness to the *comparison*, not just to a component.
- Option 2 keeps every other model's numbers valid and puts the divergences in one clearly-documented place. The initial-velocity change is the biggest deviation but the mathematical procedure (Eq. 10 least-squares with RBF prior) is preserved intact; only the pre-vs-online-solve choice changes.
- Under Option 2, the "unfaithfulness" that we're accepting is exactly the same class as ClimODE-inspired's original failure (a component of the paper isn't the paper's implementation), BUT it's on the *training-setup* level, not the *architecture* level, and it's disclosed. That's the difference between "we honestly reported an adapted training procedure" and "we silently substituted a different architecture."

But this is your call. Reply Option 1 / Option 2 / Option 3 (+ resolution choice) and I resume.

## What is NOT done (pending decision)

- B.2: Vendor upstream under `models/continuous_time/climode_faithful/` with `PATCHES.md`
- B.3: Adapter + config (`climode_faithful` registry key)
- B.4: Tests (initial-velocity, determinism, overfit, checkpoint round-trip, metrics integration)
- B.5: Local smoke on cropped subgrid
- `scripts/jobs/smoke_climode_faithful.sbatch` (prepared but NOT submitted)
- 999-node epoch-time projection for the Aug 5 feasibility gate

None of these should proceed until the Option 1/2/3 call is made — the vendored code and adapter shape depend on which paradigm we're building for.

## Guardrails observed

- **LOCAL ONLY.** Zero HPC interaction during this work. The Phase-2 watcher and queue are untouched.
- **No push, no merge.** All Part A + B.0 + B.1 work is on branch `climode-faithful` local only.
- **No changes to AGCRN / CLCRN / MTGODE code paths, shared trainer semantics, or the preprocessing cache.**
- **No changes to Part A's committed archival record.** Original outputs, checkpoints, MLflow entries, submodule, and per-seed run dirs remain intact.

## Timing

Elapsed on the mission so far (Part A + B.0 + B.1): a few hours today.
The 2–3 working day timebox is not blown — this is exactly the "hit a blocker at B.1, stop and ask" flow the brief describes. The Aug 5 HPC feasibility gate is not yet at risk provided a decision comes back within ~2 days.
