# climode_faithful integration — REPORT (paused at B.1)

## TL;DR

**Part A (segmentation) is complete and committed on branch
`climode-faithful` (8fc6010).** Live matrix now has 3 models; the
archived "ClimODE-inspired" runs are documented under
`results/archive/climode_inspired/PROVENANCE.md`.

**Part B (faithful integration) is PAUSED at B.1 for a STOP-AND-ASK
decision.** The B.0 grid regularity gate passed cleanly. When I moved
to studying upstream in B.1 I discovered a training-paradigm mismatch
that falls under the mission brief's explicit STOP-AND-ASK list.

Per the brief: *"A reported blocker is a good outcome; a quiet
workaround is the failure mode that caused this entire situation."* So
this report exists.

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
