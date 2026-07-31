# MOTHBALL — climode_faithful (frozen 2026-07-30)

**Status**: **frozen, not deleted.** The branch `climode-faithful` on GitHub
(commits `8fc6010..0ec3b3b`) preserves the complete integration. Nothing
on this branch will be run again unless the trigger condition below is
met by explicit supervisor decision.

## Why frozen

**Immediate reason**: 2026-07-30 supervisor redirection paused ClimODE
and rejected ERA5 as the primary basis for headline conclusions. See
`docs/REPORT.md` STAND-DOWN entry for the full context.

**Architectural reason** (why "just move it to SAWS" is not an option):
ClimODE isn't a model that happens to use grids — the grid **is** the
method. Every spatial operator in the paper is lattice-native:

- 2D convolutions over `(H, W)` fields (`Climate_ResNet_2D`)
- KQV attention over grid tokens (`Self_attn_conv_reg`)
- Boundary padding on lat/lon axes (`boundarypad`)
- The advection term itself: `du/dt = -v·∇u - u(∇·v)` computed via
  `torch.gradient()` on a regular lattice
- The Eq. 10 velocity fit that estimates a **velocity field** — a
  field has no native meaning on 28–45 scattered stations

Three theoretically-possible routes to force this onto SAWS all fail
on their own terms:

1. **Interpolate stations to a grid first** → manufactures our own
   gridded product from ~28–45 points, then trains the model on it.
   This is Moodley's ERA5 critique *amplified* — the model learns
   our interpolator, and the grid is nearly empty at station density.
2. **Rewrite the spatial operators for graphs** → replace convs with
   graph convs, grid gradients with graph operators, grid attention
   with station attention. Result: no longer Verma et al.'s
   architecture; it's a hand-written advection graph-ODE. We already
   have one of those. It's called "ClimODE-inspired" and lives in
   `results/archive/climode_inspired/PROVENANCE.md`. This route is
   literally the mistake this whole month was spent correcting.
3. **Don't** → the correct answer.

## What survives regardless of ClimODE's fate

Two pieces of general-purpose infrastructure remain useful on the
main branch (already merged concepts, not tied to this model):

- **Trainer hooks** (`training/trainer.py`): `uses_native_loss` flag
  and `val_subsample_n` config field. Default-off, byte-identical
  for existing adapters. Any future model needing its own training
  objective or a subsampled val set uses these.
- **Velocity cache pattern** (`velocity_cache.py`): deterministic
  per-(split, index) SHA-256 seeding for data-only precomputes.
  The concept generalizes to any expensive-but-data-only per-window
  computation (e.g. per-station Eq. 10 fits if MTGODE ever needed
  similar).

## What is preserved on this branch (frozen intact)

Everything under `models/continuous_time/climode_faithful/`:
- `model_utils.py` — vendored + P1 (reflect-both-axes padding)
- `model_function.py` — vendored + P2 (K parameterization) + P3 (1h time coord)
- `initial_velocity.py` — per-window online Eq. 10 fit
- `velocity_cache.py` — deterministic-seed cache
- `nll_loss.py` — Verma Eq. 12
- `PATCHES.md` — disclosed diff ledger
- `__init__.py` — attribution

Adapter, config, tests, sbatches, smoke scripts, ERA5 constants:
- `models/continuous_time/climode_faithful_adapter.py`
- `experiments/configs/climode_faithful.yaml`
- `scripts/smoke_climode_faithful.py`
- `scripts/smoke_climode_faithful_hpc.py`
- `scripts/warm_velocity_cache.py`
- `scripts/download_era5_constants.py`
- `scripts/jobs/smoke_climode_faithful.sbatch`
- `scripts/jobs/warm_velocity_cache.sbatch`
- `scripts/jobs/climode_faithful_anchor.sbatch`
- `tests/test_climode_faithful.py` (10 tests, all pass)
- `tests/test_grid_regularity.py` (7 tests, all pass)
- `tests/test_native_loss_hook.py` (7 tests, all pass — hooks are branch-agnostic)
- `data/constants/era5_sa_{lsm,orography}.npy` (real ERA5 constants, retrieved 2026-07-29)
- `docs/FAITHFULNESS.md` (paper→code checklist, verification records, pre-registered classification score)
- `docs/REPORT.md` (mission progression + stand-down)

At freeze: **integration was code-complete, all tests green, local
smoke green, HPC sbatch prepared but never submitted**.

## The ONLY revival trigger

Revive **only** on **explicit supervisor greenlight** of a
grid-based second ODE — most likely the "continental ERA5
exploration" pathway Moodley referenced ("preliminary runs, not
final runs" over the whole continent at coarse resolution). Any
station-based revival is architecturally impossible; see above.

If that greenlight arrives, the revival checklist:

1. **Change the grid config**: continental extent (approx S=-35 to
   N=+37, W=-20 to E=+55) at whatever resolution the supervisor
   specifies. New `H×W` — grid regularity test still applies.
2. **Re-download constants** for the new bounding box via
   `scripts/download_era5_constants.py` — update `AREA` and `GRID`
   constants in the script; run once (small CDS request).
3. **Rebuild the preprocessing cache** for the continental extent
   (new `data/era5/raw/` monthly files if not already on disk;
   pipeline auto-detects grid change via hash).
4. **Re-key the velocity cache** — the existing `velocity_cache/`
   entries are keyed to the (SA-999-node) window set and are invalid
   for a different grid; delete `preprocessing/cache/climode_faithful/`.
5. **New HPC smoke** — cannot reuse the SA smoke; need actual epoch
   timing at the continental grid size. `scripts/jobs/smoke_climode_faithful.sbatch`
   works as-is once the cache is rebuilt.
6. **Update `_lat_row_order` / `_lon_row_order` assertions in tests**
   if the grid shape changes materially.

Estimated revival cost: **2–4 days** of focused work (assuming the
raw ERA5 continental data is already downloaded).

## If the greenlight is for a SAWS second ODE instead

Do NOT try to revive this branch. Use the architectural-family
suggestion from the ClimODE-inspired PROVENANCE: **AirPhyNet**
(Hettige et al., ICLR 2024). Same physics×CT cell as ClimODE, but
graph-native by design — advection+diffusion on a station-graph
neural ODE. Same design-family citation, different faithful
representative. Would need a fresh adapter (the `climode_faithful`
adapter's shape logic is grid-specific and doesn't transfer);
however, the trainer's `native_loss` hook and the velocity_cache
pattern are both directly reusable.

## Backup + provenance

- **Local**: `/home/praise/Computer_Science_Resources/Weatherflow`
  (branch `climode-faithful`, ahead of main at `0ec3b3b`)
- **Remote**: `git@github.com:praise-jaravani/WeatherFLOW.git`
  (`origin/climode-faithful`, pushed 2026-07-30)
- **Commit chain to restore full context**:
  - `8fc6010` Part A — archive ClimODE-inspired, 3-model live matrix
  - `8d25700` B.0 grid regularity gate PASSED
  - `5603457` B.1 verifications + B.2 vendored code
  - `f0322d1` B.3+B.4+B.5 (adapter, tests, smoke, sbatch)
  - `5851046` Post-review §1–§7
  - `74d59ce` §1.1 val-subsample + §1.2 real ERA5 constants
  - `dbf9563` §4/§5 infrastructure
  - `15d1318` Phase 2 sync + smoke drip-feed prep
  - `0b1b654` Pre-registered classification score
  - `0ec3b3b` STAND-DOWN

Do not delete this branch. Do not merge it into main without an
explicit revival trigger. The option is worth preserving; the cost
of preserving it is zero; the cost of exercising it prematurely was
the mistake this document exists to prevent.
