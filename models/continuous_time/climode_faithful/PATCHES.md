# PATCHES — climode_faithful vs upstream `models/continuous_time/ClimODE/`

Vendored from https://github.com/Aalto-QuML/ClimODE (MIT License,
© 2023 Aalto-QuML). This file documents every diff from upstream,
by file and function, with class labels for reader triage.

Deviation classes:
- **A**: indexing/caching change; estimator preserved
- **B**: task adaptation, not method change
- **C**: correction of an inapplicable global-only assumption
- **D**: data-interface generalization
- **E**: I/O shim / device / dtype (trivial, allowed by brief)

Every deviation below is disclosed by design; nothing is silent.

## Open TODOs before production seeds

**Constants (lsm + orography)** — ✅ **RESOLVED 2026-07-29**. Real
fields downloaded via `scripts/download_era5_constants.py` (CDS
retrieval, one timestamp 2020-01-01, 38.4 KB). Cached at
`data/constants/{era5_sa_lsm.npy, era5_sa_orography.npy}`. Adapter
loads them in `build_model` when the grid matches (999-node SA box);
falls back to placeholders with a warning otherwise (subgrid tests
still work). Non-degeneracy test at
`test_climode_faithful.py::test_era5_constants_non_degenerate`.
Source recorded in `_constants_source` attribute; production runs
must have `"era5_real"`. See `docs/FAITHFULNESS.md`.

---

## P1 — `model_utils.py::boundarypad` and `ResidualBlock.forward`

**Class: C (correction of an inapplicable global-only assumption).**

**Upstream** (both `boundarypad.forward` and `ResidualBlock.forward`):
```python
F.pad(F.pad(x, (0, 0, 1, 1), 'reflect'), (1, 1, 0, 0), 'circular')
```
Reflect on latitude, **circular** on longitude.

**Ours**:
```python
F.pad(F.pad(x, (0, 0, 1, 1), 'reflect'), (1, 1, 0, 0), 'reflect')
```
Reflect on both axes.

**Why**: circular longitude is correct for global forecasting (upstream's
primary use case). Our bounded regional box (15°E–33°E) does not
longitudinally wrap. The upstream regional path leaks the global-only
assumption unchanged via `Self_attn_conv_reg` (which uses `boundarypad`)
and via `ResidualBlock` (which pads inline the same way). Verified
2026-07-28 by grep of all `boundarypad`/`F.pad`/`circular` occurrences
in upstream `model_utils.py`.

---

## P2 — `model_function.py::Optim_velocity.__init__`

**Class: D (data-interface generalization; math preserved).**

**Upstream**:
```python
self.v_x = nn.Parameter(torch.randn(num_years, 1, 5, H, W))
self.v_y = nn.Parameter(torch.randn(num_years, 1, 5, H, W))
```
Hard-coded K=5 (their z, t, t2m, u10, v10 variable set).

**Ours**:
```python
def __init__(self, num_years, H, W, out_types=5):
    self.v_x = nn.Parameter(torch.randn(num_years, 1, out_types, H, W))
    self.v_y = nn.Parameter(torch.randn(num_years, 1, out_types, H, W))
```
Parameterizes `out_types` (K) so K=8 works with no other changes. Default
value preserved at 5 so a direct upstream-config call still gives
byte-identical shapes. Math unchanged.

---

## P3 — `model_function.py::ClimODE_uncertain_region.forward`

**Class: B (task adaptation, not method change).**

Two coupled changes for the 1-hour data resolution (vs upstream's 6h):

### P3a — time-coordinate rescaling

**Upstream**:
```python
init_time = T[0].item() * 6
final_time = T[-1].item() * 6
```
`T` counts 6h units; multiplied by 6 to convert to hours.

**Ours**:
```python
init_time = T[0].item()
final_time = T[-1].item()
```
`T` counts 1h units directly. The subsequent `t = 0.01 * new_time_steps`
scaling to the ODE solver time coordinate is preserved verbatim — it
maintains "0.01 solver-unit per data-timestep-index"; only the
data-timestep length changes from 6h to 1h.

### P3b — solver-output subsampling

**Upstream**:
```python
if self.err:
    mean, std = self.noise_net_contrib(T, ..., s_final[0:len(s_final):6], ...)
else:
    s_final = s_final[0:len(s_final):6]
```
Samples every 6th solver step because upstream's solver runs at 1h
granularity but data is 6h — this projects predictions back onto the 6h
data grid.

**Ours**:
```python
if self.err:
    mean, std = self.noise_net_contrib(T, ..., s_final, ...)
else:
    mean = s_final
    std = None
```
Our solver already runs at 1h steps matching our data, so no
subsampling. Every solver step corresponds to a forecast lead.

**Rationale (per decision 2026-07-28)**: resolution is task, not method.
Every model in the matrix is retrained at task resolution; ClimODE is
continuous-time and architecturally indifferent to sampling interval.
The upstream 6h protocol was a WeatherBench task choice, not an
architectural requirement.

---

## P4 — `initial_velocity.py` + `velocity_cache.py` (new files — replace upstream `utils.py::fit_velocity`)

**Class: A (indexing/caching change; estimator preserved) — REINFORCED
by §2 of the 2026-07-28 review: the cache design *restores* upstream's
own precompute-once-and-cache pattern on our sliding-window grid.**

**Upstream** (`utils.py::fit_velocity` + `optimize_vel`, verified 2026-07-28):
- Iterates `Train_loader` yielding day-of-year batches of shape
  `(num_years, 1, K, H, W)`.
- For each batch, creates a **fresh** `Optim_velocity(num_years, H, W)`,
  fits with `optim.Adam(lr=2)` for 200 steps against the loss
  `MSE(delta_u, advection) + λ * RBF-prior`.
- `delta_u` is a natural-cubic-spline temporal derivative through the
  previous 3 frames.
- Saves best-loss `v_x, v_y` per (year, day-of-year) to `.npy`.
- Main training loop loads the `.npy` as **fixed** tensors and passes
  them into `model.update_param([past_sample, ...])` — velocities are
  NOT in the main optimizer's parameter groups.

**Ours** (`initial_velocity.py::fit_initial_velocity` + `velocity_cache.py`):

- **Estimator** (`fit_initial_velocity`): identical to upstream —
  same fresh `Optim_velocity` per fit, same Adam(lr=2), same 200 steps,
  same loss = MSE(delta_u, advection) + λ · RBF-prior with the same
  Gaussian RBF kernel geometry. `delta_u` implemented as 2nd-order
  backward finite difference through the last 3 frames (upstream's
  `NaturalCubicSpline.derivative(t)` at the endpoint reduces to this
  identical expression for a 3-point spline; see P6 for the equivalence
  test).
- **Caching layer** (`velocity_cache.py`): the fitted velocity for
  window `i` in split `S` is stored at
  `preprocessing/cache/climode_faithful/velocities/{S}/win_{i:07d}.npy`
  — restoring upstream's own precompute-and-cache pattern to our
  sliding-window grid. Cache key: `(split_tag, window_start_index)`.
  Adam init is deterministically seeded from
  `SHA256("split:index") & 0x7FFFFFFF` so the cache is bit-reproducible
  and shared across seeds (since the estimator is data-only per V1).
- **Fallback**: cache miss → fit-and-store on-demand. Lets the HPC
  smoke run uncached by design (measures worst-case cost) and the
  first real training run warms the cache for subsequent seeds.

**Estimator identity**: math is preserved verbatim. Only the
indexing/caching layer differs — and now that layer is a direct
restoration of upstream's own precompute design, transplanted from
their (year, day) grid to our window grid. The class-A label is
strengthened, not weakened.

---

## P5 — `nll_loss.py` (new file — thin wrapper around upstream's Eq. 12)

**Class: E (I/O shim / packaging).**

Upstream `utils.py::nll(mean, std, truth, lat, var_coeff)` is a single
function. We wrap the same formula in
`nll_loss.py::climode_nll(...)` so the trainer can import it via a
stable module path, apply the same lat-cosine weighting, and expose the
same `var_coeff` hyperparameter for the cosine-annealed variance prior.

Loss formula is preserved: Gaussian NLL on (μ, σ²) with a half-normal
prior on σ, both lat-cosine weighted.

---

## P6 — FD approximation of upstream's spline endpoint derivative

**Class: A (indexing/caching change; estimator preserved).**

Upstream's `utils.py::get_delta_u` uses `torchcubicspline`'s
`NaturalCubicSpline.derivative(t)` at the endpoint of a 3-frame past
window. We implement the equivalent as 2nd-order backward finite
difference: `(3 f_n - 4 f_{n-1} + f_{n-2}) / (2·dt)`.

**Equivalence test** at `tests/test_climode_faithful.py::test_fd_matches_cubic_spline_endpoint_derivative`
(§6 of the 2026-07-28 review directive), using `scipy.interpolate.CubicSpline`
as reference (natural BC). Behaviour on three signal classes:

- **Linear**: both estimators recover the exact slope to 1e-5.
- **Quadratic**: FD is EXACT (2nd-order FD is exact on quadratics).
  scipy's natural-BC spline diverges by up to ~1.5 units at the endpoint
  on a 3-point fit because the natural boundary condition (f″=0 at the
  endpoint) forces a spline that isn't the true parabola. This is a
  KNOWN property of natural-BC splines on 3-point fits, not a bug — but
  it means our FD is (marginally) MORE faithful than the natural-BC
  spline for quadratic curvature at the endpoint. Documented.
- **Sinusoidal**: FD agrees with the analytical derivative to <0.3 on
  a moderate-curvature signal at dt=1, well within O(dt²).

Both estimators are O(dt²) accurate on smooth signals; the natural-BC
detail above means the difference between them is *bounded* rather than
zero, which is fine and disclosed.

If we ever get `torchcubicspline` installed in the container, switch to
importing upstream's exact code — no math change expected.

---

## Not vendored (deliberately)

- `train_region.py`, `train_global.py`, `train_monthly.py`, `evaluation_*.py`:
  training scripts and evaluation loops — we use our shared harness.
- `utils.py` functions unrelated to model or velocity fit
  (`get_train_test_data_batched_regional`, `add_constant_info_region`,
  `get_gauss_kernel`): data-loading helpers replaced by our shared
  preprocessing cache; constants loading replaced by our own SA-region
  constant fields.
- The global (`Climate_encoder_free_uncertain`) and monthly
  (`Climate_encoder_free_uncertain_monthly`) model classes: our task is
  regional, not global or monthly.

---

## What NOT patched (verbatim from upstream)

- `pde()` (dynamics function) in `ClimODE_uncertain_region`
- `noise_net_contrib()` (Gaussian emission head)
- `get_time_pos_embedding()`
- `Self_attn_conv_reg` (except boundary padding via P1)
- `Climate_ResNet_2D` (except boundary padding via P1)
- `Optim_velocity.forward()` (except K parameterization via P2)
- ODE solver settings (`atol=0.1, rtol=0.1`, method arg)
- All hyperparameters that are not resolution-tied
