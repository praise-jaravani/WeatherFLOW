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

## P4 — `initial_velocity.py` (new file — replaces upstream `utils.py::fit_velocity`)

**Class: A (indexing/caching change; estimator preserved).**

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

**Ours** (`initial_velocity.py::fit_initial_velocity`):
- Same fresh `Optim_velocity` per fit, same Adam(lr=2), same 200 steps,
  same loss = MSE(delta_u, advection) + λ · RBF-prior with the same
  Gaussian RBF kernel geometry.
- `delta_u` implemented as 2nd-order backward finite difference through
  the last 3 frames (upstream's `NaturalCubicSpline.derivative(t)` at
  the endpoint reduces to this identical expression for a 3-point
  spline).
- Fitted per-sample from the input window's 12 past frames — no
  pre-caching per (year, day-of-year), because our sliding-window
  test inputs do not align with any pre-cached calendar grid.
- Best-loss velocities are returned as tensors handed to the model at
  forward-time via `update_param`.

**Estimator identity**: math is preserved verbatim. Only the
indexing/caching layer differs.

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
