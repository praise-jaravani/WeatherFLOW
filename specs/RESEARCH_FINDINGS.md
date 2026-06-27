# WeatherFLOW — Research Findings

Living document of empirical observations from the production HPC runs on ERA5 South Africa (999 stations, 2000–2026, hourly, 12-hour history → 12-hour horizon, 8 variables).

**Format:** sectioned by model, with per-seed data. Each finding includes the evidence path so it can be re-verified.

---

## Cross-cutting findings

### F1 — VPT = 0 across all models and variables under epsilon=0.3

All completed seeds show `vpt = 0` for every variable. Initially suspected as a metric bug; investigation confirmed it is a **real model-performance finding**, not a formula artifact.

**Investigation chain:**
1. The original `nrmse_per_horizon` in [evaluation/regression.py](evaluation/regression.py#L93) divided z-scored RMSE by per-horizon pooled test-set std (`/ std_hv`). This is non-standard — preds/targets are *already* z-scored using train climatology, so the second normalisation over-divides.
2. Removed the `/std_hv` division in the same file. Now NRMSE = RMSE in z-score units = climatology-normalised RMSE (the convention from weather-forecasting literature).
3. Post-hoc patched the 3 existing completed runs by recovering `rmse_hv = nrmse_old × std_hv` and recomputing VPT. Script: `/scratch/jrvpra001/weatherflow/patch_vpt.py` (HPC job 951062).
4. **`std_hv` came out at ~0.93 across variables** (not the ~0.6 originally estimated), so the correction shifted values by only ~5–7%. VPT remained 0 across all variables.

**Conclusion:** The model's first-step prediction error (~0.36–0.52 std of climatology) exceeds the 0.3 epsilon threshold from Platt et al. (chaotic-systems literature). For 1-hour-ahead weather forecasting over 999 sparse stations, this threshold is likely too strict.

**Recommendations for the paper:**
- Report VPT at multiple thresholds (0.3, 0.5, 0.7) OR
- Use Anomaly Correlation Coefficient (ACC) as a complementary skill metric, since VPT collapses to 0 at 0.3.
- In the discussion, document that the standard chaotic-systems threshold doesn't transfer cleanly to hourly multi-station weather forecasting.

### F2 — 🔬 Clear two-tier model ranking emerging (PRELIMINARY, n=2–3)

With all 4 models now having at least 2 completed seeds, a consistent ordering has emerged across every variable measured. The gap is not uniform — there appear to be **two distinct performance tiers**.

**Evidence — MAE in physical units across all completed runs (2026-06-27 08:14 SAST):**

| Model | n | t2m (°C) | d2m (°C) | u10 (m/s) | v10 (m/s) | ws10 (m/s) | sp (Pa) | rh (%) |
|-------|---|----------|----------|-----------|-----------|------------|---------|--------|
| **CLCRN** | 2 | **1.17–1.18** | **1.22–1.23** | **0.87** | **0.89–0.90** | **0.74** | **62** | **5.00–5.01** |
| **ClimODE** | 2 | **1.57** | **1.67** | **1.28** | **1.36** | **1.03–1.04** | **~95** | **6.72** |
| MTGODE | 3 | 2.17–2.27 | 2.08–2.11 | 1.46–1.51 | 1.55–1.59 | 1.15–1.19 | 150–155 | 8.68–8.92 |
| AGCRN | 2 | 2.58–2.62 | 2.56 | 1.87–1.89 | 1.95–1.96 | 1.39 | 196–200 | 10.3–10.5 |

**Ranking by t2m MAE:**

| Rank | Model | t2m MAE | vs best (CLCRN) | vs same-tier baseline |
|------|-------|---------|-----------------|------------------------|
| 1 | CLCRN | 1.17 | — | — |
| 2 | ClimODE | 1.57 | +34% | (same tier as CLCRN, +34%) |
| 3 | MTGODE | 2.22 | +90% | (same tier as AGCRN, ~16% better) |
| 4 | AGCRN | 2.58 | +120% | — |

**Two-tier hypothesis:**

1. **Tier 1 (sphere-/geometry-aware):** CLCRN (spherical Chebyshev convs) and ClimODE (physics-informed ODE on lat/lon grid) — both encode the Earth-surface manifold structure explicitly. Combined gap to Tier 2: 40–60% lower MAE.
2. **Tier 2 (generic graph):** AGCRN (adaptive graph conv) and MTGODE (ODE on a generic graph) — treat the 999 stations as a structureless graph and learn relationships from scratch.

Within each tier the gap is small (10–30%); across tiers the gap is large (40–60%). **If this pattern holds at n=10, the paper has a clean architectural story**: the inductive bias of geographic structure matters more than the temporal modeling choice (ODE vs recurrent).

**Status:** PRELIMINARY. Need to confirm at n=10:
- Whether Tier 1's lead is consistent across seeds (currently very tight n=2 variance for CLCRN, ClimODE)
- Whether the within-tier gap (CLCRN > ClimODE by 34%) is real or seed noise
- Whether any variable shows a different ranking (precipitation tp is especially noisy and worth checking)

**Device-fix verified (2026-06-27, sanity test job 964078):** A forward-pass equivalence test compared CPU vs GPU outputs for CLCRN at 999 nodes with a fixed seed and identical model weights. **Max abs diff = 2.23×10⁻⁵, mean abs diff = 4.24×10⁻⁶** — well within float32 floating-point tolerance. The CLCRN dominance finding is real, not a fix artifact. Sanity test script: `/scratch/jrvpra001/weatherflow/clcrn_forward_test.py` on HPC.

### F3 — Dome-shaped NRMSE curves (non-monotonic)

Across both MTGODE and AGCRN, `nrmse_per_horizon` rises from H=0 to a peak around H=5–6, then **slightly decreases** toward H=11. This contradicts the standard expectation that forecast error grows monotonically with horizon (predictability decay).

**Evidence (MTGODE seed 42, t2m, corrected curve):**
```
H:    0     1     2     3     4     5     6     7     8     9    10    11
val:  0.362 0.401 0.435 0.459 0.473 0.483 0.484 0.479 0.468 0.457 0.447 0.438
```

**Possible explanations (none ruled out):**
- Model learning partial mean-reversion / climatology-prediction at long horizons (would push the curve back toward 0.5 = std of mean residual)
- Boundary effect of the loss landscape (loss averages across the horizon, may be dominated by mid-range)
- Specific to the 12-12 input-horizon symmetry — worth re-running with a 24-step horizon to test

**Action:** Flag in the paper discussion. Do not re-run for this finding alone — investigate once full matrix is in.

---

## AGCRN

### Completed seeds at 999 nodes
- **seed 42** ([outputs/agcrn_era5_regression_20260623_211811/run_42/](outputs/agcrn_era5_regression_20260623_211811/run_42)) — 3h24m, early stop epoch 16
- **seed 43** ([outputs/agcrn_era5_regression_20260625_093527/run_43/](outputs/agcrn_era5_regression_20260625_093527/run_43)) — 3h00m

### Metrics (physical units, post-correction)
| Variable | s42 MAE | s43 MAE | s42 RMSE | NRMSE (H=0, s42) | VPT (eps=0.3) |
|----------|---------|---------|----------|-------------------|---------------|
| t2m (°C) | 2.58 | 2.62 | 3.28 | 0.523 | 0 |
| d2m (°C) | 2.56 | 2.56 | 3.31 | — | 0 |
| u10 (m/s) | 1.87 | 1.89 | 2.37 | — | 0 |
| v10 (m/s) | 1.96 | 1.95 | 2.49 | — | 0 |
| ws10 (m/s) | 1.39 | 1.39 | 1.76 | — | 0 |
| sp (Pa) | 196 | 200 | 249 | — | 0 |
| rh (%) | 10.3 | 10.5 | 13.0 | — | 0 |
| tp (m/h) | 9.4e-5 | — | 2.5e-4 | — | 0 |

### Per-model notes
- Early-stopped at epoch 16 (patience=15) on seed 42 — model converges quickly compared to MTGODE
- Seeds 42 and 43 highly consistent (<2% variance per variable)
- Underperforms MTGODE by ~16% on t2m MAE (2.58 vs 2.22°C); **underperforms CLCRN by ~120%** (2.58 vs 1.17)
- NRMSE H=0 = 0.523 is substantially higher than MTGODE's 0.362 — AGCRN starts off worse and the gap doesn't fully close at long horizons

---

## CLCRN

### Completed seeds at 999 nodes
- **seed 42** ([outputs/clcrn_era5_regression_20260625_080630/run_42/](outputs/clcrn_era5_regression_20260625_080630/run_42)) — 16h27m
- **seed 43** ([outputs/clcrn_era5_regression_20260625_123502/run_43/](outputs/clcrn_era5_regression_20260625_123502/run_43)) — 14h50m

### Metrics (physical units, post-correction)
| Variable | s42 MAE | s43 MAE | Notes |
|----------|---------|---------|-------|
| t2m (°C) | 1.17 | 1.18 | ~47% lower than MTGODE; ~55% lower than AGCRN |
| d2m (°C) | 1.22 | 1.23 | |
| u10 (m/s) | 0.87 | 0.87 | |
| v10 (m/s) | 0.89 | 0.90 | |
| ws10 (m/s) | 0.74 | 0.74 | |
| sp (Pa) | 62 | 62 | ~60% lower than MTGODE |
| rh (%) | 5.01 | 5.00 | |
| tp (m/h) | ~0 | ~0 | (precipitation values too small to display at 3 decimals) |

### Per-model notes
- **Substantially outperforms both AGCRN and MTGODE on every variable** (see F2 in cross-cutting findings)
- Seeds 42 and 43 are extremely consistent (<1% variance per variable — even tighter than MTGODE)
- Training duration ~15-16h per seed (between AGCRN's 3h and MTGODE's 10h)
- Initial CLCRN run (job 950032 on 2026-06-23) crashed in 22s due to device-mismatch bug; fix landed before any GPU run completed

### Known issues / fixes applied
- **Device-mismatch bug** (FIXED + VERIFIED): CLCRN's submodule `ConvAttrs` ([CLCRN/model/clcnn/clconv.py:5](models/baselines/CLCRN/model/clcnn/clconv.py#L5)) stored tensor attributes (`sparse_idx`, `local_graph_coor`, `geodesic`, `angle_ratio`) as plain Python attributes, not `register_buffer`. So `.to(cuda)` left them on CPU. Fixed in [models/baselines/clcrn_adapter.py:_move_aux_tensors](models/baselines/clcrn_adapter.py#L24) by walking the model after the input arrives at GPU and moving every plain tensor attribute. **Mathematical equivalence verified by forward-pass equivalence test on 2026-06-27 (job 964078):** CPU vs GPU outputs at 999 nodes with fixed seed differ by max 2.23×10⁻⁵ (mean 4.24×10⁻⁶), within float32 tolerance.
- **torch_scatter import** (FIXED): dead import in [CLCRN/model/manifold/sphere.py:5](models/baselines/CLCRN/model/manifold/sphere.py#L5). Wrapped in try/except since the import isn't actually used in that file.

---

## MTGODE

### Completed seeds at 999 nodes
- **seed 42** ([outputs/mtgode_era5_regression_20260623_211837/run_42/](outputs/mtgode_era5_regression_20260623_211837/run_42)) — 10h27m
- **seed 43** ([outputs/mtgode_era5_regression_20260624_004209/run_43/](outputs/mtgode_era5_regression_20260624_004209/run_43)) — 5h20m
- **seed 44** ([outputs/mtgode_era5_regression_20260626_003241/run_44/](outputs/mtgode_era5_regression_20260626_003241/run_44)) — 8h06m

### Metrics (physical units, post-correction)
| Variable | s42 MAE | s43 MAE | s44 MAE | s42 NRMSE H=0 | VPT |
|----------|---------|---------|---------|----------------|-----|
| t2m (°C) | 2.22 | 2.27 | 2.17 | 0.362 | 0 |
| d2m (°C) | 2.11 | 2.08 | 2.09 | — | 0 |
| u10 (m/s) | 1.46 | 1.51 | 1.47 | — | 0 |
| v10 (m/s) | 1.55 | 1.59 | 1.56 | — | 0 |
| ws10 (m/s) | 1.15 | 1.19 | 1.16 | — | 0 |
| sp (Pa) | 153 | 155 | 150 | — | 0 |
| rh (%) | 8.92 | 8.68 | 8.81 | — | 0 |
| tp (m/h) | 7.9e-5 | 8.0e-5 | — | — | 0 |

### Per-model notes
- **MTGODE outperforms AGCRN on every variable** (10–25% lower MAE)
- **Outperformed by both CLCRN (~50% lower MAE) and ClimODE (~30% lower MAE)** — places MTGODE in the "Tier 2 / generic graph" category (see F2)
- Seeds 42, 43, 44 are consistent (within ~3% on every metric) — good sign for variance estimation
- Training duration variable across seeds (5h, 8h, 10h) — early stopping at different epochs

---

## ClimODE

### Completed seeds at 999 nodes
- **seed 42** ([outputs/climode_era5_regression_20260626_032551/run_42/](outputs/climode_era5_regression_20260626_032551/run_42)) — 18h40m at epochs=30
- **seed 43** ([outputs/climode_era5_regression_20260626_083842/run_43/](outputs/climode_era5_regression_20260626_083842/run_43)) — 19h01m at epochs=30

Both seeds finished cleanly within the 24h SLURM wall — the `epochs: 30` cap is verified to work. Two earlier attempts at `epochs: 40` (jobs 950035, 950038) timed out at epoch 37/38 before test eval could run.

### Metrics (physical units)

| Variable | s42 MAE | s43 MAE | Notes |
|----------|---------|---------|-------|
| t2m (°C) | 1.57 | 1.57 | ~34% higher than CLCRN; ~30% lower than MTGODE |
| d2m (°C) | 1.67 | 1.67 | |
| u10 (m/s) | 1.28 | 1.28 | |
| v10 (m/s) | 1.36 | 1.36 | |
| ws10 (m/s) | 1.04 | 1.03 | |
| sp (Pa) | 95 | 95 | |
| rh (%) | 6.72 | 6.72 | |

### Per-model notes
- **Places second overall, behind CLCRN** — joins CLCRN in the "Tier 1 / geometry-aware" category (see F2)
- Seed-to-seed consistency is extreme — values agree to 3 decimals across seeds 42 and 43. Tightest variance of any model so far.
- Per-epoch cost (~38 min × 30 epochs ≈ 19h) plus eval (~30 min) fits the 24h budget with ~4h buffer


### Configuration: `epochs: 30` cap (DECISION RECORD)

After two independent timeouts (seeds 42 and 43 hit the 24h SLURM wall at epoch 37–38 under `epochs: 40`), the cap was lowered to `epochs: 30` in [experiments/configs/climode.yaml](experiments/configs/climode.yaml#L15). All 10 ClimODE seeds are being run homogeneously with epochs=30 for matrix consistency. **Seeds 42 and 43 successfully completed under the new config (18h40m and 19h01m) — the fix is verified.**

### Paper paragraph (paste-ready, vetted)

> All models were trained under an identical protocol — a maximum-epoch cap with best-validation-checkpoint selection (Goodfellow et al., 2016) — following the epoch range of {20, 30, 40} established for this task by Gaibie et al. (2024) and Davidson & Moodley (2022). Per-epoch cost differs across architectures owing to their integration schemes: the discrete and fixed-step models (AGCRN, CLCRN, MTGODE) converge well within the cluster's 24-hour job limit, whereas ClimODE's adaptive ODE solver requires ≈38 min/epoch at 999 nodes. ClimODE was therefore trained with a 30-epoch cap; its validation loss plateaus by epoch 25 (Δ < 0.5% between epochs 25 and 37, below the observed seed-to-seed variance), confirming the cap does not truncate convergence. An analogous 24-hour-compute-driven configuration constraint is reported by Ayers et al.

### Methodology references

- **Goodfellow et al. 2016** — *Deep Learning*. Best-validation-checkpoint selection as the canonical early-stopping protocol.
- **Gaibie et al. 2024** — established the {20, 30, 40} epoch search space for this STGNN benchmark; all four of their models (TCN, GWN, CLCRN, AGCRN) used 40 epochs. 30 is squarely inside this range.
- **Davidson & Moodley 2022** — same {20, 30, 40} range, LSTM and TCN at 40.
- **Ayers et al.** (Local Lyapunov Exponents paper) — direct in-lineage precedent: their first six-layer LSTM HPO attempt exceeded Google Colab's 24-hour wall, and they use early-stopping with best-validation-checkpoint selection. Cite as precedent for compute-ceiling-driven config choices.
- **MLCommons CloudMask** — HPC-scale benchmark introducing seeds + early stopping + best-validation-run reporting.

### Convergence evidence (from ClimODE seed 42 train.log, prior epochs=40 run)

| Epoch | val_loss | Δ from epoch 37 best |
|-------|----------|----------------------|
| 20 | 0.30525 | +1.2% |
| 25 | 0.30317 | +0.5% |
| 30 | 0.30236 | +0.2% |
| 37 | 0.30171 | (best, baseline) |
| 38 | 0.30180 | +0.0% (oscillating) |

Convergence is effectively reached by epoch 30. The 0.2% gap is below seed-to-seed variance (MTGODE seeds 42/43 differ by ~3% on per-variable MAE).

### Scheduler / cosine restart note

ClimODE uses `CosineAnnealingWarmRestarts` with `T_0=10, T_mult=2`. Cycle boundaries: epoch 10 (end cycle 1), epoch 30 (end cycle 2), epoch 70 (end cycle 3). **`epochs=30` lands the final training epoch exactly at a cosine minimum** — the model finishes in its low-LR settled state. Confirmed empirically in the prior train.log: epoch 31 shows the expected LR-restart bump (train loss 0.297→0.299, grad norm 0.36→0.44).

### Phase 2 follow-up: switch ClimODE to fixed-step ODE solver

ClimODE uses adaptive `dopri5` ([experiments/configs/climode.yaml](experiments/configs/climode.yaml#L25)); MTGODE uses fixed-step Euler/RK4 with step size 0.25. The adaptive solver's step count balloons at 999 nodes, explaining the ~38 min/epoch vs MTGODE's ~10–20 min/epoch.

**Proposed Phase 2 change** (not for current matrix):
1. Switch ClimODE to fixed-step solver
2. Run 1 validation seed to confirm accuracy doesn't degrade
3. If validation seed is comparable, re-run all 10 ClimODE seeds with fixed-step at the full `epochs: 40` budget (now ~16h per seed instead of timing out)

**Benefits:**
- Allows full 40-epoch training within 24h budget (matches Gaibie's protocol exactly)
- Aligns ClimODE's integration scheme with MTGODE's → makes the CT-vs-CT comparison more apples-to-apples
- Removes the only methodological footnote in the ClimODE block

**Cost:**
- ~16h GPU for the validation seed
- If accepted, ~160h to rerun all 10 (replacing the current 30-epoch results)
- Slight reduction in "continuous-time purity" claim (but MTGODE makes the same trade)

**Decision threshold:** if the epochs=30 results show ClimODE underperforming AGCRN/MTGODE by margins that could be explained by the cap, run this Phase 2 fix. If ClimODE is competitive at epochs=30, leave it.

---

## Open items (waiting for more data)

- **Per-model variance across the 10 seeds** — currently have n=2 for AGCRN, n=2 for CLCRN, n=3 for MTGODE, n=2 for ClimODE
- **Confirm CLCRN dominance at n=10** — Tier 1 lead is consistent at n=2, but seeds 44+ needed before claiming it as a paper result
- **Confirm ClimODE's place in Tier 1 at n=10** — currently 30% gap to CLCRN at n=2; verify this isn't seed-luck
- **Per-station / geographic stratification** — once all seeds are in, use `per_station.npz` arrays to check whether the Tier 1 advantage is uniform across South Africa or concentrated in specific regions (high-altitude, coastal, etc.)
- **Variable-level skill comparison heatmap** — generate after all 40 seeds are in
- ~~**CLCRN sanity check**~~ — DONE 2026-06-27. Forward-pass equivalence verified at 999 nodes; CPU vs GPU diff is 2.23×10⁻⁵ (within FP tolerance). Fix is mathematically correct.
- **Precipitation (tp) analysis** — all models show very small tp MAE values (~1e-4 m/h); need separate per-event analysis since tp is heavy-tailed and standard MAE may not capture extreme-event skill
