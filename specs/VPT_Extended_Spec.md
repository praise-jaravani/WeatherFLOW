# VPT Extended — Implementation Specification

**Project:** WeatherFLOW (CSC4019Z) · Praise Jaravani
**Doc status:** Draft for review — decisions marked ⚠ require Praise sign-off; one item requires Dr. Moodley sign-off
**Executing agent:** Claude Code, under the rails in this document
**Last updated:** 2026-08-03

---

## 0. Purpose and scope

This spec upgrades the project's Valid Prediction Time evaluation in three tiers, motivated by (a) the Moges/Skokos/Moodley ChaosNetBench (CNB) paper's rollout-based VPT definition, (b) Henok Moges' 2026-08-03 email critique of batch-pooled thresholding (sent re: Génevieve, applies identically to our pipeline), and (c) the floor/censoring resolution problems already identified in our own results (Tier-2 VPT=0 everywhere; CLCRN t2m/sp right-censored at ≥12).

**Deliverable A — ε-sensitivity grid.** Recompute first-crossing VPT at multiple thresholds from *already stored* per-seed `nrmse_per_horizon` curves. No model execution. Resolves censoring at both ends; demonstrates ranking is ε-invariant.

**Deliverable B — per-sample VFH distributions.** Replace the single batch-pooled curve with per-forecast-sample first-crossing times, summarised as distributions with explicit censoring fractions and survival curves. This is the substantive fix Henok's email points at. One streaming test-set pass per checkpoint; no retraining.

**Deliverable C (optional, gated) — block-autoregressive rollout VPT.** True CNB-comparable VPT: predictions fed back as input in 12-h blocks, chained to 48 h, first crossing of NRMSE = 1.0. Runs on existing checkpoints. Gated on Moodley sign-off and the post-Aug-5 Phase-3 budget.

**Explicitly out of scope:** retraining any model; changing the 12→12 task; modifying `evaluation/regression.py::valid_prediction_time` (it is reused verbatim); touching teammate adapters.

---

## 1. Definitions and naming (binding for all reporting)

Two distinct metrics. They never share a table, axis, or sentence without their qualifier.

| Term | Abbrev. | Definition | Cap | Primary ε |
|---|---|---|---|---|
| **Valid Forecast Horizon** | VFH | First horizon step h ∈ {1..12} at which per-sample NRMSE crosses ε, within a single direct 12-step forecast from ground-truth input. Our existing metric, renamed per Henok's suggestion ("valid forecast horizon"), now computed per sample (Deliverable B) as well as pooled (legacy). | 12 | 0.3 (headline), full grid reported |
| **Rollout VPT** | RVPT | First step at which NRMSE crosses ε under block-autoregressive rollout (predictions fed back as input), following Pathak et al. (2018), Gilpin (2023), and CNB. | 12·K (K = rollout blocks) | 1.0 (Pathak/CNB convention), grid secondary |

**Naming rationale (for the paper's methods section):** our models are direct multi-step forecasters evaluated within their trained horizon; CNB's VPT is defined under autoregressive rollout where error compounds through feedback. Reporting our within-horizon metric as unqualified "VPT" in a paper supervised by a CNB co-author invites a definitional challenge. VFH is Henok's own suggested term. Cite Pathak 2018, Gilpin 2023, and CNB on first use; state in one sentence that VFH is not rollout VPT.

**Normalisation (differs from CNB — state it, don't hide it):** all NRMSE uses the **frozen train-split per-node σ** (leakage guard), so NRMSE = 1.0 ⇔ climatology-equivalent error. CNB normalises by the test trajectory's own σ. Ours is the leakage-safe choice and arguably cleaner; it is a deliberate deviation, disclosed in one line.

**Censoring notation (binding):** floor values render as **≤1** (crossing at step 1); ceiling values as **≥12** (VFH) or **≥12K** (RVPT). Numeric storage keeps the existing integer convention (`0` for step-1 crossing, `H` for never-crossed); the presentation layer applies the notation. Every distribution summary reports `%censored_low` and `%censored_high` explicitly.

---

## 2. Shared conventions

- **Variable order (everywhere):** `[t2m, d2m, tp, u10, v10, sp, ws10, rh]` — matches `test_metrics.json`.
- **Shapes:** direct forecasts `(B, H=12, N=999, d=8)`; per-sample curves `(S, 12, 8)`; rollout curves `(S₀, 12·K, 8)`.
- **Z-space computation:** per-sample NRMSE at (s, h, v) = √(mean over nodes of squared z-space error). Because normalisation is per-node×variable with train stats, pooling nodes in z-space weights each node by its own climatological σ — identical convention to the existing pipeline. No re-division by any test-set std (the historical over-normalisation bug must not be reintroduced; see comment at `regression.py:94-97`).
- **Crossing function:** reuse `evaluation/regression.py::valid_prediction_time(curve, epsilon)` unchanged (first crossing; dips back under do not rescue).
- **ε grid:** `{0.15, 0.2, 0.3, 0.5, 0.75, 1.0}`. Purpose per band: 0.15–0.2 de-censors CLCRN t2m/sp; 0.3 continuity with existing deck; 0.5–0.75 lifts Tier-2 off the floor (Henok's suggested band); 1.0 = climatology-equivalence, CNB/Pathak semantics.
- **Determinism:** `model.eval()`, `torch.no_grad()`, fixed dataloader order, no dropout/sampling at inference.
- **Statistical treatment (anti-pseudoreplication rail):** inference stays at seed level. Per-sample/per-start values within a seed are overlapping windows of one test period and are **not** independent; they are summarised (median, IQR, censoring fractions, survival curve) per seed, then the 10 seed medians feed the existing Wilcoxon machinery (floor p = 0.00195 at n = 10, unchanged). Start-level distributions are reported descriptively only. CNB's stride-12 "reduce pseudoreplication" note is the precedent.

---

## 3. Deliverable A — ε-sensitivity grid

**Inputs:** per-seed `test_metrics.json` → `nrmse_per_horizon` dicts (pooled curves, per variable), for all 30 seeds (3 models × 10), plus the two baseline JSONs (`regression-persistence.json`, `regression-climatology.json`).

**Computation:** for each model × seed × variable × ε in grid: `valid_prediction_time(curve, ε)`. Aggregate per model × variable × ε: mean, median, min, max over seeds, plus counts censored at 0 and at 12. Baselines: single deterministic curve each, same grid.

**Outputs:**
```
outputs/vpt_extended/eps_grid/
  {model}_eps_grid.json        # {var: {eps: {mean, median, min, max, n_floor, n_ceiling, per_seed:[...]}}}
  baselines_eps_grid.json
  figures/eps_sensitivity_{var}.png   # VFH vs ε, one line per model + persistence
```

**Acceptance:** ranking CLCRN > MTGODE ≥ AGCRN preserved at every ε where any model is off the floor/ceiling; CLCRN t2m and sp de-censored (finite VFH) at ε ≤ 0.2 — if not, report as a finding, not a failure. Runtime: minutes, local, CPU. **No HPC required** beyond an `rsync` of the small per-seed JSONs if any are missing locally.

---

## 4. Deliverable B — per-sample VFH

### 4.1 What changes

Current: one NRMSE curve pooled over all test samples and nodes → one VFH per variable. Failure mode (Henok, verbatim concern): "early failure for some samples may be averaged out." A pooled 7.1 h cannot distinguish "all forecasts degrade gently past 7 h" from "80% survive to 12 h, 20% blow up at 2 h." At the floor, AGCRN and MTGODE both read 0 with no resolution despite a ~25% pooled-NRMSE gap.

New: per test sample s, compute curve `(12, 8)` by pooling **nodes only**; apply first-crossing per sample per variable per ε; summarise the distribution.

### 4.2 Streaming computation (memory rail — non-negotiable)

The historical 141 GB eval peak came from accumulating full `(S, H, N, d)` preds+targets and denormalising copies. This job must never materialise those arrays.

Per batch: run `pred = adapter.predict(x, adj)` → `(B, 12, 999, 8)`; compute `sq = (pred − y)²` in z-space; reduce nodes → `(B, 12, 8)`; `sqrt` → append to a growing `(S, 12, 8)` float32 array (~18 MB total at S ≈ 46k); discard pred/target. Peak memory target < 40 GB (cache ~22 GB + model + one batch); request `--mem=128G` as margin, never rely on it.

### 4.3 Per-seed outputs

```
outputs/vpt_extended/per_sample/{model}/seed_{s}/
  vfh_curves.npz        # nrmse: (S, 12, 8) float32; sample_t0: (S,) int64 (test-set index of forecast origin)
  vfh_values.npz        # vfh: (S, 8, n_eps) int8, eps_grid stored as attr
```

Per-model summary (after all 10 seeds):
```
outputs/vpt_extended/per_sample/{model}_vfh_summary.json
# per variable × ε: {median_of_seed_medians, iqr, pct_floor, pct_ceiling,
#                    survival_curve: [frac_valid_at_h for h in 1..12],  # mean over seeds
#                    per_seed: {seed: {median, iqr, pct_floor, pct_ceiling}}}
```

Figures: per-variable **survival curves** (fraction of forecasts still valid at each lead time; one line per model, persistence dashed) at ε = 0.3 and ε = 1.0. These are the headline visual upgrade.

### 4.4 Execution plan

- One SLURM job per model (3 jobs total — respects QOS 5-job cap without the watcher), seeds processed sequentially inside the job. `--partition=l40s --account=l40sugrd --mem=128G --time=24:00:00` (CLCRN is the slow one; each seed ≈ one ordinary test eval).
- **Resumability rail:** before each seed, skip if its `vfh_values.npz` exists and passes a size/NaN check. Write per-seed outputs immediately on completion.
- **Head-node rail (standing policy):** everything via `sbatch`. No `ssh hpc "python …"`, no `singularity exec` on the head node, ever.

### 4.5 Consistency check (required)

For each seed, the node-pooled mean of per-sample curves must reproduce the stored pooled `nrmse_per_horizon` within float tolerance (rtol 1e-3). Mismatch = stop, investigate, do not proceed to summaries. This is the guard against silently computing a different quantity.

---

## 5. Deliverable C — block-autoregressive rollout VPT (gated)

### 5.1 Why it is mechanically available

The task geometry gives **closure**: input `(12, N, 8)`, output `(12, N, 8)` — the predicted block exactly replaces the input window, and the output contains every channel the model consumes. All three models satisfy this through the shared `forward(x, adj) → pred` adapter contract. CNB precedent covers the "trained on supervised windows, evaluated under rollout" objection explicitly; exposure bias under feedback is the measurement, not a confound.

### 5.2 Stage-0 verification (blocking)

Before any rollout code: audit each adapter's true input spec. Confirm inputs are exactly the 8 z-scored channels (no calendar encodings, no static coordinate channels concatenated at input). Evidence so far (trainer contract, USTD notes) says yes; verify by reading `predict()`/`forward()` for AGCRN, MTGODE, CLCRN adapters and the dataset `__getitem__`. If any auxiliary deterministic features exist, they are supplied from **truth** (calendar) or **statics** (coordinates) at every rollout step — closure still holds; the loop just carries them alongside. Record the audit result in `PATCHES.md`-style note.

### 5.3 The loop (reference semantics)

```python
window = truth_z[t0-12 : t0]                      # (12, N, 8) z-scored, ground-truth seed
curve  = []                                        # per-step NRMSE, per variable
for k in range(K):                                 # K = 4 → 48 h (⚠ decision: 4 vs 6)
    pred = adapter.predict(window, adj)            # (12, N, 8)

    # ---- feedback hygiene, in PHYSICAL units ----
    p = denorm(pred)                               # per-node train stats
    p[..., TP]   = np.clip(p[..., TP], 0, None)    # no negative precipitation
    p[..., WS10] = np.hypot(p[..., U10], p[..., V10])          # re-derive, don't feed back model's ws10
    p[..., RH]   = np.clip(magnus_tetens(p[..., T2M], p[..., D2M]), 0.0, 100.0)  # re-derive + clamp
    pred = norm(p)

    truth_block = truth_z[t0 + 12*k : t0 + 12*(k+1)]
    curve.append(per_step_node_pooled_nrmse(pred, truth_block))   # (12, 8)
    window = pred                                  # ← the rollout step
curve = concat(curve)                              # (12K, 8)
rvpt[v] = valid_prediction_time(curve[:, v], eps)  # per start, per variable, per ε
```

**Hygiene rationale (goes in methods):** the model's predicted `ws10` will not equal `hypot(u10, v10)` of its own predicted components; feeding back internally inconsistent channels pushes the model off the training-data manifold for reasons that are artifacts of channel redundancy, not forecast error. Re-derivation reproduces exactly how the dataset was constructed, so fed-back windows share the training data's internal structure. `tp ≥ 0` for the same reason. `d2m` is left untouched; the `rh` clamp absorbs any transient `d2m > t2m`.

`magnus_tetens` must be **imported from / bit-identical to** the preprocessing implementation (constants 17.625 / 243.04 or whatever `era5_parser.py` actually uses — verify, do not assume). Same for the ws10 derivation. Unit test parity against the preprocessing functions on known values.

### 5.4 Guards

- **NaN/Inf in `pred`:** terminate that start's rollout at the current step; record crossing at that step; increment `n_numerical_failures`; continue the job. (MTGODE's ODE on off-manifold input is the expected offender.)
- **Early stop:** once a start's NRMSE exceeds 5.0 on all 8 variables, stop extending that start (crossings already recorded).
- **Mean-collapse screen (CNB analog):** flag starts whose curve sits in [0.9, 1.1] for ≥ 12 consecutive steps without crossing 1.0 — these are climatology-collapse, not tracking. Report `pct_collapsed` per model; with first-crossing at ε = 1.0 most collapse crosses anyway, but the screen catches hovering. Eyeball ≥ 5 flagged trajectories during the pilot.
- **Regression guard (required test):** block k = 0 of the rollout must equal the model's ordinary direct 12-step forecast for the same `t0` (bitwise in z-space, before hygiene). If not, the loop is wired wrong.

### 5.5 Starts, cost, baselines

- Start points: every `t0` with `t0−12 ≥ test_start` and `t0+12K ≤ test_end`, stride 24 → ≈ 1,800–1,900 starts. Starts batch together: K sequential batched passes of ~1,900 samples per seed ≈ **less compute than one stride-1 test eval**. 3 jobs (one per model, 10 seeds sequential), `--mem=128G --time=12:00:00`, same resumability and head-node rails as B.
- **Reference rows:** persistence RVPT (copy `truth[t0−1]` forward 12K steps) and climatology RVPT (flat 1.0 curve, never crosses ε = 1.0 → ≥12K by definition; report for completeness). Computed analytically from truth, no model.
- Float16 storage for rollout curves is acceptable (`(1900, 48, 8)` per seed ≈ 1.2 MB); values stay < 5.0 by the early-stop rule.

### 5.6 Outputs

```
outputs/vpt_extended/rollout/{model}/seed_{s}/
  rollout_curves.npz    # nrmse: (S0, 12K, 8) float16; start_t0: (S0,) int64; flags: (S0,) uint8 bitmask {nan, early_stop, collapse}
  rvpt_values.npz       # rvpt: (S0, 8, n_eps) int16
outputs/vpt_extended/rollout/{model}_rvpt_summary.json   # same summary schema as VFH, plus n_numerical_failures, pct_collapsed
figures/rvpt_survival_{var}.png, rvpt_box_{var}.png
```

### 5.7 Gate

Deliverable C proceeds only if **all** of: (i) Moodley sign-off (framed as the zero-retraining partial answer to the 24 h-horizon question); (ii) the faithful-ClimODE spike's post-Aug-5 budget is not compromised — the spike outranks this; (iii) B's pilot is green. **Hard kill: if C is not producing summaries by Aug 14, it is dropped to future work with no extensions.** If the faithful ClimODE passes its own gates later, it plugs into B and C unchanged via the adapter contract, using its predictive mean μ as the point forecast.

---

## 6. Sequencing, gates, and schedule

| Stage | What | Where | Gate to pass |
|---|---|---|---|
| G0 | Adapter input audit (§5.2) + re-derivation parity tests vs preprocessing | local CPU | audit note written; parity tests green |
| A | ε grid from stored curves | local CPU | acceptance in §3; ~0.5 day |
| G1 | B end-to-end smoke at N=10, 1 seed, tiny test slice | local CPU | shapes, consistency check (§4.5), summaries produced |
| B-pilot | CLCRN seed 42 alone on HPC | sbatch, l40s | consistency check on real data; wall time recorded |
| B-full | 3 model jobs, 10 seeds each | sbatch | all 30 `vfh_values.npz` present; summaries + survival figures |
| G2 | C decision point | — | §5.7 gate (Moodley + budget + B green) |
| C-pilot | CLCRN seed 42, K=4, stride 24 | sbatch | regression guard (§5.4) green; ≥5 flagged trajectories eyeballed |
| C-full | 3 model jobs | sbatch | summaries + figures; **kill date Aug 14** |

Deadline fit: A + B complete before the Aug 10 scaffold; C figures can land by ~Aug 12 without blocking writing. Nothing here competes with the faithful-ClimODE spike for the Aug 5 window — A is local, B is small and can queue behind the spike.

---

## 7. Required tests (`tests/test_vpt_extended.py`)

Mirror the style of `tests/test_evaluation.py` (synthetic arrays, no real data):

1. `test_vfh_per_sample_known_values` — hand-built per-sample curves → exact crossings.
2. `test_vfh_floor_and_ceiling` — curve[0] ≥ ε → 0; never-crossing → 12; summary reports `pct_floor`/`pct_ceiling` correctly.
3. `test_survival_curve_monotone` — survival fraction non-increasing in h.
4. `test_pooled_consistency` — mean of per-sample curves reproduces pooled curve (synthetic; mirrors §4.5).
5. `test_rollout_first_block_equals_direct` — §5.4 regression guard, with a stub adapter.
6. `test_rollout_window_replacement` — after step k, window ≡ hygiene-processed pred of step k.
7. `test_rederive_ws10_parity` / `test_rederive_rh_parity` — against the preprocessing implementations on known values (including a d2m > t2m case → rh clamps to 100).
8. `test_tp_clamp_nonnegative`.
9. `test_nan_terminates_start_not_job` — stub adapter emits NaN at block 2; other starts unaffected; flag set.
10. `test_eps_grid_shapes_and_types` — Deliverable A output schema.
11. `test_no_full_pred_accumulation` — B's streaming path never holds more than one batch of `(B, H, N, d)` (assert via a memory-counting stub or by API construction: the reducer receives batches, not the full array).

All green locally before any sbatch submission.

---

## 8. Reporting rules (binding for deck and paper)

1. VFH and RVPT never share a table or figure without their qualifier; captions name the metric, ε, and cap.
2. Censoring notation per §1 in every table; every distribution summary carries `pct_floor` / `pct_ceiling`.
3. Methods text states: metric definitions with Pathak 2018 / Gilpin 2023 / CNB citations; the frozen-train-σ normaliser and its deviation from CNB; the feedback-hygiene rules; the seed-level inference rail.
4. Decoding-strategy note carried wherever VFH/RVPT rankings appear: CLCRN's decoder is internally autoregressive over its 12 steps while AGCRN/MTGODE emit all steps at once; this confound biases **against** CLCRN (error compounding is the harder setting), so the reported geometry advantage is, if anything, understated. Under RVPT, expect feedback to be less kind to CLCRN than within-window numbers were — if its lead survives, that is a stronger headline; if it narrows, that is a real finding about where geometric priors earn their keep. Either way it is reported, not adjudicated in advance.
5. CNB cross-reference: AGCRN's instability there (valid rollouts 24→18→0, "optimization-level failure," "architectural choice, not scale, determines robustness") is cited as an independent synthetic-system replication of our seed-variance and compute-inversion findings.
6. RVPT partially answers the Moodley 24 h question: evaluation beyond 12 h without retraining, with the honest caveat that models were trained as 12-step forecasters and RVPT measures survival past the training regime.

---

## 9. Decisions required

| # | Decision | Owner | Recommendation |
|---|---|---|---|
| D1 | Adopt "VFH" naming + qualified definitions in deck & paper | Praise | Yes — costs a find-and-replace, buys definitional safety with a CNB co-author supervising |
| D2 | Headline ε for VFH | Praise | Keep 0.3 for continuity; full grid in appendix; survival curves shown at 0.3 and 1.0 |
| D3 | Rollout length K | Praise | K=4 (48 h); extend to 6 only if CLCRN sp still censored at ε=1.0 |
| D4 | Run Deliverable C at all | **Moodley** | Propose as the zero-retraining answer to the 24 h question; spike retains priority; kill date Aug 14 |
| D5 | RVPT placement | Praise + Moodley | Appendix + one main-text figure if results are clean; main text only if it resolves the Tier-2 tie decisively |
| D6 | Faithful ClimODE inclusion in B/C if its gates pass | Praise | Yes, automatic via adapter contract, using μ |

---

## 10. File map (new code only)

```
evaluation/vpt_extended.py        # per-sample VFH, survival, summaries, ε grid; reuses valid_prediction_time
evaluation/rollout.py             # rollout loop, hygiene, guards; imports derivations from preprocessing
scripts/run_eps_grid.py           # Deliverable A (local)
scripts/run_per_sample_vfh.py     # Deliverable B (sbatch entrypoint; per-model, seeds sequential, resumable)
scripts/run_rollout_vpt.py        # Deliverable C (sbatch entrypoint; same pattern)
slurm/vfh_per_model.sh            # --partition=l40s --account=l40sugrd --mem=128G --time=24:00:00
slurm/rvpt_per_model.sh           # --mem=128G --time=12:00:00
tests/test_vpt_extended.py
outputs/vpt_extended/…            # per §3/§4/§5 schemas
```

Existing files untouched except: none required. (`analyze_matrix.py` may later consume the new summaries; that is follow-up, not this spec.)
