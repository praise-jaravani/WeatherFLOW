# Prediction Horizon Extension — Multi-Horizon SAWS Runs + Optuna Handoff

**Decided:** 2026-08-04 · **Author:** Praise Jaravani (context from Moodley meeting)
**Scope:** Replace the SAWS 12h-only training regime with 48h training + 7-horizon truncation reporting. Match Davidson & Gaibie's per-horizon reporting methodology while training only once per model.

---

## What changed

**Before:** all three models trained at `input_len=12, horizon=12` (SAWS pipeline pre-2026-08-04). VPT reported only at h=12.

**After (this document):** all three models trained at `input_len=12, horizon=48`. Evaluation reports at **seven** sub-horizons `h ∈ {3, 6, 9, 12, 24, 36, 48}` by direct truncation of the 48-step forecast — the model's step-`h` prediction is the metric for horizon `h`.

**Univariate output (2026-08-04 amendment).** Per Gaibie (2024), each experiment predicts ONE target variable at a time (input still multivariate: all 6 features as context). Production runs iterate over targets **{t2m, sp, rh, ws}**. `tp` and `wd` remain input features. This means the reported per-horizon MAE at h=3/6/9/12/24/36/48 is per-target — 4 targets × 3 models × 7 horizons = 84 numbers per metric.

**Network expansion (2026-08-04 note).** The 25-station network is what we have today. SAWS is scheduled to deliver ~20 additional stations in the same 5 provinces in the coming days, taking us to **N=45**. The geographic footprint stays identical; the network just becomes denser. HPO runs now on N=25 due to end-of-week deadline pressure; a targeted re-optimisation of topology-sensitive hyperparameters (`k_neighbors`, `graph_constructor.k`, `cheb_k`, `embed_dim`) is planned when N=45 arrives. See [OPTUNA_HPO.md](OPTUNA_HPO.md) §"N-sensitivity + planned delta-HPO" for the protocol.

**Why this recipe:** Davidson (2022) trained 5 separate models (one per horizon in {3,6,9,12,24}). Gaibie (2024) same. Training a separate model per horizon at our scale would be 7 × 3 × 10 = 210 multi-seed runs — infeasible before the Aug deadline. Instead we train once per model at the max horizon and truncate — a defensible single-model reporting variant. The model that emits 48-step forecasts obviously *contains* its 3/6/9/12/24/36-step predictions, and reporting them is the honest per-horizon skill.

---

## What was updated in code

| File | Change |
|---|---|
| [experiments/configs/mtgode.yaml](../experiments/configs/mtgode.yaml) | `horizon: 12 → 48`; input_len unchanged (12) pending Optuna |
| [experiments/configs/agcrn.yaml](../experiments/configs/agcrn.yaml) | same |
| [experiments/configs/clcrn.yaml](../experiments/configs/clcrn.yaml) | same |
| [evaluation/vpt_extended.py](../evaluation/vpt_extended.py) | Added `DEFAULT_REPORT_HORIZONS = (3, 6, 9, 12, 24, 36, 48)`; new `per_horizon_report_from_curves()` helper; `aggregate_eps_grid_across_seeds` now infers `H_cap` from data (was hardcoded to 12) |
| [tests/test_vpt_extended.py](../tests/test_vpt_extended.py) | 4 new tests covering the multi-horizon truncation + dynamic H_cap |

**Cache invalidated automatically** because `horizon` is part of `pipeline.py`'s hash key. The rebuild at horizon=48 was smoke-tested — sample counts, shapes, finiteness all verified (see `data/saws/PIPELINE_AUDIT.md`). No further action needed there.

---

## Closure-invariant impact (rollout module)

The rollout module (`evaluation/rollout.py`) requires `input_len == horizon` for the closed-loop feedback invariant (spec §5.1). With `input_len=12, horizon=48` this is broken, so **`rollout_predict` will raise** if called at this config.

**Impact:** RVPT (Deliverable C of VPT_Extended_Spec) is dormant until input_len is tuned to match. Direct VFH (Deliverable B) is unaffected and remains the primary evaluation.

**Resolution paths:**
1. Optuna lands `input_len = 48` → closure restored → rollout works at K=1 (48-step, same as direct forecast, so RVPT ≡ VFH — trivial).
2. Optuna lands `input_len = 24, horizon = 24` → closure restored → RVPT works at K=2 (48h rollout).
3. Optuna lands anything asymmetric → windowed-feedback rollout logic needed (out of scope for this doc).

Recommendation: whichever `input_len` Optuna picks, retrain the production models with `horizon = input_len` if it's ≥ 24. That preserves rollout capability at K=2 for 48h coverage, matching CNB's methodology.

---

## How to run the 7-horizon evaluation

After multi-seed training completes at `horizon=48`:

```bash
# Deliverable A (ε-grid VFH from stored pooled curves):
python scripts/run_eps_grid.py --model mtgode agcrn clcrn --dataset saws

# Deliverable B (per-sample VFH per seed):
for seed in 42 43 44 45 46 47 48 49 50 51; do
  python scripts/run_per_sample_vfh.py \
      --model mtgode --dataset saws --seed $seed \
      --checkpoint outputs/mtgode_saws_regression_<TS>/seed_${seed}/best.pt
done

# Cross-seed aggregate (produces {model}_vfh_summary.json):
python scripts/run_per_sample_vfh.py \
    --model mtgode --dataset saws --checkpoint /dev/null \
    --aggregate 42 43 44 45 46 47 48 49 50 51
```

The 7-horizon truncation report (`per_horizon_report_from_curves`) is called from within the per-sample VFH driver on each seed's curve. Aggregate output includes NRMSE at h ∈ {3, 6, 9, 12, 24, 36, 48} per variable per seed.

---

## Optuna handoff (next up)

Two things Optuna should tune, in priority order:

1. **`input_len` per model** — the biggest architectural degree of freedom left open. Davidson's range: {12, 24, 48, 72, 168}. Gaibie's CLCRN range: {48, 72, 168}. Suggested search space:
   - AGCRN: {12, 24, 48}
   - MTGODE: {12, 24, 48}
   - CLCRN: {12, 24, 48, 72}   (per Gaibie's HPO range)
2. **Per-model hyperparameters** — hidden_dim, learning_rate, dropout, etc. Existing configs already track Gaibie's HPO-optimal values; Optuna should confirm/refine.

Optimization objective: **validation MAE at h=24** — matches Davidson's HPO choice (§3.4 of his paper) and is the horizon Moodley explicitly named. Report validation MAE at all 7 horizons for the winning config to catch cases where the h=24 optimum is bad at h=48 or vice versa.

Constraint: **if `input_len` lands at any value that equals horizon (48 preferred, or 24), closure invariant is preserved → RVPT usable.** If Optuna picks an asymmetric config, flag it explicitly in the HPO report so Praise can decide whether to accept the asymmetric option or force symmetry.

Compute budget: at N=25 SAWS scale, each trial is ~5 minutes on GPU (from empirical MTGODE runs). With 50 trials × 3 models = 150 trials × 5 min ≈ 12.5 GPU-hours. Cheap.

Suggested new spec file: `specs/OPTUNA_HPO.md` (not yet written).

---

## Bottom line

- 7 horizons will be reported (3/6/9/12/24/36/48) — meeting Moodley's 24h+ demand and exceeding Gaibie/Davidson's 24h max
- One training regime per model (train-at-max-then-truncate), not seven — order-of-magnitude compute saving vs Davidson-lineage
- Cache invalidated + rebuilt, smoke-verified
- Rollout temporarily dormant until input_len tuning, direct VFH primary
- 31/31 VPT tests green including the four new horizon-extension coverage tests
