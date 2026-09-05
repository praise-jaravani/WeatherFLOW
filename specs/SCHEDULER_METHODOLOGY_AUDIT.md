# Scheduler Methodology Audit — CosineAnnealingWarmRestarts vs. Source Papers

**Created:** 2026-08-20
**Status:** OPEN — fix not yet applied
**Scope:** All four production models: AGCRN, STGODE, CLCRN, MTGODE

---

## 1. The Problem

Our training harness applies `CosineAnnealingWarmRestarts` (T_0=10, T_mult=2, η_min=1e-6)
uniformly to all models. This scheduler does not appear in any of the four source papers.
It was added as a "uniform harness" choice with the reasoning that uniformity enables
fair inter-model comparison. That reasoning is backwards: fair benchmarking means
reproducing each model as its authors trained it. A performance gap attributable to
scheduler choice, not model architecture, is noise, not signal.

---

## 2. What the Source Papers Actually Use

| Model | Paper | Optimizer | LR | Scheduler | Epochs | Batch |
|---|---|---|---|---|---|---|
| AGCRN | Bai et al. (2020) | Adam | 0.003 (best from search `{0.0007, 0.001, 0.003, 0.005, 0.009}`) | **None** | ≤100 (patience 15) | not stated |
| STGODE | Fang et al. (2021) | Adam | 0.01 | **None** | 200 | 32 |
| CLCRN | Lin et al. (2022), App. D1 | Adam | 0.01 | **Step decay ×0.05 per 10 ep, first 50 ep** | ≤100 | 32 |
| MTGODE | Jin et al. (2022) | Adam | 0.001 (multi-step, METR-LA) | **MultiStepLR(milestones=[100], gamma=0.1)** — paper pseudocode writes generic `LRScheduler(η, i)`; resolved from `run_multi_step.py` + `run/metr-la.sh` in official repo (2026-08-24) | 200 | 64 |

Sources confirmed directly from PDFs on 2026-08-20.

---

## 3. What Our Pipeline Uses

```yaml
scheduler:
  type: CosineAnnealingWarmRestarts
  T_0: 10
  T_mult: 2
  eta_min: 1.0e-6
```

Applied identically in `experiments/configs/agcrn.yaml`, `clcrn.yaml`, `mtgode.yaml`,
and the STGODE config.

---

## 4. Gap Per Model

| Model | Paper scheduler | Our scheduler | Gap |
|---|---|---|---|
| AGCRN | None | CosineAnnealingWarmRestarts | Added scheduler not in paper |
| STGODE | None | CosineAnnealingWarmRestarts | Added scheduler not in paper |
| CLCRN | Step decay ×0.05/10ep (first 50 ep) | CosineAnnealingWarmRestarts | Replaced step decay with cosine restarts |
| MTGODE | MultiStepLR(milestones=[100], gamma=0.1) per epoch — confirmed from `run_multi_step.py` (2026-08-24) | CosineAnnealingWarmRestarts | Does not match; paper uses a single step-drop at epoch 100, we use cosine restarts |

**Known interaction (documented):** For AGCRN, `CosineAnnealingWarmRestarts` with
lr=0.01 caused compounding training instability at N=32 — the cosine restart at epoch 10
kicks lr back to 0.01 after initial descent, causing divergence. The lr was repaired to
0.003 (Bai et al.'s own optimum) to mitigate this, but the scheduler itself remains
non-paper-faithful. See `upload/DIAGNOSTIC_agcrn_lr_repair.md` for the full diagnostic.

---

## 5. Current Campaign State (as of 2026-08-20)

The production campaign (190 tasks total) is running on HPC and locked — no mid-campaign
config changes are being made.

| Model | Regression tasks | Classification tasks | Campaign state |
|---|---|---|---|
| AGCRN | 40 (4 vars × 10 seeds) — **COMPLETE** | 10 — pending | Done; results in `results/` |
| STGODE | ~31/40 complete | 10 — pending | In-progress on HPC |
| CLCRN | 0/40 | 10 — pending | In-progress on HPC (lr repaired to 0.01) |
| MTGODE | ~16/40 complete | 10 — pending | In-progress on HPC |

All completed runs used CosineAnnealingWarmRestarts. Re-running would require
100–200+ GPU-hours per model.

---

## 6. Fix Options

### Option A — Disclose, no re-runs (pragmatic)
Add a disclosure paragraph to the paper Methods section acknowledging that
CosineAnnealingWarmRestarts was applied uniformly as a harness choice, that it is not
derived from any of the four source papers, and that AGCRN/STGODE use no scheduler in
their originals while CLCRN uses step decay. Cite this document.

**Cost:** Zero GPU-hours. Results remain as-is.
**Risk:** Reviewer may flag scheduler non-faithfulness as a methodological weakness.

### Option B — Re-run all models with paper-faithful schedulers
- AGCRN: remove scheduler entirely (constant lr=0.003, Adam, patience 15)
- STGODE: remove scheduler entirely (constant lr=0.01, Adam, 200 epochs)
- CLCRN: replace CosineAnnealing with step decay ×0.05 per 10 ep (first 50 ep)
- MTGODE: paper doesn't specify type — either remove or use a simple ExponentialLR/StepLR

**Cost:** ~400–600 GPU-hours (full 10-seed × 4-var campaigns for each model).
**Risk:** New results may differ from current results; deadline pressure is high.

### Option C — Re-run only the two clear-cut cases (AGCRN + STGODE)
AGCRN and STGODE are the most straightforward: their papers explicitly use no scheduler.
Removing CosineAnnealing for these two is a clean, defensible change.
CLCRN and MTGODE have more complex situations (step decay / unspecified decay) and could
be kept with disclosure.

**Cost:** ~200–400 GPU-hours (AGCRN 40 tasks + STGODE ~9 remaining + AGCRN cls 10 tasks).
**Risk:** Inconsistent treatment across models; requires justification.

### Option D — Retroactive ablation (post-campaign)
After the main campaign completes, run a small 1-seed ablation comparing
no-scheduler vs CosineAnnealing for AGCRN and STGODE. Report both numbers in an
appendix. Main results stay as-is; the ablation quantifies the scheduler effect.

**Cost:** ~40 GPU-hours (1 seed × 4 vars × 2 models).
**Risk:** If the effect is large, reviewers will question the main results.

---

## 7. Prescribed Fix (to be filled in once decision is made)

**Decision:** _[PENDING — to be recorded here]_

**Chosen option:** _[A / B / C / D]_

**Changes required:**
- [ ] `experiments/configs/agcrn.yaml`: _[action]_
- [ ] `experiments/configs/stgode.yaml`: _[action]_
- [ ] `experiments/configs/clcrn.yaml`: _[action]_
- [ ] `experiments/configs/mtgode.yaml`: _[action]_
- [ ] Re-generate campaign manifest(s): _[yes/no]_
- [ ] Paper Methods disclosure paragraph: _[draft text]_

---

## 8. Draft Methods Disclosure (Option A wording)

> "All models are trained with the Adam optimizer and early stopping (patience 15).
> Learning rate schedules follow each model's source paper where specified: CLCRN
> follows Lin et al. (2022) Appendix D1 (step decay ×0.05 per 10 epochs, applied
> for the first 50 epochs). AGCRN and STGODE do not specify a scheduler in their
> source papers (Bai et al. 2020; Fang et al. 2021); MTGODE (Jin et al. 2022)
> specifies learning rate decay without a named type. In our implementation, all four
> models share a CosineAnnealingWarmRestarts schedule (T₀=10, T_mult=2, η_min=1e-6)
> applied as a uniform harness to enable direct comparison under identical optimisation
> conditions. We acknowledge this departs from paper-faithful training for AGCRN and
> STGODE, and note this as a limitation."

---

## 9. Native Conditions for a Faithful Rerun

If a rerun is ever decided (Option B or C), the target configuration per model is:

| Model | Native scheduler | lr to use | Notes |
|---|---|---|---|
| AGCRN | **None — constant lr** | 0.003 (Bai §4.2 published optimum) | Bai §4.2: no scheduler mentioned. Remove `scheduler:` block entirely. |
| STGODE | **None — constant lr = 0.01** | 0.01 (Fang §5.3 verbatim) | "We train our model using Adam optimizer with a learning rate of 0.01." |
| CLCRN | **Step decay ×0.05 every 10 epochs, first 50 epochs** | 0.01 (our repaired value) | Lin et al. (2022) Appendix D1. Equivalent to `StepLR(step_size=10, gamma=0.05)` applied for epochs 1–50 only, then constant thereafter. |
| MTGODE | **MultiStepLR(milestones=[100], gamma=0.1)** — one drop at epoch 100 by ×0.1 | 0.001 (paper METR-LA default) or 0.006 (HPO v2 winner) | Resolved 2026-08-24 from local clone `models/continuous_time/MTGODE/run_multi_step.py:99` + `run/metr-la.sh`. Paper uses multi-step training (H=48 matches); `lr_decay=True` by default. `scheduler.step()` called once per epoch at `run_multi_step.py:189`. Single-step script uses different milestones=[20,40], gamma=0.5 — not relevant for our task. |

---

## 10. AGCRN Scheduler-Interaction Finding (2026-08-24)

**Scheduler step granularity confirmed:** `scheduler.step()` is called **once per epoch**
([trainer.py:282](../training/trainer.py)), not inside the batch loop. The LR is stable
within each epoch (273 batches all at the same value) and changes only at epoch boundaries.
The "per-batch oscillation" hypothesis is definitively ruled out.

**The actual failure mechanism for lr=0.01 + CosineAnnealingWarmRestarts:**

With T_0=10, T_mult=2, lr_max=0.01 stepped per epoch:
- Epoch 1: lr=0.010000 (starts hot)
- Epoch 10: lr=0.000246 (near-minimum)
- Epoch 11: lr=0.010000 (RESTART — second cycle, T_i=20)
- Epoch 16: patience fires (15 epochs since epoch-1 peak) → **killed here**

The diagnostic showed this run peaking at epoch 1 and dying at epoch 16. The interaction:
patience=15 counting from epoch 1's peak runs out 5 epochs into the second cycle, before
the restarted LR has had time to anneal. The model gets killed while still at high LR
(~0.0085 at epoch 16), never reaching annealed territory.

**Gen's result (2026-08-14):** Running lr=0.01 *without* a scheduler in her independent
AGCRN implementation, she observed 32 healthy training epochs. This is strong evidence
that lr=0.01 is not pathological in isolation — but two variables differed (no scheduler
AND different codebase), so it cannot isolate the scheduler as the sole cause.

**Implication for the paper:** The current disclosure ("lr=0.01 diverged on our N=32
network") is imprecise. The accurate framing is:

> "Gaibie's lr=0.01 was unstable under our CosineAnnealingWarmRestarts scheduler
> (peaked epoch 1, patience-killed at epoch 16 before annealing could recover);
> we adopted Bai et al.'s published optimum of lr=0.003, which is stable under
> the same harness."

The production campaign (40/40 AGCRN runs at lr=0.003 + scheduler) is valid and
internally consistent. The issue is the attribution of the failure, not the fix.

**The 2×2 that would fully resolve this:**

| | Scheduler ON | Scheduler OFF |
|---|---|---|
| lr=0.01 (Gaibie) | ✅ pathological (ep 1 peak, ep 16 death) | ⚠️ Gen: ~healthy, different codebase |
| lr=0.003 (production) | ✅ healthy (all 40 seeds) | ❓ unknown |

6 runs (2 cells × 3 seeds × t2m only) would settle the remaining unknowns.
Decision: **not yet scheduled** — paper framing fix is the minimum required action.

---

## 12. Related Documents

- `upload/DIAGNOSTIC_agcrn_lr_repair.md` — AGCRN lr=0.01 pathology at N=32; CosineAnnealing interaction documented
- `upload/SPEC_hyperparameters_all_models.md` — full per-model hyperparameter audit; §MTGODE Note, §CosineAnnealing docstring correction
- `specs/CLCRN_SP_LR_REPAIR.md` — CLCRN lr=0.05 → 0.01 repair and gate diagnostic
- `specs/HYPERPARAMETERS.md` — MTGODE HPO v2 winner record
