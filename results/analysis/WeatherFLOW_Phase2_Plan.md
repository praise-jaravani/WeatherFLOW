# WeatherFLOW — Phase 2 Consolidated Plan

**Version 1.0 — 9 July 2026.** Phase 2 window: 7–27 July. Progress Demonstration: 27 July.
This document consolidates the proposal requirements, the codebase audit, and a literature review of published extreme-weather classification work into one executable plan. Items marked ⚠ need team (and ideally supervisor) sign-off **before any production seeds launch**.

---

## 0. Executive summary

- **Task:** per-variable extreme-exceedance detection — parallel binary classification tasks per (station, variable, horizon step). Output `(B, H, N, d_label)` logits; labels same shape. **No ClassificationHead** — the shape mismatch is removed at the source by not collapsing labels.
- **One methodological delta from Phase 1:** the loss (WBCE via `BCEWithLogitsLoss` with per-variable `pos_weight`). Data, graphs, splits, models, seeds (42–51), and hyperparameters are all unchanged.
- **Thresholds:** 95th percentile per (station, variable), computed on the **train split only** (already what `labels.py` does; the literature confirms the code over the proposal's "test set" wording — see §2.3).
- **Two label gotchas found in the literature:** precipitation thresholds must be audited for the dry-hour degeneracy (§2.5), and the seasonal/diurnal concentration of positives must be quantified (§2.6).
- **Metrics:** proposal set (accuracy, precision, recall, F1, AUC-ROC) + PR-AUC + CSI; decision threshold selected on validation, not fixed at 0.5.
- **Scope:** the 4 Phase-1 models × 10 seeds; teammates wire their families from the worked example. Optional but recommended: a thresholded-regression baseline from Phase-1 checkpoints (§5.2) — the sharpest direct test of the DS criterion.
- **Streaming refactor: deferred** (post-demo). Measure MaxRSS at GPU smoke and lower `--mem` instead.
- **Timeline:** label audit + sign-off by Fri 11 Jul → smoke tests through Mon 13 → seeds launched Tue 14–Thu 16 → analysis 23–26 Jul → demo 27 Jul.

---

## 1. Fixed constraints from the proposal

Phase 2 reformulates prediction as binary classification for extreme event detection; no new data; models retrained with weighted binary cross-entropy as the key methodological change; extremes labelled by 95th-percentile thresholds; evaluation with precision, recall, F1, AUC-ROC (accuracy appears in the metrics section); Wilcoxon signed-rank at phase close; results presented at the Progress Demonstration on 27 July. Everything below satisfies these constraints; where this plan is more specific than the proposal, the proposal is silent, and where it corrects the proposal (threshold split), §2.3 gives the citable justification.

---

## 2. Label definition ⚠ (the decision that defines the Phase 2 result)

### 2.1 Definition (paste-ready for the paper)

> We formulate extreme event detection as parallel binary classification tasks. For each label variable *v* and station *n*, the observation at forecast step *h* is labelled extreme if it exceeds θₙ,ᵥ, the 95th percentile of that station–variable series computed on the **training split only**. This follows the reference-period convention of percentile-based extreme indices (Zhang et al., 2011, ETCCDI) and the per-variable climatological-exceedance evaluation protocol used for data-driven weather models (Lam et al., 2023; Olivetti & Messori, 2024). Labels are constructed per variable rather than pooled across variables, mirroring the per-variable stratified evaluation of Phase 1.

Shapes: labels `y ∈ {0,1}^(T, N, d_label)` cached, windowed to `(B, H, N, d_label)`; model logits `(B, H, N, d_label)`.

### 2.2 Why per-variable (evidence)

Every published classification/exceedance study found operates per phenomenon or per variable; none ORs heterogeneous variables into one class:

| Work | Label definition |
|---|---|
| GraphCast (Lam et al., *Science* 2023) | Extreme heat/cold as binary exceedance above/below a percentile of a reference climatology, computed **separately per variable, month, time of day, location** (removes diurnal/seasonal effects) |
| Olivetti & Messori (*GMD* 2024) | Cold, hot, and wind-speed extremes evaluated as **three separate categories**, thresholds per region or per grid point |
| Lopez-Gomez et al. (*AIES* 2023) | Heat only: temperature-anomaly exceedance; custom extreme-weighted losses beat MSE on the heatwave task |
| SeaCast (2025, graph-based) | SST > calendar-day 90th percentile of a satellite climatology; HSS for detection skill |
| Jacques-Dumas 2022 / Miloshevich 2023 | Heatwave occurrence (single phenomenon) as CNN classification |
| Extreme-rainfall classification (NHESS 2023, CatRaRE) | Rain only, warning-level thresholds (25 mm/h) |
| ETCCDI indices (Zhang et al. 2011) | The canonical extremes framework: all 27 indices are **single-variable**; upper and lower tails are separate indices (TX90p vs TX10p) |
| Compound-event literature | Physically motivated conjunctions (hot-AND-dry), an open research frontier — not an undifferentiated OR |

Internal reasons (from earlier analysis, all still standing): per-variable restores the ~5% positive rate that motivates WBCE (OR-of-8 was empirically heading for 15–30%, dissolving the "severe imbalance" premise); it is the exact classification analogue of Phase 1's per-variable stratified MAE and peak-intensity metrics; the ANY-label is derivable post hoc from per-variable predictions but not vice versa; and it is *less* code (§3.1).

### 2.3 Threshold split — and where the proposal's "test set" wording came from

Olivetti & Messori threshold on their evaluation year **because they evaluate pretrained models and never train anything** — with no training, there is nothing to leak into. The proposal's "consistent with standard practice" sentence almost certainly traces to that evaluation-only setting. WeatherFLOW *trains* on the labels, so thresholds must come from the training split, exactly as ETCCDI computes percentiles over a fixed base period and GraphCast builds its climatology from designated reference years. `labels.py` already does this. **Action:** none in code; the paper's methodology states train-split thresholds and quietly supersedes the proposal wording.

### 2.4 Label variable set ⚠

Recommended default: `label_variables: [t2m, d2m, tp, ws10, rh, sp]` — i.e. **exclude u10 and v10 from labels** (they remain model inputs).

- u10/v10 are signed: an upper-tail threshold flags strong westerlies/southerlies but never equally strong easterlies/northerlies — a directionally biased "extreme".
- ws10 is the correct wind-extreme target. Olivetti & Messori additionally show that errors in separately-treated u/v components compound non-linearly into wind-speed errors, especially for extremes — a citable reason to define wind extremes on speed, not components.
- sp and rh upper tails are statistical exceedances rather than hazards (high pressure ≈ anticyclonic calm); keep them but frame all labels in the paper as *95th-percentile exceedance detection*, not hazard detection. If the team prefers, dropping sp is defensible; dropping rh less so (humid heat is impact-relevant via d2m/rh).
- Cold extremes (lower tails) are out of Phase 2 scope; ETCCDI treats lower tails as separate indices, so this is a clean "future work" line, not a flaw.
- Fallback if the team wants zero deliberation: keep all 8 with the exceedance framing and a limitations paragraph. The config makes both a one-line change.

### 2.5 Precipitation special case ⚠ (Stage-0 gate)

ETCCDI's R95p computes the 95th percentile **over wet days only** (≥ 1 mm), because dry intervals dominate the distribution. At hourly resolution most hours are dry at most South African stations, so a pooled 95th percentile of tp may sit at or near **zero** — silently redefining "extreme precipitation" as "any precipitation".

**Rule (proposed):** the Stage-0 audit prints θ_tp per station. If, at a station, the fraction of zero/trace hours in train exceeds 90% (equivalently θ_tp is at trace level), compute θ_tp as the 95th percentile of **wet hours** (tp ≥ 0.1 mm/h) instead, citing the ETCCDI wet-day convention. One rule, applied identically across all three workstreams.

### 2.6 Pooled vs deseasonalised thresholds

GraphCast, SeaCast, XAI4Extremes and ETCCDI all use calendar-day or month/hour-relative climatologies; a pooled threshold on temperature mostly labels hot summer afternoons, and part of the classification signal becomes "what time of year is it".

**Decision: keep the pooled, unconditional threshold for the Phase 2 matrix.** Reasons: (i) it is the *same* definition Phase 1's stratified-MAE and peak-intensity metrics already use, preserving the regression↔classification symmetry that makes the two phases one coherent test; (ii) absolute exceedances are operationally meaningful (ETCCDI has absolute-threshold index families for exactly this reason); (iii) the demo clock. **Obligations:** the audit reports positive-rate by season and hour-of-day per variable so the concentration is quantified, the paper states the definition precisely and names climatology-relative thresholds as the standard alternative, and a month×hour-relative sensitivity run is a listed Phase 3 / discussion candidate. If time allows, per-season recall goes in an appendix.

### 2.7 `label_mode` config

`per_variable` (default) | `any`. The ANY label is **derived at analysis time only**: OR of per-variable thresholded predictions for the deterministic label; `1 − ∏ᵥ(1 − p̂ᵥ)` for a probability. Never trained on. This preserves the demo option of a single headline "any-extreme" number at zero training cost. The drafted `heads.py` is archived; it becomes relevant only if the team overrules per-variable, which the evidence argues against.

---

## 3. Model changes (deliberately minimal)

### 3.1 No classification head

With labels at `(B, H, N, d_label)`, every adapter's existing output already matches. The regression output layer's raw values serve as logits (all wrappers end in linear projections — verify per model in Stage 1); models are retrained from scratch, so the network learns the logit scale. The `max()` collapse at `dataset.py:59` is removed behind the `label_mode` flag. **This replaces the entire 16-file head-wiring plan (old item 4A):** teammates' adapters need no changes at all.

### 3.2 Loss

`BCEWithLogitsLoss(pos_weight=w)` with `w ∈ ℝ^{d_label}`, `wᵥ = N_neg,v / N_pos,v` computed from **train** labels (≈19 at a 5% rate; tp will differ under the wet-hour rule). Computed once at run start, logged to MLflow and the config hash. `pos_weight` of shape `(d_label,)` broadcasts over the trailing dimension — one-line change in `losses.py` plus vectorising `compute_class_weights`.

**No resampling.** Miloshevich et al. (2023) tested 10× majority-class undersampling directly against no undersampling: prediction skill identical within error bars; the only benefits were memory/compute, and probabilities then require correction for the undersampling rate. pos_weight is the proposal-specified mechanism and keeps the Phase 1 → Phase 2 methodological delta equal to exactly one thing. Focal loss exists in the literature (used for lightning classification) but belongs, if anywhere, in Phase 3's custom-loss extension, not in this matrix.

### 3.3 Hyperparameters

Carry Phase-1 configs unchanged; intervene per model only on demonstrated convergence failure (monitor val loss and val F1 per epoch). ClimODE-inspired keeps the adaptive solver and the 30-epoch cap — its Tier-1 Phase-1 finish closed the fixed-step-solver question by that follow-up's own decision rule.

### 3.4 Verifications folded into Stage 1

- No output activation on any wrapper's regression path (logits must be unbounded).
- Trainer casts classification targets to float for `BCEWithLogitsLoss` (already handled at `trainer.py:214-216`; re-check with 4-D targets).
- `cli.py _build_loss` wires WBCE with the vector pos_weight (old item 4D).

---

## 4. Evaluation protocol

1. **Proposal-mandated:** accuracy, precision, recall, F1, AUC-ROC.
2. **Additions:** PR-AUC (ROC curves are inflated by abundant easy negatives under heavy imbalance; PR curves are not — Sofaer et al. 2019, Saito & Rehmsmeier 2015) and **CSI** = TP/(TP+FP+FN) (the standard rare-event contingency score in forecast verification, alongside HSS/PSS — Wilks 2011; used by station-level exceedance-forecasting work and by SeaCast). CSI chosen over HSS for simplicity; swap is trivial.
3. **Decision threshold:** per (model, variable), select τ on the **validation** split by max-F1; report τ in the results table. 0.5 is not used — pos_weight deliberately shifts the operating point, so sigmoid outputs are not calibrated probabilities. AUC-ROC and PR-AUC remain threshold-free.
4. **Aggregation:** confusion counts per (station, variable) → pooled across stations per variable for the headline per-variable table → **macro-F1 across label variables** as the single-number model score. Per-station F1/recall maps for geographic analysis. NaN-skip (station, variable) cells with zero test positives — expected for tp at arid stations — mirroring Phase 1's per-seed NaN handling; report coverage counts.
5. **Statistics:** Wilcoxon signed-rank across seeds 42–51 on macro-F1 (primary) and per-variable F1 (secondary), identical procedure to the Phase-1 analysis. (If per-station significance maps are ever wanted, Olivetti & Messori's Benjamini–Hochberg FDR correction across grid points is the pattern to copy.)
6. **Trivial baselines in every table:** always-negative (F1 = 0, accuracy ≈ negative rate — which is exactly why accuracy is reported but never headlined) and the climatological-frequency predictor. Five minutes of code, permanent context.

---

## 5. Experimental matrix

### 5.1 Core (proposal-mandated)

AGCRN, CLCRN, MTGODE, ClimODE-inspired × 10 seeds (42–51) × the same dataset coverage as the completed Phase-1 matrix × classification task. Baselines trained independently; comparisons via Wilcoxon at analysis, as in Phase 1.

### 5.2 Thresholded-regression baseline (recommended; the direct DS test)

For each Phase-1 checkpoint: run **inference only** on the test set and threshold the regression forecast at the same train-split θₙ,ᵥ → binary predictions → same contingency metrics. Zero training cost. This asks the sharpest chaos-primed question available: *does distribution-sensitive training (WBCE) beat post-hoc thresholding of MSE-trained regression, per architecture?* If WBCE wins on recall/F1, that is direct evidence for the DS criterion; if it doesn't, that is a genuinely publishable negative result about where the smoothing pathology lives. Evaluating deterministic forecasts categorically is precisely the GraphCast / Olivetti & Messori mode, so the comparison is precedented.

Practicalities: requires Phase-1 raw test predictions or a re-inference pass from checkpoints (minutes–hours per checkpoint; fills queue gaps). Priority order if time is short: MTGODE + ClimODE-inspired only → defer rest to Phase 3. **Queue the WBCE matrix first; this rides in the gaps.**

---

## 6. Codebase change list

| File | Change | Replaces |
|---|---|---|
| `preprocessing/labels.py` | Confirm train-only percentiles (already true); add tp wet-hour rule (§2.5); emit threshold table artifact | — |
| `data/dataset.py` | Remove `max()` collapse behind `label_mode`; add `label_variables` selection | Old 4A (heads + 16 files) — **deleted** |
| `training/losses.py` | Vectorise `compute_class_weights` → `(d_label,)`; pass vector `pos_weight` | — |
| `evaluation/classification.py` | Per-variable loop; add PR-AUC + CSI; validation-threshold selection helper; per-station arrays with NaN-skip | Old 4F |
| `training/trainer.py` | Classification branch calls per-station save; **no streaming refactor** (deferred post-demo; exact streaming AUC is impossible without retaining scores — measure MaxRSS at Stage 4 and set `--mem` ≈ 96G instead of 256G) | Old 4B |
| `scripts/analyze_matrix.py` | Task-aware metric routing; macro-F1 Wilcoxon; per-variable tables; F1 maps | Old 4E |
| `scripts/audit_labels.py` | **NEW (Stage 0):** per-(station,variable) thresholds incl. tp diagnostics; positive rates per split, per variable, per season and hour-of-day; OR-rate for reference | — |
| `tests/test_adapters.py` | 4 tests per model, per-variable shapes: logits `(B,H,N,d_label)`; `predict()` ∈ [0,1]; regression path unchanged; one WBCE step decreases loss | Old §5 tests, adapted |
| `experiments/configs/*` | `task`, `label_mode`, `label_variables`, auto/logged `pos_weight`, `decision_threshold: val_f1` | Old 4C (no per-model tuning up front — unchanged) |

Per-model config tuning: none up front, exactly as previously agreed; revisit only on convergence failure.

---

## 7. Execution sequence

| Stage | When | What | Gate |
|---|---|---|---|
| 0 | Thu 9 – Fri 10 Jul | `audit_labels.py`; produce sign-off table (thresholds, tp diagnostics, positive rates incl. season/hour, OR-rate) | **Team + supervisor sign-off on §2 by Fri 11 Jul** ⚠ |
| 1 | Fri 10 – Sat 11 | Local unit smoke, CPU, N=10, all 4 models: shapes, probability range, regression regression-test, loss step | All tests green |
| 2 | Sat 11 – Sun 12 | Local integration: `cli.py --task classification --runs 1 --max-nodes 10 --epochs 3` for AGCRN, MTGODE, ClimODE-inspired (CLCRN needs GPU) | Finite decreasing loss; metrics JSON has per-variable P/R/F1/AUC/PR-AUC/CSI |
| 3 | Mon 13 | HPC CPU smoke (ada, N=100): container, cache, full metrics path | Clean run |
| 4 | Mon 13 – Tue 14 | HPC GPU smoke (l40s, 1 epoch): device checks, per-epoch wall time, **MaxRSS → set `--mem`** | Clean run + budget numbers |
| 5 | Launch Tue 14 – Thu 16 | Production 4 × 10 seeds via the drip-feed watcher; est. 200–600 GPU-h (~Phase-1 scale) | **Hard gate: all seeds queued by 16 Jul** |
| — | Queue gaps | Thresholded-regression baseline inference (§5.2), MTGODE + ClimODE-inspired first | Optional |
| Analysis | 23 – 26 Jul | `analyze_matrix`, Wilcoxon, per-variable tables, F1 maps, demo figures | — |
| Demo | 27 Jul | Progress Demonstration | — |

Streaming refactor, month×hour-relative threshold sensitivity, and remaining thresholded-regression checkpoints: **post-demo / Phase 3 backlog.**

---

## 8. Team coordination ⚠

Sign-off checklist (one short meeting or async doc, before Stage 5):
1. Per-variable multi-label formulation (§2.1–2.2).
2. Label variable set: 6-channel (u10/v10 excluded) vs all-8 (§2.4).
3. tp wet-hour rule and its trigger condition (§2.5).
4. Train-split thresholds — confirmation of existing code (§2.3).
5. Validation-selected decision thresholds + PR-AUC/CSI additions (§4).
6. Shared-file merge plan: `dataset.py`, `losses.py`, `classification.py`, `analyze_matrix.py` (teammates' adapters untouched by design).

Note for Génevieve: per-variable value-space labels suit diffusion models naturally — ensemble exceedance frequency *is* a probability forecast, with no conceptual awkwardness from a collapsed logit. Note for Zoleka: STLGRU/Lite-STGNN inherit the whole path with zero adapter changes once her Phase-1 runs land.

---

## 9. Phase-2 risk additions

| Risk | Mitigation |
|---|---|
| Degenerate tp threshold (dry-hour dominance) | Stage-0 gate + wet-hour rule (§2.5) |
| Zero-positive (station, variable) test cells | NaN-skip + coverage reporting (§4.4) |
| Seasonal/diurnal concentration of positives | Quantified in audit; per-season recall appendix if material; sensitivity run in backlog (§2.6) |
| Decalibrated probabilities under pos_weight | Validation-selected τ; stated in paper (§4.3) |
| Convergence differences under WBCE | Per-epoch val monitoring; per-model LR intervention only on failure (§3.3) |
| Queue congestion vs 27 Jul demo | Launch-by-16-Jul hard gate; drip-feed watcher; demo can present partial seeds with the caveat if catastrophe strikes |

---

## 10. Paper hooks

- **Methodology paragraph:** §2.1, ready to adapt.
- **Citable spine:** Lam et al. 2023 (per-variable climatological-exceedance protocol); Olivetti & Messori 2024 GMD (per-phenomenon extremes evaluation; u/v→speed error compounding; Watson-criteria framing); Olivetti & Messori 2024 GMD *review* (WBCE, Eq. 3 — already in project knowledge); Zhang et al. 2011 (ETCCDI reference-period and wet-day conventions); Lopez-Gomez et al. 2023 (extreme-weighted losses beat MSE — pre-loads Phase 3); Miloshevich et al. 2023 (reweighting vs resampling); Sofaer et al. 2019 / Saito & Rehmsmeier 2015 (PR-AUC under imbalance); Wilks 2011 (contingency skill scores).
- **The stated scientific question:** does the Phase-1 ordering (CLCRN > ClimODE-inspired > MTGODE > AGCRN) survive when the objective stops rewarding conditional-mean smoothing — and does WBCE training beat thresholded MSE-regression per architecture (§5.2)? Either answer advances the chaos-primed DS thesis.

---

## 11. Papers to add to project knowledge

| # | Paper | Link |
|---|---|---|
| 1 | Lam et al. 2023, *GraphCast* (Science; extremes protocol in §8.3 of the arXiv version) | https://arxiv.org/pdf/2212.12794 |
| 2 | Olivetti & Messori 2024, *Do data-driven models beat numerical models in forecasting weather extremes?* (GMD 17:7915) — note: 52 MB PDF | https://gmd.copernicus.org/articles/17/7915/2024/gmd-17-7915-2024.pdf |
| 3 | Lopez-Gomez et al. 2023, *Global extreme heat forecasting using neural weather models* (AIES) | https://arxiv.org/pdf/2205.10972 |
| 4 | Miloshevich et al. 2023, *Probabilistic forecasts of extreme heatwaves…* (Phys. Rev. Fluids) | https://arxiv.org/pdf/2208.00971 |
| 5 | Jacques-Dumas et al. 2022, *Deep learning-based extreme heatwave forecast* (Frontiers in Climate) | https://www.frontiersin.org/journals/climate/articles/10.3389/fclim.2022.789641/full |
| 6 | ETCCDI index definitions (Zhang et al. 2011 is paywalled; this is the canonical definitions page) | https://etccdi.pacificclimate.org/list_27_indices.shtml |
| 7 | Sofaer et al. 2019, *The area under the precision-recall curve as a performance metric for rare binary events* (Methods Ecol. Evol.) | https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13140 |
| 8 | Saito & Rehmsmeier 2015, *The precision-recall plot is more informative than the ROC plot…* (PLOS ONE) | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432 |
| 9 | Bülte et al. 2025, *GNNs for enhancing ensemble forecasts of extreme rainfall* (context for related work) | https://arxiv.org/pdf/2504.05471 |
| 10 | *Forecasting threshold exceedance of atmospheric variables at a specific location* (2026; PSS/HSS/CSI verification conventions) — PDF text extraction fails programmatically; save via browser | https://arxiv.org/abs/2605.31079 |

---

## Appendix A — Paste-able answers for the coding agent (supersedes the previous list)

1. **Label definition:** per-variable multi-label. Delete the `max()` collapse behind `label_mode: per_variable` (default) with `label_variables: [t2m, d2m, tp, ws10, rh, sp]` (pending team sign-off; all-8 fallback). Labels stay `(B,H,N,d_label)`; **no `heads.py`, no adapter wiring**. `any` is derived at analysis time only.
2. **Thresholds:** train split, per (station, variable), already correct in `labels.py`; add the tp wet-hour rule per §2.5. Build `scripts/audit_labels.py` first — nothing else proceeds until its table is signed off.
3. **Loss:** `BCEWithLogitsLoss` with per-variable `pos_weight` vector from train labels, logged. No resampling, no focal loss.
4. **Scope:** 4 Phase-1 models × 10 seeds. Teammates onboard from the worked MTGODE example; their adapters need zero changes.
5. **Metrics:** proposal set + PR-AUC + CSI; per-variable and macro-F1 aggregation; validation-selected decision threshold per (model, variable); NaN-skip zero-positive cells; trivial baselines in tables.
6. **Streaming refactor:** deferred post-demo. Measure MaxRSS at GPU smoke; set `--mem` ≈ 96G for classification jobs.
7. **New optional job type:** thresholded-regression inference from Phase-1 checkpoints (§5.2), queued in gaps after the WBCE matrix.
8. **Testing sequence:** Stages 0–5 per §7; hard launch gate 16 July.
