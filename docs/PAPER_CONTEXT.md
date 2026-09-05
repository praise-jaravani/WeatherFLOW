# WeatherFLOW — Paper-Writing Context Index

**Purpose:** single-file entry point for the paper-writing agent. Every document in the repo that is relevant to Introduction, Background & Related Work, or Experimental Design is listed below with:
- **What it is** — one-line summary
- **Paper section** it supports
- **Ready-quality** — whether it contains paper-ready prose blocks / tables that can be lifted directly, versus notes/spec material that needs synthesis
- **Cite-anchors** — specific §s / line ranges when a document contains a discrete artefact (a disclosure paragraph, a decision table, a diagram)

**What we have vs what we're waiting for:**
- ✅ Experimental setup fully documented (Phase 1 regression + Phase 2 classification, all model integrations, HPO methodology, baselines, diagnostics)
- ⏳ **Results are pending** — MTGODE HPO complete, STGODE HPO ~60% done on pod, full Phase 1/2 campaign has not been submitted yet. The paper-writing agent should NOT write "Results" sections.

---

## 1. Top-level project descriptors

| File | What it is | Paper section | Ready? | Notes |
|---|---|---|---|---|
| [README.md](../README.md) | Project overview — task, motivation, model families, dataset, team, layered architecture (OpenSTL-inspired) | **Introduction** | Directly | Contains the one-paragraph mission statement + model table. First place to draw from. |
| [specs/project_overview.md](../specs/project_overview.md) | Longer-form project scope, layer separation rationale, phase structure | **Introduction**, **Method: framework** | Directly | The "why this design" prose lives here. |
| [results/README.md](../results/README.md) | 32-station SAWS network definition + Davidson (2022) selection criterion | **Method: data** | Directly | Contains network-selection prose, N=32 justification, per-station missingness disclosure requirement |
| [specs/status_and_roadmap.md](../specs/status_and_roadmap.md) | Live phase tracker | **Not for paper** — internal state | — | Skip for paper; use for orientation |

---

## 2. Background & Related Work

**No dedicated "related work" spec exists** — the citations are distributed across the model-integration docs and the Phase-2 formulation. Consolidate from:

| Source | Contribution | Paper section |
|---|---|---|
| [docs/PHASE2_FORMULATION.md](PHASE2_FORMULATION.md) §6 | **9-citation literature trail** for the extreme-detection formulation — Zhang et al. 2011 (ETCCDI), SAWS operational heatwave, Zscheischler 2020 (compound events), WMO SPI, Xu 2024 (ExtremeCast), Pasche 2025 (named-event validation), Nurmohamed 2026 (ML-heatwave review), Wang 2025 (marine-heatwave review), Galib 2022 (DeepExtrema), Kim 2025 (ClimateNet). Each with what it contributed. | Related Work — extreme detection & labelling; Background — meteorological setup |
| [specs/HYPERPARAMETERS.md](../specs/HYPERPARAMETERS.md) | Per-model paper citations for architectural choices: **Bai et al. 2020** (AGCRN), **Lin et al. 2022** (CLCRN), **Jin et al. 2022** (MTGODE), **Fang et al. 2021** (STGODE), **Gaibie et al. 2024** (SAWS predecessor HPO) | Related Work — STGNN architectures |
| [README.md](../README.md) | **OpenSTL (Tan et al. NeurIPS 2023)** as the layered-benchmarking-framework inspiration | Related Work — benchmarking frameworks |
| [specs/PHASE1_25NODE_CAMPAIGN.md](../specs/PHASE1_25NODE_CAMPAIGN.md) | Historical 25-node campaign (2013-2020 Lekuba data) — pre-audit reference for how model rankings shift under implementation fixes | Related Work — motivating this study's improvements; or Discussion |
| Model adapters (`models/continuous_time/*.py`, `models/baselines/*.py`) | Each adapter's docstring lists the upstream paper + verified implementation details | Related Work — model descriptions |

**Recommendation:** the paper-writing agent should read `docs/PHASE2_FORMULATION.md §6` verbatim first — it already contains 9 fully-scoped citations with contributions.

---

## 3. Experimental Design

This is the biggest section. Divided by sub-topic below.

### 3.1 Data pipeline (SAWS ingestion → cache)

| File | What it is | Ready? |
|---|---|---|
| [specs/preprocessing_design.md](../specs/preprocessing_design.md) | Six-step pipeline: parse → derive → impute → split → normalise → adjacency. Cache hashing scheme, hash inputs, deterministic-per-config guarantees. | Directly (with light prose editing) |
| [preprocessing/saws_parser.py](../preprocessing/saws_parser.py) — module docstring | Rain-blank-is-zero convention, canonical 87672-hour grid, Davidson criterion | Directly — cite for data ingestion prose |
| [results/README.md](../results/README.md) | N=32 selection + 32-vs-24 tradeoff table | Directly |

### 3.2 Adjacency graph construction

| File | What it is | Ready? |
|---|---|---|
| [specs/ADJACENCY_AUDIT.md](../specs/ADJACENCY_AUDIT.md) | Final adjacency spec — Gaussian-kernel with distance threshold; σ=200 km, cutoff=500 km; SAWS-specific parameter choices | Directly |
| [specs/Adjacency_Audit_Spec.md](../specs/Adjacency_Audit_Spec.md) | Design rationale + alternatives considered (KNN, unnormalised, per-node σ) | Synthesise |

### 3.3 Loss, horizon, evaluation metric decisions

| File | What it is | Ready? |
|---|---|---|
| [specs/HORIZON_EXTENSION.md](../specs/HORIZON_EXTENSION.md) | Why H=48h (Moodley's 24h+ demand); horizon-truncated evaluation at h ∈ {3, 6, 9, 12, 24, 36, 48} | Directly |
| [specs/HYPERPARAMETERS.md § Training loss (2026-08-08 decision)](../specs/HYPERPARAMETERS.md) | Uniform MAE decision + reasoning (previous MSE/L1 heterogeneity across model workstreams) | Directly — the "why MAE" paragraph is written |
| [specs/VPT_Extended_Spec.md](../specs/VPT_Extended_Spec.md) | Valid Prediction Time (VPT) metric definition, NRMSE curve, Pathak et al. 2018 ε=0.3 threshold provenance | Directly |
| [specs/CROSS_FAMILY_COMPARISON_AUDIT.md](../specs/CROSS_FAMILY_COMPARISON_AUDIT.md) | Cross-family (baselines vs continuous-time vs lightweight) comparison methodology, airtight-clause additions from Chikwanha/Fuyana coord | Directly for methods; some parts are internal-coord (skip) |

### 3.4 Per-model hyperparameters (Phase 1 regression)

| File | What it is | Ready? |
|---|---|---|
| [specs/HYPERPARAMETERS.md](../specs/HYPERPARAMETERS.md) | **Single source of truth** — per-model table with every value's paper citation. AGCRN, CLCRN, MTGODE, STGODE. Contains the AGCRN lr-repair disclosure paragraph (§AGCRN learning-rate repair) verbatim paper-ready. Also contains MTGODE pre-audit quarantine label rules (§MTGODE + §2026-08-10 update block). | **Highest priority for lift-and-adapt** — much of the paper's Method section maps 1:1 to this file's tables. |
| [experiments/configs/*.yaml](../experiments/configs/) | Each config carries inline paper-citation comments per knob (e.g. `agcrn.yaml` has Bai + Gaibie citations) | Directly — cross-check against HYPERPARAMETERS.md |

**AGCRN lr repair (2026-08-13):** the diagnostic that repaired AGCRN's transplanted lr=0.01 to Bai's optimum 0.003 (42% val-MAE reduction). Contains a pre-drafted disclosure paragraph paper-ready.

- Report: [results/diagnostics/agcrn_n32_health/REPORT.md](../results/diagnostics/agcrn_n32_health/REPORT.md) — full evidence, 4-lr comparison table, val-curve plot ([val_curves.png](../results/diagnostics/agcrn_n32_health/val_curves.png))
- Disclosure paragraph: [specs/HYPERPARAMETERS.md § AGCRN learning-rate repair (2026-08-13)](../specs/HYPERPARAMETERS.md) — copy-paste-ready

### 3.5 Model integration audits

| File | What it is | Ready? |
|---|---|---|
| [specs/STGODE_INTEGRATION_AUDIT.md](../specs/STGODE_INTEGRATION_AUDIT.md) | Full peer-review audit + response for STGODE — paper spatial kernel (Eq 4), semantic adjacency (Eq 6), post-construction ODE-time override, gradient smoke, adapter divergences. Contains §0 response-to-review + §11 M7 inventory + §12 STGODE Phase-2-classification-incompatibility rationale + §13 re-smoke evidence. | Directly for method; §12 for scope (STGODE excluded from Phase 2 with reason) |
| [models/continuous_time/mtgode_model.py](../models/continuous_time/mtgode_model.py) — module docstring | MTGODE audit findings 0, 2, 5 (adjoint gradient severing, node_dim/hidden_dim conflation, CGP coefficient α=1→2) with paper-citable prose | Directly |
| [specs/HYPERPARAMETERS.md § MTGODE](../specs/HYPERPARAMETERS.md) | Contains "MTGODE (pre-audit) never pooled with N=32 numbers" disclosure paragraph + full paragraph on the three audit findings (line 457 has the paper-ready version) | Directly — the audit paragraph is written for the paper |
| [specs/PHASE1_25NODE_CAMPAIGN.md § 507](../specs/PHASE1_25NODE_CAMPAIGN.md) | Alternate wording of the MTGODE pre-audit disclosure — pick whichever fits paper voice better | Choose one |

### 3.6 Phase 2 (classification) — labels, baselines, formulation

| File | What it is | Ready? |
|---|---|---|
| [docs/PHASE2_FORMULATION.md](PHASE2_FORMULATION.md) | **THE Phase-2 methodology document.** Problem statement, exact label math (§2), baseline suite (§3), pre-registered decision rules (§4), scope (§5), literature trail (§6), follow-on decision (§7). Written to paper standard — draft your Methods and part of your Related Work by copy-adapting this. | **Directly, top-to-bottom** |
| [results/diagnostics/phase2_labels/REPORT.md](../results/diagnostics/phase2_labels/REPORT.md) | Audit trail — cache-hash-invariance rationale, threshold-fit statistics, two-scheme climatology comparison table (5.98× / 2.57× / 3.08× / 2.44× static vs 0.93× / 0.87× / 0.90× / 1.30× CH), pos_weight before/after, plumbing smoke verdict, verification against every acceptance criterion | Directly — Appendix / supplementary |
| `results/diagnostics/phase2_labels/window_sweep.{json,md}` **(will exist after sweep completes ~90 min)** | Window-selection sweep — 12 candidates evaluated on cell count / real deseasonalisation / synthetic control | Directly — Appendix table for the window justification |
| [specs/WeatherFLOW_Classification_Head.md](../specs/WeatherFLOW_Classification_Head.md) | Classification-head design decisions: multi-label per-variable BCE with pos_weight; val-selected τ per variable; PR-AUC / F1 / CSI metric suite | Directly |

### 3.7 HPO (hyperparameter optimisation) methodology

| File | What it is | Ready? |
|---|---|---|
| [specs/OPTUNA_HPO.md](../specs/OPTUNA_HPO.md) | HPO v2 design — sequential Optuna TPE + MedianPruner, in-process trainer coupling, `trial.report()` + `should_prune()`, best-so-far reporting to survive cosine-restart bumps, startup RUNNING→FAILED guard, fixed trial seed, target=t2m protocol | Directly |
| [specs/HYPERPARAMETERS.md § HPO](../specs/HYPERPARAMETERS.md) | Historical HPO v1 (random-search) PARKED disclosure + HPO v2 design + per-model SEARCH_SPACE definitions with paper citations for each candidate value | Directly — the SEARCH_SPACE tables ARE the paper's HPO tables |
| [scripts/hpo_run_optuna.py](../scripts/hpo_run_optuna.py) — module docstring | Design rationale for sequential (not parallel) execution; SQLite storage choice; walltime-chain via afterany | Directly cite for infrastructure prose |
| [runs/hpo_v1_random_search_parked/](../runs/hpo_v1_random_search_parked/) | Preserved HPO v1 artefacts (superseded, retained for provenance) | Provenance — reference in Discussion |

### 3.8 Runtime infrastructure (HPC + parallel workers)

| File | What it is | Ready? |
|---|---|---|
| [specs/HPC_Parallel_Worker_System.md](../specs/HPC_Parallel_Worker_System.md) | Manifest-based worker chain (§1.3 completion contract, §2 breadth-first ordering, §5 divergence handling, K=3 co-tenancy) | Directly for methods (Reproducibility / Computational protocol) |
| [specs/Run_Plan_Spec.md](../specs/Run_Plan_Spec.md) | Task manifest spec — 160 regression + 30 classification = 190 tasks; round-robin-by-seed slow-first ordering; deadline-insurance rationale | Directly |
| [specs/WeatherFLOW_HPC_Runbook.md](../specs/WeatherFLOW_HPC_Runbook.md) | Container + pypkgs + sbatch layout; head-node preflight; walltime chain semantics | Directly for reproducibility appendix |
| [scripts/hpc_preflight.sh](../scripts/hpc_preflight.sh) | Head-node-safe preflight — verifies environment before HPO submission (bootstrapped from earlier failures) | Cite for reproducibility (post-mortem-inspired) |

### 3.9 Baselines

| File | What it is | Ready? |
|---|---|---|
| [scripts/compute_baselines.py](../scripts/compute_baselines.py) | Regression + classification baselines: persistence, climatology (regression); persistence, base-rate (classification) | Directly — reference for baseline math |
| [scripts/compute_seasonal_climatology.py](../scripts/compute_seasonal_climatology.py) | Phase-2 seasonal-climatology baseline (cell-frequency map with fallback ladder for sparse tp) | Directly — this is the paper's calendar-signal opponent |
| [results/baselines/*.json](../results/baselines/) | Baseline results (both schemes) — schema mirrors model classification summaries | Data for Results tables |

### 3.10 Seeds, reproducibility, evaluation

| File | What it is | Ready? |
|---|---|---|
| [training/seed.py](../training/seed.py) — module docstring | Seed protocol: 10 seeds {42..51} per (model, target); every trial's determinism guaranteed by `seed_everything()` at run start | Directly cite for reproducibility |
| [scripts/health_gate.py](../scripts/health_gate.py) — module docstring | Per-(model, target) training-health gate — codifies AGCRN diagnostic's criteria (best epoch ≥ 3, ratio ≤ 0.97) for reuse against first-completed campaign tasks | Directly for Discussion — how residual per-target risk is handled inside the campaign |
| [evaluation/regression.py](../evaluation/regression.py) — module docstring | Per-station-per-variable MAE / RMSE / sMAPE / VPT; extreme-stratified metrics; ERA5 variable naming | Cite for metric definitions |
| [evaluation/classification.py](../evaluation/classification.py) — module docstring | Multi-label F1 / PR-AUC / CSI / ROC-AUC; val-F1 threshold selection; per-station coverage | Cite for metric definitions |

---

## 4. Content the agent should NOT use

- **Anything under `results/hpo_v1_random_search_parked/`, `results_saws_25node_phase1/`, `results_archive_era5_999n_and_phase2/`, `preprocessing/cache/saws.25node.bak/`** — historical/superseded data. Reference by pointer only ("MTGODE (pre-audit) results at N=25 archived at [path]"), never draw numerical claims from.
- **`docs/FAITHFULNESS.md`, `docs/TEAMMATE_INTEGRATION.md`** — internal coord docs about the CLIMODE-faithful branch (mothballed 2026-08-11) and Praise/Génevieve/Zoleka handoffs. Not paper material.
- **`docs/REPORT.md`** — old teammate handoff report; superseded by the newer per-diagnostic REPORTs.
- **`specs/status_and_roadmap.md`** — live phase tracker; internal.
- **`tracking/`, `outputs/`, `runs/`** — ephemeral working state.

---

## 5. What isn't documented yet

The following are true but not written anywhere the agent can lift directly. Flag these for you (Praise) to draft before the paper agent needs them:

1. **Team contribution split** — README.md gestures at it (Praise / Génevieve / Zoleka workstreams) but the paper needs an explicit author-contribution statement.
2. **Compute cost / carbon disclosure** — HPC runbook has the infrastructure spec, but total GPU-hours consumed for the paper isn't tracked centrally. Estimate at write time from `sacct`.
3. **Ethics / data licence** — SAWS data is provided under a specific agreement; that agreement's citable form needs a paragraph you draft.
4. **Limitations section source material** — scattered across STGODE_INTEGRATION_AUDIT.md (adapter divergences), PHASE2_FORMULATION.md §5 (scope), agcrn diagnostic (single-seed diagnostic caveat), plumbing smoke's tp-noise footnote. Should be consolidated for the paper agent to work from.

---

## 6. Recommended reading order for the paper agent

1. [README.md](../README.md) — 5 min — what the project is
2. [docs/PHASE2_FORMULATION.md](PHASE2_FORMULATION.md) — 15 min — the Phase-2 methodology is the most-polished doc in the repo; sets the tone for what "paper-ready prose" looks like in this project
3. [specs/HYPERPARAMETERS.md](../specs/HYPERPARAMETERS.md) — 20 min — the paper's Method section will be ~30% based on this file
4. [specs/STGODE_INTEGRATION_AUDIT.md](../specs/STGODE_INTEGRATION_AUDIT.md) + [results/diagnostics/agcrn_n32_health/REPORT.md](../results/diagnostics/agcrn_n32_health/REPORT.md) + [results/diagnostics/phase2_labels/REPORT.md](../results/diagnostics/phase2_labels/REPORT.md) — 30 min — the three diagnostic threads (STGODE peer-review response, AGCRN lr repair, Phase-2 label upgrade). Each has paper-ready prose blocks.
5. [specs/preprocessing_design.md](../specs/preprocessing_design.md) + [specs/ADJACENCY_AUDIT.md](../specs/ADJACENCY_AUDIT.md) — 10 min — data + graph pipeline
6. [specs/OPTUNA_HPO.md](../specs/OPTUNA_HPO.md) + [specs/HPC_Parallel_Worker_System.md](../specs/HPC_Parallel_Worker_System.md) + [specs/Run_Plan_Spec.md](../specs/Run_Plan_Spec.md) — 15 min — HPO + infrastructure

**Everything else is skimmable or reference-only.**

---

## 7. Key numbers the paper agent will need on hand

Committed values, all traceable:

- **Network**: N = 32 SAWS stations, 2016-01-01 → 2025-12-31 hourly (87,672 timesteps), 6 canonical variables {t2m, tp, rh, wd, ws, sp}, ≤10% overall missingness (Davidson 2022 criterion)
- **Splits**: 60/20/20 chronological (52,603 / 17,534 / 17,535 hours) — Phase-1 target variables {t2m, sp, rh, ws}, Phase-2 label variables {t2m, tp, rh, ws}
- **Windows**: input_len = 12h, horizon = 48h (evaluated at h ∈ {3, 6, 9, 12, 24, 36, 48} by truncation)
- **Seeds**: 10 seeds per (model, target) — {42..51}, `seed_everything()` at task start
- **Phase 1 run count**: 4 models × 4 targets × 10 seeds = 160 tasks
- **Phase 2 run count**: 3 classification-compatible models (STGODE excluded — see STGODE audit §12) × 10 seeds = 30 tasks (multi-label multi-var per run)
- **Combined manifest**: 190 tasks, interleaved-by-seed for deadline insurance
- **Loss**: uniform MAE for regression; multi-label WBCE (per-variable pos_weight = (1−p)/p) for classification
- **Threshold percentile**: 95th; comparator strict `>`; two schemes (static per-(N,V) vs calendar-hour-relative — reviewer-selected default: `calendar_hour_relative` with day_win=7, hour_win=1 pending window-sweep confirmation)
- **HPO**: Optuna TPE + MedianPruner, 30 trials per model, seed=42 fixed per trial, target=t2m only; train_hours_cap=17520 (2 train years) per trial; MTGODE + STGODE HPO on RunPod pod (2026-08-13, ~13 GPU-hours); AGCRN + CLCRN not HPO'd (Gaibie-published values, AGCRN lr repaired to Bai's 0.003)
- **Baselines**: persistence, base-rate, climatology (regression) + persistence, base-rate, seasonal-climatology under both label schemes (classification)
- **Compute**: HPC (UCT ICTS L40S partition, `l40sugrd` QOS, 24h wallclock chain via `afterany`, `MAX_LINKS=5`, `K=3` co-tenancy) + RunPod ad-hoc for HPO overflow
