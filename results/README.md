# WeatherFLOW — Results Archive (Praise's models only)

Everything you need for post-analysis and the paper write-up lives here. No source code, no checkpoints — just measurements, plots, and the docs describing what happened.

**Scope:** this archive tracks results for **Praise's four models only** — the **baselines** (AGCRN, CLCRN) and the **continuous-time** family (MTGODE, ClimODE-inspired). Teammates run and track their own models independently: Zoleka owns STLGRU + Lite-STGNN (lightweight); Génevieve owns DiffSTG + USTD (diffusion). RESEARCH_FINDINGS still contains context sections for those models — treat those as background, not archive results.

## Structure

```
results/
├── README.md                       (this file)
│
├── analysis/                       Living context docs from experiments
│   ├── RESEARCH_FINDINGS.md        Phase 1 empirical findings, per-model narrative
│   ├── HANDOFF_STATUS.md           Live HPC state, memory peaks, known issues
│   ├── WeatherFLOW_Phase2_Plan.md  Phase 2 spec (labels, WBCE, metrics, stages)
│   └── phase2_label_audit/         Stage-0 sign-off table for Phase 2 labels
│       ├── SUMMARY.md              Positive rates, tp wet-hour trigger, seasonal/diurnal breakdown
│       └── *.csv                   Threshold tables
│
├── phase1_regression/              PHASE 1 — regression (complete, n=10 per model, 4 core models)
│   ├── {model}_regression_summary.json  TRUE aggregate: n_seeds, per-seed epochs_actually_run,
│   │                                     epochs_cap, patience, n_nodes, mae/rmse/mae_extreme with
│   │                                     mean/std/median, VPT with right-censoring counts.
│   │                                     Only the 4 core models — STLGRU/Lite-STGNN CPU smokes excluded.
│   └── runs/
│       └── {model}_seed_{N}/       One dir per (model, seed) pair — 40 total (10×4)
│           ├── test_metrics.json   MAE, NRMSE curves, VPT, per-variable
│           ├── history.json        Train/val loss per epoch
│           ├── per_station.npz     (N, d) per-station MAE arrays for geographic analysis
│           └── train.log           Full run log
│
├── phase2_classification/          PHASE 2 — classification (in progress, ~17/40 seeds so far)
│   ├── STATUS.md                   ← current results snapshot; regenerate as new seeds land
│   ├── {model}_classification_summary.json  TRUE aggregate: n_seeds, per-seed epochs_actually_run,
│   │                                     epochs_cap (heterogeneous for CLCRN — see notes), patience,
│   │                                     n_nodes, macro_f1/auc_roc/pr_auc + per-variable equivalents
│   │                                     with mean/std/median.
│   └── runs/
│       └── {model}_seed_{N}/
│           ├── test_metrics.json   Per-variable F1, precision, recall, CSI, AUC-ROC, PR-AUC
│           ├── history.json        Train/val WBCE loss per epoch
│           ├── per_station.npz     (N, d_label) per-station arrays for geographic analysis
│           └── train.log
│
├── plots/                          Phase 1 headline figures
│   ├── SUMMARY.md                  Phase 1 final analysis (matrix, Wilcoxon, per-variable)
│   ├── mae_heatmap.png             Model × variable heatmap
│   ├── mae_variance_box.png        Seed-variance stratification
│   ├── nrmse_per_horizon.png       Horizon curves (the dome-shape finding)
│   └── geographic_{var}.png        Per-station geographic maps
│
└── baselines/                          Persistence + climatology reference forecasts
    ├── SUMMARY.md                      ← start here. Skill scores, F1 lifts, paste-ready paragraphs.
    ├── skill_scores.json               Machine-readable MSSS + F1/AUC lifts per (model, baseline, variable)
    ├── regression-persistence.json     MAE, MSE, RMSE, NRMSE-per-horizon, VPT for the persistence baseline
    ├── regression-climatology.json     Same schema for climatology (train mean per station×variable)
    ├── classification-persistence.json F1, precision, recall, AUC-ROC, PR-AUC for extreme-persistence
    └── classification-climatology.json Same for constant-rate climatology (~5% positive rate)
```

## Headline results

**Phase 1 (regression, n=10 per model, COMPLETE):** see [plots/SUMMARY.md](plots/SUMMARY.md).  
Ranking on t2m MAE: **CLCRN (1.17) < ClimODE-inspired (1.57) < MTGODE (2.19) < AGCRN (2.85)**. All six pairwise Wilcoxons hit p = 0.00195 (theoretical floor at n=10).

**Phase 2 (classification, in progress):** see [phase2_classification/STATUS.md](phase2_classification/STATUS.md).  
Same ranking is replicating: **CLCRN (F1=0.71) > ClimODE-inspired (0.64) > MTGODE (0.51) > AGCRN (0.31)**.

**The story:** geometry-aware architectures (CLCRN, ClimODE-inspired) beat generic-graph architectures (MTGODE, AGCRN) by 40–60%, consistently across both regression and classification, with 10–100× smaller seed-to-seed variance. See [analysis/RESEARCH_FINDINGS.md](analysis/RESEARCH_FINDINGS.md) F2 for the full argument.

**Baseline defence:** all four models comprehensively beat the climatology baseline. CLCRN and ClimODE-inspired additionally beat the persistence baseline on every variable; MTGODE and AGCRN lose to persistence on surface pressure and dew point (further tier evidence). See [baselines/SUMMARY.md](baselines/SUMMARY.md) for the full skill-score tables and paste-ready paper paragraphs.

## What's included / excluded

- **Included**: `test_metrics.json` (all pooled + per-variable metrics), `history.json` (train/val loss curves), `per_station.npz` (station-level arrays for geographic stratification), `train.log` (full run trace including HPO decisions, memory logs, warnings), the model-level `summary.json` aggregates, Phase 1 headline figures.
- **Excluded**: `checkpoint.pt` (3.1 MB × 40+ runs = ~130 MB and not needed for post-analysis unless you want to reload models for further inference).

## How to regenerate STATUS.md as more Phase-2 seeds land

Run the Phase 2 status generator (also copies newly-completed run dirs into `runs/`):

```bash
# From project root, after rsync'ing HPC outputs:
python3 scripts/analyze_matrix.py --classification    # writes plots + SUMMARY_phase2.md
```

Or use the ad-hoc snapshot builder — see the script inline at the end of [phase2_classification/STATUS.md](phase2_classification/STATUS.md).

## Provenance

- **Data**: ERA5 reanalysis, South Africa region only, 999 stations, 2000–2026, hourly, 8 variables (t2m, d2m, tp, u10, v10, sp, ws10, rh).
- **Compute**: UCT HPC, l40sugrd QOS (5-slot cap, 24h wall, `--mem=256G`).
- **Split**: chronological 60/20/20, thresholds/normaliser fit on train only.
- **Seeds**: 42–51 (10 per model).
- **Loss**: MSE (Phase 1) → weighted BCE with per-variable pos_weight (Phase 2).
- **Metric selection**: max-F1 decision threshold τ chosen on validation split per (model, variable) — never fixed at 0.5.
