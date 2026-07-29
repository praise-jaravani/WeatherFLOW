# Phase 2 — Classification Results (Interim, live 3-model matrix)
_Snapshot generated 2026-07-29 10:46 SAST_

## Task

Per-variable extreme-event classification. Labels = 95th-percentile per (station, variable) computed on training split only. See [WeatherFLOW_Phase2_Plan.md](../analysis/WeatherFLOW_Phase2_Plan.md) for full spec.

## Live-matrix scope (as of 2026-07-26 pivot)

**ClimODE-inspired archived** pending faithful ClimODE reimplementation (`climode_faithful` on branch `climode-faithful`). Live matrix tracks 3 models: AGCRN, CLCRN, MTGODE. See [../archive/climode_inspired/PROVENANCE.md](../archive/climode_inspired/PROVENANCE.md).

## Completion status

**20 / 30 seeds landed** (target: 10 seeds × 3 live-matrix models). Watcher drip-feeding remaining 10 seeds; ETA ~5-7 days at current cadence.

| Model | Done | Missing |
|-------|------|--------|
| clcrn | [42, 43, 44, 45, 46, 47] | [48, 49, 50, 51] |
| mtgode | [42, 43, 44, 45, 46, 47, 48] | [49, 50, 51] |
| agcrn | [42, 43, 44, 45, 46, 47, 48] | [49, 50, 51] |
| climode (ARCHIVED) | [42, 43, 44, 45, 46, 47] | see PROVENANCE |

## Macro metrics — mean ± std across available seeds

| Rank | Model | n | F1 | AUC-ROC | PR-AUC | Precision | Recall |
|------|-------|---|-----|---------|--------|-----------|--------|
| 1 | **CLCRN** | 6 | 0.7112 ± 0.0029 | 0.9780 ± 0.0006 | 0.7799 ± 0.0037 | 0.6923 | 0.7333 |
| 2 | **MTGODE** | 7 | 0.5173 ± 0.0199 | 0.9301 ± 0.0064 | 0.5219 ± 0.0303 | 0.4692 | 0.5791 |
| 3 | **AGCRN** | 7 | 0.2995 ± 0.0130 | 0.8129 ± 0.0098 | 0.2295 ± 0.0151 | 0.2276 | 0.4472 |

## Wilcoxon pairwise (paired by seed)

At n=10 minimum p is 2^-9 = 0.00195. Wilcoxon requires ≥6 for p<0.05.

| Comparison | n | W | p | Winner |
|------------|---|---|---|--------|
| clcrn vs mtgode | 6 | 0.0 | 0.0312 | clcrn |
| clcrn vs agcrn | 6 | 0.0 | 0.0312 | clcrn |
| mtgode vs agcrn | 7 | 0.0 | 0.0156 | mtgode |

## Per-variable F1 (mean across seeds)

| Model | t2m | d2m | tp | ws10 | rh | sp |
|-------|---|---|---|---|---|---|
| clcrn | 0.801 | 0.756 | 0.536 | 0.638 | 0.636 | 0.898 |
| mtgode | 0.594 | 0.549 | 0.382 | 0.415 | 0.453 | 0.711 |
| agcrn | 0.297 | 0.363 | 0.226 | 0.200 | 0.243 | 0.468 |
