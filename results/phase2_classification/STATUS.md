# Phase 2 — Classification Results (Interim)
_Snapshot generated 2026-07-19 16:33 SAST_

## Task

Per-variable extreme-event classification. Labels = 95th-percentile per (station, variable) computed on training split only. See [WeatherFLOW_Phase2_Plan.md](../analysis/WeatherFLOW_Phase2_Plan.md) for full spec.

## Completion status

**17 / 40 seeds landed** (target: 10 seeds × 4 models). Remaining seeds drip-feeding via HPC watcher; ETA ~7 days.

| Model | Done | Missing |
|-------|------|--------|
| agcrn | [42, 43, 44, 45, 46] | [47, 48, 49, 50, 51] |
| mtgode | [42, 43, 44, 45, 46] | [47, 48, 49, 50, 51] |
| climode | [42, 43, 44, 45] | [46, 47, 48, 49, 50, 51] |
| clcrn | [42, 43, 45] | [44, 46, 47, 48, 49, 50, 51] |

**Note on clcrn:44:** Timed out at 24h wall in first attempt (job 1029536). Config now caps `epochs: 100 → 60`; retry queued.

## Macro metrics — mean ± std across available seeds

| Rank | Model | n | F1 | AUC-ROC | PR-AUC | Precision | Recall |
|------|-------|---|-----|---------|--------|-----------|--------|
| 1 | **CLCRN** | 3 | 0.7096 ± 0.0003 | 0.9777 ± 0.0000 | 0.7780 ± 0.0003 | 0.6912 | 0.7315 |
| 2 | **CLIMODE** | 4 | 0.6389 ± 0.0013 | 0.9596 ± 0.0008 | 0.6872 ± 0.0021 | 0.6140 | 0.6681 |
| 3 | **MTGODE** | 5 | 0.5062 ± 0.0071 | 0.9264 ± 0.0024 | 0.5048 ± 0.0090 | 0.4558 | 0.5719 |
| 4 | **AGCRN** | 5 | 0.3016 ± 0.0143 | 0.8144 ± 0.0107 | 0.2318 ± 0.0166 | 0.2298 | 0.4478 |

## Wilcoxon signed-rank on macro F1 (paired by seed)

At n=10 the theoretical minimum p-value is 0.00195. Current n values are insufficient for that, but the direction (winner) is settled where every seed pair goes the same way.

| Comparison | n shared | W | p | Winner |
|------------|----------|---|---|--------|
| clcrn vs climode | 3 | 0.0 | 0.2500 | clcrn |
| clcrn vs mtgode | 3 | 0.0 | 0.2500 | clcrn |
| clcrn vs agcrn | 3 | 0.0 | 0.2500 | clcrn |
| climode vs mtgode | 4 | 0.0 | 0.1250 | climode |
| climode vs agcrn | 4 | 0.0 | 0.1250 | climode |
| mtgode vs agcrn | 5 | 0.0 | 0.0625 | mtgode |

## Per-variable AUC-ROC (mean across seeds)

| Model | t2m | d2m | tp | ws10 | rh | sp |
|-------|---|---|---|---|---|---|
| clcrn | 0.990 | 0.988 | 0.952 | 0.968 | 0.970 | 0.998 |
| climode | 0.982 | 0.980 | 0.918 | 0.926 | 0.954 | 0.996 |
| mtgode | 0.952 | 0.953 | 0.871 | 0.888 | 0.913 | 0.981 |
| agcrn | 0.816 | 0.894 | 0.762 | 0.716 | 0.772 | 0.928 |

## Per-variable F1 (mean across seeds)

| Model | t2m | d2m | tp | ws10 | rh | sp |
|-------|---|---|---|---|---|---|
| clcrn | 0.800 | 0.756 | 0.535 | 0.635 | 0.634 | 0.898 |
| climode | 0.736 | 0.696 | 0.454 | 0.512 | 0.573 | 0.863 |
| mtgode | 0.578 | 0.535 | 0.374 | 0.402 | 0.445 | 0.703 |
| agcrn | 0.300 | 0.365 | 0.228 | 0.201 | 0.245 | 0.470 |

## Ranking cross-reference to Phase 1

Phase 1 (regression, t2m MAE): `CLCRN (1.17) < ClimODE (1.57) < MTGODE (2.19) < AGCRN (2.85)`  
Phase 2 (classification, macro F1): `CLCRN (0.71) > ClimODE (0.64) > MTGODE (0.51) > AGCRN (0.31)`  

**Same ranking, different task.** Two-tier structure (geometry-aware CLCRN/ClimODE vs generic-graph MTGODE/AGCRN) replicates cleanly.

Variance signature also replicates: CLCRN std=0.0003 vs AGCRN std=0.0143 on macro F1 — Tier 1 models are ~46× more stable seed-to-seed than Tier 2. Same pattern as Phase 1.
