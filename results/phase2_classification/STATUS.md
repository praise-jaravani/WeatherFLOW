# Phase 2 — Classification Results (Interim)
_Snapshot generated 2026-07-26 10:34 SAST_

## Task

Per-variable extreme-event classification. Labels = 95th-percentile per (station, variable) computed on training split only.

## Completion status

**21 / 40 seeds landed** (target: 10 seeds × 4 models).

| Model | Done | Missing |
|-------|------|---------|
| clcrn | [42, 43, 45, 46] | [44, 47, 48, 49, 50, 51] |
| climode | [42, 43, 44, 45, 46] | [47, 48, 49, 50, 51] |
| mtgode | [42, 43, 44, 45, 46, 47] | [48, 49, 50, 51] |
| agcrn | [42, 43, 44, 45, 46, 47] | [48, 49, 50, 51] |

## Macro metrics — mean ± std across available seeds

| Rank | Model | n | F1 | AUC-ROC | PR-AUC |
|------|-------|---|-----|---------|--------|
| 1 | **CLCRN** | 4 | 0.7112 ± 0.0028 | 0.9780 ± 0.0005 | 0.7798 ± 0.0032 |
| 2 | **CLIMODE** | 5 | 0.6387 ± 0.0013 | 0.9596 ± 0.0007 | 0.6870 ± 0.0019 |
| 3 | **MTGODE** | 6 | 0.5105 ± 0.0116 | 0.9279 ± 0.0040 | 0.5113 ± 0.0167 |
| 4 | **AGCRN** | 6 | 0.2992 ± 0.0141 | 0.8126 ± 0.0106 | 0.2291 ± 0.0163 |

## Wilcoxon signed-rank on macro F1 (paired by seed)

| Comparison | n shared | W | p | Winner |
|------------|----------|---|---|--------|
| clcrn vs climode | 4 | 0.0 | 0.1250 | clcrn |
| clcrn vs mtgode | 4 | 0.0 | 0.1250 | clcrn |
| clcrn vs agcrn | 4 | 0.0 | 0.1250 | clcrn |
| climode vs mtgode | 5 | 0.0 | 0.0625 | climode |
| climode vs agcrn | 5 | 0.0 | 0.0625 | climode |
| mtgode vs agcrn | 6 | 0.0 | 0.0312 | mtgode |

## Per-variable AUC-ROC (mean across seeds)

| Model | t2m | d2m | tp | ws10 | rh | sp |
|-------|---|---|---|---|---|---|
| clcrn | 0.990 | 0.988 | 0.952 | 0.969 | 0.970 | 0.998 |
| climode | 0.982 | 0.980 | 0.918 | 0.926 | 0.954 | 0.996 |
| mtgode | 0.953 | 0.954 | 0.872 | 0.891 | 0.915 | 0.982 |
| agcrn | 0.813 | 0.892 | 0.760 | 0.713 | 0.769 | 0.928 |

## Per-variable F1 (mean across seeds)

| Model | t2m | d2m | tp | ws10 | rh | sp |
|-------|---|---|---|---|---|---|
| clcrn | 0.802 | 0.757 | 0.536 | 0.638 | 0.636 | 0.898 |
| climode | 0.736 | 0.695 | 0.454 | 0.511 | 0.572 | 0.863 |
| mtgode | 0.582 | 0.540 | 0.377 | 0.408 | 0.451 | 0.705 |
| agcrn | 0.297 | 0.362 | 0.226 | 0.199 | 0.243 | 0.468 |

## Cross-reference to Phase 1

Phase 1 (t2m MAE): `CLCRN (1.17) < ClimODE (1.57) < MTGODE (2.19) < AGCRN (2.85)`  
Phase 2 (macro F1): `CLCRN > ClimODE > MTGODE > AGCRN` — same ranking.
