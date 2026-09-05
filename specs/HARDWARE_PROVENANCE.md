# Hardware Provenance — WeatherFLOW Final Campaign

Which hardware ran which cells. Kept as a spec rather than encoded in directory names so `runs/` stays flat and the paper's analysis code doesn't have to care.

## Scope

The 190-task campaign (4 models × 4 targets × 10 seeds regression + 3 models × 10 seeds classification, plus CLCRN cls added later = 30 classification total) ran across three hardware venues under identical code, configs, and preprocessing cache.

## Cell-by-cell venue

| Cell | Seeds | Venue | Wallclock hardware detail |
|---|---|---|---|
| AGCRN regression (t2m, sp, rh, ws) | 42-51 | **RunPod RTX 4090** (Community Cloud) | ~$0.34/hr, 24 GB VRAM, 8 vCPU cgroup ~7.65 cores |
| MTGODE regression (t2m, sp, rh, ws) | 42-51 | **RunPod RTX 4090** (Community Cloud) | same as above |
| STGODE regression (t2m, sp, rh, ws) | 42-48 (partial), 49-51 pending | **UCT HPC L40S** (partition `l40s`) | 5× srvrocgpu[012-016], 4× L40S per node, QOS `l40sugrd` (2 GPU max/user) |
| CLCRN regression (t2m, sp, rh, ws) — full 10 seeds | 42-51 pending | **UCT HPC L40S** | same as STGODE |
| CLCRN regression — seed 42 preview (t2m, sp, rh, ws) | 42 only | **RunPod RTX 4090** (`appropriate_brown_spoonbill`) | same 4090 setup as AGCRN/MTGODE; done pre-HPC to unblock paper writing |
| AGCRN classification | 42-51 | **RunPod RTX 4090** | same 4090 |
| MTGODE classification | 42-51 | **RunPod RTX 4090** | same 4090 |
| CLCRN classification | 42-51 pending | **UCT HPC L40S** | (never ran on pod) |

## Software stack — identical across venues

- Python 3.10
- PyTorch 2.2.0 + CUDA 12.1
- torch-scatter 2.1.2+pt22cu121, torch-geometric 2.8
- numpy 1.26.4
- Full deps: `requirements.txt`
- Preprocessing cache: `preprocessing/cache/saws/` (SHA hash `c8a618bc...`), identical file bytes on both pod and HPC
- Configs: `experiments/configs/*.yaml`, identical

## Within-cell consistency

- **Within each cell, all seeds ran on the same hardware.** Seed statistics (mean ± std, paired tests) are unaffected by hardware choice.
- The one exception is **CLCRN regression** — seed 42 on pod 4090, seeds 43-51 on HPC L40S. This is called out explicitly in the paper's Methods. When reporting CLCRN reg numbers, tables mark seed 42 with a footnote `*` = 4090; unmarked seeds = L40S.
- CLCRN classification is entirely L40S — no hardware mix.

## Reason for the split

- **Pod-first for AGCRN + MTGODE reg + all pod-ran cls** — HPC queue wait times ~24-48h; RunPod Community Cloud 4090 at $0.34/hr was faster + cheaper for small models where memory bandwidth isn't the bottleneck.
- **HPC-first for STGODE reg + CLCRN** — STGODE cost ~6.7h per task (400 min); CLCRN is memory-bandwidth-bound. L40S bandwidth (864 GB/s vs 4090's 1008 GB/s) is comparable for CLCRN, and HPC's free compute allocation makes long tasks cheaper by construction.
- **CLCRN seed-42 preview on pod** — 2026-08-19 addition, driven by paper-write deadline pressure. Reviewer signed off on the split-cell provenance as a Methods footnote.

## Paper Methods footnote (draft)

> All 32-station SAWS training runs used identical software (PyTorch 2.2.0 + CUDA 12.1, numpy 1.26.4, torch-geometric 2.8) and identical preprocessed feature/label caches. AGCRN and MTGODE regression cells, and all AGCRN/MTGODE classification cells, ran on RunPod Community Cloud RTX 4090 GPUs. STGODE regression, CLCRN regression (seeds 43-51), and CLCRN classification cells ran on the University of Cape Town HPC's L40S partition (srvrocgpu[012-016], QOS `l40sugrd`). CLCRN regression seed 42 ran on the same RunPod 4090 setup as AGCRN/MTGODE; this single-seed hardware split is noted in Table X. Within-cell seed statistics are hardware-uniform except where noted.

## Update discipline

Update this doc whenever a new cell moves venue or a new seed lands on non-standard hardware. Keep entries dated in git history rather than in-file to avoid rot.
