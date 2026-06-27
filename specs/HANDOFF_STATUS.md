# WeatherFLOW HPC Status — Handoff Snapshot

**Last updated:** 2026-06-25 08:35 SAST (Thursday morning)

## ClimODE epochs cap lowered to 30 (2026-06-25)

After both ClimODE seeds 42 and 43 timed out at epoch 37/40 under `epochs: 40` (24h SLURM wall hit before test eval), [experiments/configs/climode.yaml](experiments/configs/climode.yaml#L15) was lowered to `epochs: 30`. Justification, references (Goodfellow 2016, Gaibie 2024, Davidson & Moodley 2022, Ayers et al., MLCommons), and the paste-ready paper paragraph are in [specs/RESEARCH_FINDINGS.md](specs/RESEARCH_FINDINGS.md) under the ClimODE section. ClimODE seeds 42 and 43 have been added to the front of the watcher queue and are rerunning. **Phase 2 follow-up**: switching ClimODE to a fixed-step ODE solver could restore the full 40-epoch budget — captured in RESEARCH_FINDINGS.md and status_roadmap memory.


## 🚨 Head-node policy incident (2026-06-24 11:12 SAST)

I ran `ssh hpc "singularity exec ... python patch_vpt.py"` directly. The cache load (~22 GB) consumed all available RAM on the head node. UCT HPC admin (Andrew Lewis) emailed Praise a warning. **Rule going forward (now in memory):** NEVER run `python` / `singularity exec` / any memory-allocating workload directly over `ssh <head>` — always `sbatch` it, even for tiny one-shots. The only safe head-node commands are `squeue`, `sacct`, `sinfo`, `sbatch`, `ls`, `cat <small file>`. The drip-feed watcher is fine (only does `squeue` + occasional `sbatch`).

**Phase:** 1 — production runs on full 999-node ERA5 (2000–2026)

---

## ✅ Done

- ERA5 download complete: 317/317 files in `data/era5/raw/`
- ERA5 rsync to HPC complete: `/scratch/jrvpra001/weatherflow/WeatherFLOW/data/era5/raw/` (317 files)
- Preprocessing cache rebuilt on HPC: `preprocessing/cache/era5/meta.json` shows `n_months=317`, `date_range=2000-01→2026-05`, `n_nodes=999`
- `expver` xarray merge fix applied: [preprocessing/era5_parser.py:79](preprocessing/era5_parser.py#L79) uses `compat="override"`
- `--seed-start` CLI argument added to [cli.py:72-75](cli.py#L72-L75); `train_array.sh` updated to use it
- All SLURM scripts patched with `--pwd /weatherflow` (singularity working dir)
- All SLURM scripts now request `--mem=256G` (32G OOM'd weekend, 128G OOM'd Mon-Tue — see "Memory peak" below)
- Local test suite: 160 passed / 5 skipped

## 🟢 In-flight on HPC (as of 12:36 SAST Mon)

5 initial production jobs queued — all use `train_array.sh` (1 seed per job, --mem=128G, --time=24h):

| Job ID | Model | Seed |
|--------|-------|------|
| 947744_42 | AGCRN | 42 |
| 947745_42 | MTGODE | 42 |
| 947746_43 | MTGODE | 43 |
| 947747_42 | ClimODE | 42 |
| 947748_43 | ClimODE | 43 |

**Watcher now runs on HPC head node** (survives local WSL restarts):
- Script: `/scratch/jrvpra001/weatherflow/hpc_watcher.sh`
- Log: `/scratch/jrvpra001/weatherflow/logs/watcher.log`
- Stdout: `/scratch/jrvpra001/weatherflow/logs/watcher.out`
- Process: `pgrep -f hpc_watcher` on HPC head

Queued for drip-feed:
- **AGCRN seeds 43–51** (9 more)
- **MTGODE seeds 44–51** (8 more)
- **ClimODE seeds 44–51** (8 more)
- **CLCRN seeds 42–51** (10 — newly added)

## 🟢 CLCRN unblocked (no container rebuild needed)

The `torch_scatter` import in [CLCRN/model/manifold/sphere.py:5](models/baselines/CLCRN/model/manifold/sphere.py#L5) was **unused dead code**. Patched with a try/except wrapper. Import smoke test passes inside the existing container. CLCRN seeds 42–51 added to the watcher's queue.

Pre-built PyG wheel approach was tried first but failed with ABI mismatch — NGC base image uses dev PyTorch `2.2.0a0` while PyG wheels are compiled against stable `2.2.0` (different symbol exports). The wheel download + extraction still exists at `/scratch/jrvpra001/weatherflow/pypackages/` if needed later.

Container rebuild attempted locally in tmux but kept dying when WSL shut down. Not needed anymore.

## ✅ First production runs at 999 nodes COMPLETED

As of 2026-06-24 10:55:
- AGCRN seed 42 — completed (job 950031, 3h24m)
- MTGODE seed 42 — completed (job 950033, 10h27m)
- MTGODE seed 43 — completed (job 950034, 5h20m)
- ClimODE seeds 42, 43 — still running (~5h, ~3h)
- CLCRN seed 42 retry — queued (job 951019), tests the device fix in [clcrn_adapter.py:_move_aux_tensors](models/baselines/clcrn_adapter.py)

256G mem held without OOMing. **3 production runs / 40 = 7.5% of Phase 1 done.**

**Watcher caveat:** The previous watcher died around 21:27 SAST Tuesday for unknown reasons after only submitting 1 extra job (climode:43). Queue dropped to 2 jobs for ~13 hours overnight, costing ~30 wasted GPU-slot-hours. New watcher running with explicit "WATCHER STARTED" and "NEXT UP" log markers so failures are obvious. **Issue:** rsync of the watcher script does NOT preserve the executable bit by default — must `chmod +x` after rsync. This is now in [hpc_handoff.md](../../../.claude/projects/-home-praise-Computer-Science-Resources-Weatherflow/memory/hpc_handoff.md).

## 🔴 What did NOT complete over the weekend

All 5 production jobs OOM-killed (32G mem limit too tight). **No test_metrics.json from 999-node runs exists yet.** Only smoke tests at `--max-nodes 10` have completed metrics.

| Job | Model | Elapsed | Result |
|-----|-------|---------|--------|
| 943817_42 | MTGODE seed 42 | 9h58m | OOM (MaxRSS 31.9G) |
| 943835_43 | MTGODE seed 43 | 5h08m | OOM |
| 943836_44 | MTGODE seed 44 | 8h06m | OOM |
| 943837_45 | MTGODE seed 45 | 5h13m | OOM |
| 943894_42 | AGCRN seed 42 | 2h48m | OOM (during test eval — see TODO below) |

## 🔬 Research findings (separate doc)

Per-model metrics, the VPT=0 finding, dome-shaped NRMSE curves, and all empirical observations are tracked in [specs/RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md). Update that doc after each new seed completes.

## 📋 Known issues / future work

- **Test evaluation memory peak ~141 GB at 999 nodes**:
  - `all_preds + all_targets` accumulated in [trainer.py:_evaluate_test](training/trainer.py#L268): ~35 GB
  - `compute_all` in [evaluation/regression.py](evaluation/regression.py#L111) allocates `denormalise(preds)` + `denormalise(targets)` as NEW arrays (preserves originals for `mae_pooled_norm` line 171): +34 GB
  - extreme_mask bool 4D: +4 GB
  - Loaded ERA5 cache + torch tensor copies (3 splits): ~22 GB
  - Stratified MAE + peak intensity intermediates: ~17 GB extra during compute
  - **Total peak ≈ 141 GB**, forcing `--mem=256G`
  - Streaming refactor (TODO marker in [trainer.py:265-268](training/trainer.py#L265-L268)) would drop peak well below 32 GB but requires refactoring both trainer and `compute_all`. Not blocking now that 256G works.
- **No checkpointing between seeds in `train_single.sh`** — currently using `train_array.sh` exclusively (1 seed per job), so this is moot for now. If we ever switch back to `train_single.sh` for fast models, add a per-seed resumability mechanism.
- **QOS limit = 5 jobs/user on l40sugrd** — requires the watcher pattern. If UCT raises this limit, the watcher can be removed.

## Key paths

| What | Where |
|------|-------|
| Code on HPC | `/scratch/jrvpra001/weatherflow/WeatherFLOW/` |
| Container on HPC | `/scratch/jrvpra001/weatherflow/container.sif` |
| Outputs on HPC | `/scratch/jrvpra001/weatherflow/WeatherFLOW/outputs/` |
| SLURM logs | `/scratch/jrvpra001/weatherflow/logs/` |
| HPC watcher script | `/scratch/jrvpra001/weatherflow/hpc_watcher.sh` |
| HPC watcher log | `/scratch/jrvpra001/weatherflow/logs/watcher.log` |

## How to check status quickly

```bash
# 1. HPC queue
ssh jrvpra001@hpc.uct.ac.za "squeue -u jrvpra001"

# 2. Completed runs (have test_metrics.json)
ssh jrvpra001@hpc.uct.ac.za \
  "find /scratch/jrvpra001/weatherflow/WeatherFLOW/outputs/ -name test_metrics.json | sort"

# 3. Watcher progress (on HPC)
ssh jrvpra001@hpc.uct.ac.za "tail /scratch/jrvpra001/weatherflow/logs/watcher.log"

# 4. Watcher process alive on HPC
ssh jrvpra001@hpc.uct.ac.za "pgrep -f hpc_watcher && echo alive || echo dead"
```

If the HPC watcher dies, restart with:
```bash
ssh jrvpra001@hpc.uct.ac.za "nohup /scratch/jrvpra001/weatherflow/hpc_watcher.sh > /scratch/jrvpra001/weatherflow/logs/watcher.out 2>&1 & disown"
```
