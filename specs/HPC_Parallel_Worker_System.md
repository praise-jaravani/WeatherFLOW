# HPC Parallel Worker System — operator's guide

**Purpose:** run many independent GPU tasks (a "campaign") on a shared HPC where you have a strict QOS cap on concurrent GPUs, limited submit slots, and no admin access. Designed for reproducible, resumable, self-perpetuating execution against a fixed manifest of tasks.

**Status:** production-proven — completed the 25-station SAWS regression campaign (120 tasks) in ~24 wall-hours across 4 links, surviving 1 hardware fault and 3 code bugs introduced mid-flight, with zero lost seeds. Will run Phase 2 (classification) and the 60-node regression campaign through the same system.

**Design goals** ([specs/Run_Plan_Spec.md](Run_Plan_Spec.md) has the original design; this doc is the operator's version):
- **C1 (hard)**: no task is ever lost to any failure (walltime, node crash, code bug, network fault).
- **C2**: minimum jobs submitted (~1 chain link per lineage per 24 h, not one per task).
- **C3**: partial results usable within hours, not weeks — drip-feed of completions from round-robin ordering.
- **C4**: skip-if-complete idempotence — safe to re-run any component at any time.

## 1. What it does

A **manifest** of task IDs (e.g. `mtgode_t2m_s42`) is walked in a defined order. **Workers** are GPU jobs that claim tasks off the manifest, train them, checkpoint per-epoch, write a `summary.json`, and move on to the next. Each worker submits its own **successor** at startup — a chain that auto-terminates when the manifest drains. Each worker runs **K concurrent training processes on one GPU** for co-tenancy speedup.

## 2. Architecture

```
                     ┌──────────────────────┐
                     │  runs/manifest.json  │  (built once via scripts/build_manifest.py)
                     │  [tid_1, tid_2, ...]  │
                     └──────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │ (lineage A)            │              (lineage B)
        ▼                        ▼                        ▼
   ┌─────────┐              ┌─────────┐              ┌─────────┐
   │ link N-1 │ ──afterany─▶│  link N  │ ──afterany─▶│ link N+1 │
   │ sbatch  │              │  sbatch  │              │ sbatch   │
   └─────────┘              └─────────┘              └─────────┘
       │  (each link runs 24h wall-clock)
       │
       ▼
   worker.sh (bash wrapper on compute node)
       │
       │  ┌── writes runs/live_jobids.txt (squeue snapshot, refreshed every 5min)
       │  ├── traps SIGUSR1 (20min before walltime end) → forwards to python children
       │  └── submits its own successor via `sbatch --dependency=afterany:$SLURM_JOB_ID`
       │
       ▼
   singularity exec ... python3 -u scripts/run_worker.py  (× K processes, staggered 20s apart)
       │
       │  each process loop:
       │    1. call next_incomplete_task(manifest, claims_dir, runs_dir)
       │       → sweeps dead-jobid claims first (reads runs/live_jobids.txt)
       │       → walks manifest, atomic mkdir-claim on first unclaimed task
       │    2. Trainer.train(stop_check=SIGUSR1_flag)
       │       → per-epoch state.pt checkpoint (atomic write via os.replace)
       │       → per-epoch reseed torch/numpy/python via f(base_seed, epoch)
       │       → divergence guard: 3 consecutive non-finite epochs → FAILED_DIVERGED marker + skip
       │    3. write summary.json + test_metrics.json + per_station.npz
       │    4. GOTO 1
       │
       ▼
   runs/<task_id>/{summary.json, checkpoint.pt, state.pt, per_station.npz, train.log}
```

## 3. Component inventory

### Files created by the system

| Path | Role | Written by |
|---|---|---|
| `runs/manifest.json` | Ordered list of task IDs. Round-robin by seed, slow models first per round. | `scripts/build_manifest.py` (once, at campaign start) |
| `runs/claims/<task_id>/meta.json` | Atomic claim marker. `mkdir` is the lock; `meta.json` records jobid/pid/host/ts. | `orchestration.claim.try_claim` (per proc, per task) |
| `runs/live_jobids.txt` | Snapshot of `squeue -h -u $USER -o '%A'` refreshed every 5 min. | `worker.sh` background loop |
| `runs/<task_id>/state.pt` | Full training state (model, opt, scheduler, RNG, epoch). Atomic write. | `Trainer._save_full_state` per epoch |
| `runs/<task_id>/checkpoint.pt` | Best-val model weights only. | `Trainer._save_checkpoint` per epoch |
| `runs/<task_id>/summary.json` | Completion marker. Contains train/val loss history + test_metrics. | `run_worker._prepare_and_run_task` |
| `runs/<task_id>/per_station.npz` | Per-station regression metrics for geographic stratification. | `Trainer._save_per_station` |
| `runs/<task_id>/FAILED_DIVERGED` | Marker: 3 consecutive non-finite epochs. Task is NOT re-attempted. | `Trainer.train` on divergence |
| `runs/<task_id>/train.log` | Per-epoch train/val/grad_norm; `# RESUMED from epoch N` marker on resume. | `Trainer` |
| `logs/wf_worker-<jobid>.out` | SLURM stdout of the wrapper. Contains `DISPATCH` + `SNAPSHOT` markers. | `worker.sh` |
| `logs/wf_worker-<jobid>-p<i>.out` | Per-process stdout inside the container. | `singularity exec` redirect |

### Scripts and code

| Path | Role |
|---|---|
| `slurm/worker.sh` | SBATCH wrapper. Reads `run_worker.py` fresh from disk at each link. Sets `WF_LIVE_JOBIDS_FILE`, forwards SIGUSR1, submits successor. |
| `scripts/run_worker.py` | Python driver. Loop: claim → train → summary → repeat. |
| `scripts/build_manifest.py` | Emits `runs/manifest.json` in §2 order (round-robin by seed, slow first per round). |
| `scripts/manifest_incomplete.py` | Exit code 0 iff any task lacks `summary.json` AND has no `FAILED_DIVERGED`. Called by `worker.sh` before submitting successor. |
| `scripts/manual_sweep.sh` | Bash-only head-node script: release claims whose jobid isn't in `squeue -u $USER`. Emergency use — normal ops shouldn't need it. |
| `scripts/chain_status.sh` | Head-node dashboard: prints per-model completion counts, EMA, active jobs. |
| `scripts/compute_baselines.py` | Persistence + climatology baselines. Once per (dataset, network) — seed-independent. |
| `scripts/analyze_saws25.py` | Aggregate `runs/*/summary.json` → `phase1_regression/<model>_summary.json` + plots. Idempotent. |
| `orchestration/manifest.py` | `build_manifest`, `is_complete`, `is_failed_diverged`. |
| `orchestration/claim.py` | `try_claim` (atomic mkdir), `stale_claim_sweep`, `next_incomplete_task`, `live_squeue_jobids`. |
| `training/trainer.py` | `Trainer.train` — the per-task loop with per-epoch checkpoint, resume, SIGUSR1 stop, divergence guard. |
| `results/sync_from_hpc.sh` | Local: rsync HPC `runs/` → `results/runs/`, then re-run `analyze_saws25.py`. Idempotent. |

## 4. Chain mechanism — how links propagate

Each `worker.sh` invocation (a "link"):

1. **At startup, before any work**:
   - Probes `squeue` is in PATH — fails loud if not (environment-probe rail).
   - Writes `runs/live_jobids.txt` atomically (`> tmp.$SLURM_JOB_ID` then `mv`).
   - Starts a background bash loop that refreshes `live_jobids.txt` every 5 min (per-lineage `.tmp.$SLURM_JOB_ID` avoids mv races).
   - Emits the `SNAPSHOT n=<count> jobids` marker as proof.
   - Emits the `DISPATCH LINK_INDEX=<N> K_PROCS=<K> SLURM_JOB_ID=<id> host=<h>` marker.
2. **Submits successor** (unless capped or manifest drained):
   ```bash
   sbatch --dependency=afterany:$SLURM_JOB_ID \
          --export=ALL,LINK_INDEX=$((LINK+1)),K_PROCS=3,... \
          worker.sh
   ```
   `afterany` means the successor becomes eligible when this link EXITS FOR ANY REASON (walltime, crash, cancel). This is the crash-recovery mechanism.
3. **Launches K python processes** in parallel, staggered ~20s apart (avoids simultaneous cache-load spikes). Each writes to `logs/wf_worker-<jobid>-p<i>.out`.
4. **Signal-forwards SIGUSR1**: `#SBATCH --signal=B:USR1@1200` delivers a signal 20 min before walltime. The bash trap locates each child's python PID via `pgrep -P $starter_pid -f python3` and sends `kill -USR1 $py_pid` (apptainer's `starter` doesn't forward signals — probed 2026-08-05).
5. **Re-wait loop**: after the trap fires, bash's `wait` returns immediately; the wrapper loops through remaining CHILD_PIDS, reaping each as it exits.
6. **On exit**: successor's dependency clears → it becomes eligible → competes for a GPU slot.

Chain terminates when either:
- `MAX_LINKS=10` per lineage (hard runaway cap).
- `manifest_incomplete.py` returns exit code 1 (no `summary.json`-lacking task left).

## 5. Concurrency model

Physical constraints (UCT HPC, l40s partition, `l40sugrd` QOS):
- **MaxTRESPU=`gres/gpu:l40s=2`**: max 2 concurrent GPU allocations across all our jobs.
- **MaxSubmitPU=5**: max 5 jobs submitted (running + pending).
- **Assoc MaxWall=24h**: hard cap per job.

Design uses these as follows:
- **2 lineages** initially submitted → each self-perpetuates via afterany. Steady state: 2 R + 2 dependent-PD + 1 spare = 4/5 MaxSubmit.
- **K=3 processes per GPU**: co-tenant training. L40S has 48GB VRAM; our models (~2-3 GB each at N=25) fit ~15× over. Compute utilisation 99% at K=3 (near-saturated). Per-worker throughput multiplier m₃ ≈ 1.42 vs K=1.
- **Effective concurrent tasks**: 2 GPUs × 3 procs = **6 simultaneous trainings**.
- **Total campaign throughput** (25-node measured): ~120 tasks / ~24 wall-h from campaign start = ~5 tasks/h across the whole system.

Per-process resource in `worker.sh` SBATCH block:
- `--cpus-per-task=12` (K=3 × OMP_NUM_THREADS=2 = 6 kernel threads + I/O headroom in remaining 6)
- `--mem=32G` (fits every node; the 44G-free srvrocgpu015 stays backfill-eligible)
- `--gres=gpu:l40s:1` (one GPU per sbatch)
- `--time=24:00:00` (assoc MaxWall)
- `--exclude=srvrocgpu014` (defensive; kept after 2026-08-05 ECC event)

## 6. Failure modes and how each is handled

| Event | Handling | Result |
|---|---|---|
| Walltime hit mid-epoch | SIGUSR1 (20 min early) → trainer checkpoints + exits 0 → afterany successor resumes from state.pt | ≤1 epoch repeated |
| Node crash / SLURM cancel | afterany successor still triggers (not afterok); state.pt from last checkpoint = last completed epoch | ≤1 epoch repeated |
| Python process crash (e.g. sys.modules bug) | Sibling procs unaffected; wrapper reaps zombie; task's claim persists but auto-releases when the SLURM job dies + next sweep runs | Task lost until next link |
| GPU ECC / uncorrectable memory error | Whole worker dies → afterany successor spawns → probably lands on different node | ≤ 1 link lost |
| Training divergence (3 consecutive non-finite epochs) | `FAILED_DIVERGED` marker written; task never re-attempted; surfaced by chain_status.sh | Task marked, campaign continues |
| Two workers race for same task | `mkdir claims/<task_id>/` is atomic; exactly one succeeds | No double-claims |
| Worker dies without cleaning claim | Sweep at next worker startup reads `live_jobids.txt`, releases claims whose jobid isn't live | Auto-recovery |
| `squeue` fails (network / not in PATH) | Reader treats file-missing or file-stale >15 min as `_SQUEUE_UNKNOWN` → sweep does nothing → conservative | No mass wrong-release |
| Chain drains manifest | `manifest_incomplete.py` returns 1 → no successor submitted → chain terminates | Clean exit |

## 7. The environment-probe rail

Standing rule added 2026-08-06 after three environment misses cost real time:

**Any external command or capability the code depends on MUST be probed in its actual execution context before being relied on, and MUST fail loud, not silent.**

Historical misses:
- **apptainer SIGUSR1 forwarding** (2026-08-05): assumed apptainer forwards signals to the contained process. It does not. Fix: `pgrep -P $starter_pid -f python3` + signal the python directly. Documented in worker.sh.
- **association-level MaxWall** (2026-08-05): assumed QOS + partition were authoritative. Association has its own MaxWall=24h. Learned when a `--time=36:00:00` sbatch was rejected with `AssocMaxWallDurationPerJobLimit`. Fixed to 24h.
- **container `squeue`** (2026-08-06): assumed `subprocess.check_output(["squeue", ...])` would work inside the singularity container. `squeue` is not in the container's PATH. Sweep silently returned `_SQUEUE_UNKNOWN` forever. Fixed with wrapper-side snapshot to `runs/live_jobids.txt`.

Enforcement: `worker.sh` explicitly probes `command -v squeue` at startup, exits fatal if missing. The `SNAPSHOT n=<count>` line in the log is the greppable proof the probe passed and the snapshot has real data.

## 8. Standing rails (unchanged since kickoff)

- **sbatch only** on HPC. No head-node python execution beyond trivial file operations. Head-node bash script `manual_sweep.sh` is allowed (bash-only, no python).
- **Sanctioned artefact inspection**: `bash results/sync_from_hpc.sh` then `torch.load` **locally** on the copied files. Remote `singularity exec ... python3 -c 'torch.load(...)'` is banned.
- **Monitor policy**: exactly ONE monitor at any time, running LOCALLY, 5-min minimum tick, `ssh + squeue + ls/grep -l` only. Ten-second-tick patterns banned.
- **If the environment can't satisfy a requirement, stop and report the gap — never substitute and reframe.**

## 9. Operator playbook

### Launching a campaign (fresh)

```bash
# On local, in repo root
python scripts/build_manifest.py \
    --out /path/to/runs/manifest.json \
    --seeds 42 43 44 45 46 47 48 49 50 51 \
    --models clcrn mtgode agcrn \
    --targets t2m sp rh ws

# Verify manifest length + first/last entries
head runs/manifest.json && tail runs/manifest.json

# Deploy repo to HPC (--checksum for reliability)
rsync -avz --checksum ./ hpc:/scratch/jrvpra001/weatherflow/WeatherFLOW/

# From HPC head node — build baselines for this network (once, seed-independent)
ssh hpc "cd /scratch/jrvpra001/weatherflow/WeatherFLOW && \
    singularity exec --bind /scratch/jrvpra001:/scratch/jrvpra001 \
    --env PYTHONPATH=/scratch/jrvpra001/pypkgs container.sif \
    python3 scripts/compute_baselines.py --dataset saws --skip-classification"

# Submit 2 initial workers (per MaxTRESPU=2 GPU cap)
ssh hpc "cd /scratch/jrvpra001/weatherflow/WeatherFLOW && \
    for i in 1 2; do sbatch slurm/worker.sh; done"

# Verify submission
ssh hpc "squeue -u \$USER -o '%.10i %.15j %.2t %.10M %.10L %R'"
```

### Sync + aggregate while campaign runs (idempotent, run anytime)

```bash
bash results/sync_from_hpc.sh
# → rsyncs /scratch/.../runs/ → results/runs/
# → auto-runs scripts/analyze_saws25.py
# → refreshes results/phase1_regression/*.json + results/plots/{SUMMARY.md, mae_bar.png, val_convergence.png}
```

### Status check (head-node, safe)

```bash
ssh hpc "cd /scratch/jrvpra001/weatherflow/WeatherFLOW && bash scripts/chain_status.sh"
# OR one-liner queue + completion counts:
ssh hpc "squeue -u \$USER; ls /scratch/jrvpra001/weatherflow/WeatherFLOW/runs/*/summary.json 2>/dev/null | wc -l"
```

### Emergency: manual claim release (rare)

```bash
ssh hpc "bash /scratch/jrvpra001/weatherflow/WeatherFLOW/scripts/manual_sweep.sh"
# Releases claim dirs whose jobid isn't in current `squeue`. Safe to run any time.
```

### Shelving results before starting a new campaign

```bash
mv results/ results_<descriptor>/    # e.g. results_saws25node_2013_2020
mkdir -p results/{runs,phase1_regression,baselines,analysis,plots,archive}
cp results_<descriptor>/sync_from_hpc.sh results/    # or edit for the new campaign
# Then write a fresh results/README.md describing the new campaign's scope
```

### Cancelling and re-submitting (with care)

- **Do not cancel successors** — they carry code fixes forward via afterany. Cancel only the wounded PARENTS.
- **Do not scancel and immediately submit fresh** — the afterany successor is already pre-positioned. `scancel <parent>` promotes it to eligible.
- **If you MUST resubmit fresh**: cancel ALL your jobs first, then `sbatch worker.sh` × 2. New chain starts clean.

## 10. Post-mortem of the 25-node campaign

**Ran 2026-08-04 to 2026-08-07, ~72 wall-hours from submission (~24 hours from first R). Delivered 120/120 completions with 0 lost seeds.**

Failures survived without losing a single seed:
1. **CUDA ECC** on srvrocgpu014 (2 initial jobs died in 8s) — recovered via afterany + `--exclude=srvrocgpu014` in worker.sh
2. **`SLURM_TIMELIMIT` unbound** (my bug, added observability echo) — 2 more workers died in 10s — fix + resubmit clean
3. **Association MaxWall** miss — a 36h `--time` was rejected — fix to 24h and resubmit
4. **`sys.modules['model']` collision** (my bug, missed pattern) — 4 procs died on MTGODE→AGCRN transition — `_purge_shared_model_module()` fix, deployed for link 2+
5. **`squeue` missing inside container** (my bug, no environment probe) — sweep silently spared all claims → 82 stale claims accumulated — manual head-node sweep + WF_LIVE_JOBIDS_FILE fix
6. **1 CLCRN stuck-at-init** (rh_s45) — not a code bug, seed sensitivity at lr=0.05 — reported honestly in results

Lessons that became design rules:
- Environment-probe rail (§7)
- Sanctioned artefact-inspection method (§8)
- Monitor policy (§8)
- Chain design proved out: 4 links total, self-perpetuated, no manual re-submits needed after the initial cancellations

## 11. What this system does NOT do

- **Doesn't handle HPO or hyperparameter sweeps.** Each task is a fixed config. HPO would need a separate manifest generator.
- **Doesn't cross-validate.** Splits are fixed by preprocessing.
- **Doesn't manage the data cache.** Preprocessing must be done separately (once) before the campaign; workers assume `preprocessing/cache/saws/` is populated.
- **Doesn't clean up completed-task claim dirs.** Cosmetic residue in `runs/claims/` (skipped by `is_complete` check anyway). Cleared implicitly when the owning SLURM job dies + a sweep runs successfully.
- **Doesn't handle Phase 2 classification differently** — same infrastructure, but the manifest tasks would run `--task classification` and the aggregation would need updates to handle F1/AUC instead of MAE. Straightforward extension for the next campaign.

## 12. Adapting for the 32-node campaign (executed 2026-08-08)

**Status update (2026-08-08):** originally planned as a 60-node campaign against the
new SAWS-60-node data delivery. Independent audit (Gen + Zoleka + Claude Code from
`data/saws/audit/audit_60node.py`, all three independently arrived at the same number)
established that only **32 stations** pass ≤10% overall missingness across all 6
variables for 2016-2025 — see [`memory/60node_campaign_planned.md`](../../.claude/…) +
[`data/saws/AUDIT.md`](../data/saws/AUDIT.md). The campaign is therefore N=32, not 60,
and all §12 items below reflect that.

Sequence (as executed 2026-08-08):
1. New SAWS-60-node data received; audit computed the 32-station whitelist.
2. Wrote a new block-format parser at `preprocessing/saws_parser.py` (rain-blank=0
   convention per legend, differs from old Lekuba long-form). Old parser preserved
   at `preprocessing/saws_parser_25node_lekuba.py`.
3. Rebuilt `preprocessing/cache/saws/` with the 32-station arrays (splits
   52603/17534/17535 = 87672 h).
4. Recomputed baselines: `python scripts/compute_baselines.py --dataset saws --skip-classification`.
5. Shelved the 25-node results: `mv results/ results_saws_25node_phase1/`.
6. Fresh `results/{runs,phase1_regression,baselines,analysis,plots,archive}` + updated `README.md`.
7. Rebuilt manifest: `python scripts/build_manifest.py --out runs/manifest.json`
   (same 120 tasks: 3 models × 4 targets × 10 seeds).
8. Deploy to HPC (all system code is network-size-agnostic; no changes needed).
9. **MTGODE `graph_constructor.k` — set to 13 at N=32** (density-target 13/32 = 41%
   preserves the ~40% neighbours rule we documented for N=25 → k=10). The earlier
   §12 draft anticipated N=60 and speculated k=20 would become paper-defensible again;
   at the actual N=32, k=13 is the correct value and is locked in
   [`experiments/configs/mtgode.yaml`](../experiments/configs/mtgode.yaml).
10. **MTGODE audit fixes folded in same commit** (per 2026-08-08 external audit,
    Findings 0/2/5): `ode.adjoint: false` (was `true` — graph learner had frozen
    grads under adjoint mode), `ode.cgp_alpha: 2.0` (was hardcoded 1.0 — paper Eq.6
    requires 2.0), `graph_constructor.node_dim: 40` (was conflated with hidden_dim=64
    — paper §5.2 = 40). See `specs/HYPERPARAMETERS.md` for the full disclosure.
11. **Uniform MAE loss** across MTGODE/AGCRN/CLCRN (was MSE) — matches AGCRN's
    `L1Loss` default, CLCRN's `masked_mae_loss`, MTGODE §4.4. Diverges from
    Davidson & Moodley (2022) — documented departure.
12. Submit 2 initial workers. Watch first link converge on healthy K=3, mixed-model
    completions within a few hours.

**HPO status (2026-08-08):** deliberately parked. The reviewer's calendar analysis
was decisive: 100 trials × ~1h on 6 concurrent slots = 17h optimistic / 30–50h with
queue friction, in front of a Wed draft with N=32 production not yet started. Any
future HPO is post-draft; the design owed is **Option B reframed as Gaibie's protocol**
(pre-sampled random search via a manifest generator, uses this system's existing
worker/claim/sweep machinery untouched, 5 knobs: lr/batch/dropout/hidden_dim/k,
input_len=12 fixed because receptive-field `R = nfe₁·6+1` truncates 48/72 silently at
the paper's 1.0/0.25 solver, cgp_alpha/node_dim/tanhalpha/ODE times pinned at paper
values per audit). Option A (Optuna distributed with SQLite on NFS) is a documented
silent-corruption path and is rejected unless replaced with JournalFileStorage +
constant_liar + stale-RUNNING-trial sweep at link boundaries.

Same drill for **Phase 2 (classification)**: change `--task classification` in run_worker's config load, update `analyze_saws25.py` to handle F1/AUC/PR outputs, submit. Same chain design.
