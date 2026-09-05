# SAWS Production Matrix — Job Submission & Run Plan Specification

**Project:** WeatherFLOW (CSC4019Z) · Praise Jaravani
**Scope:** 120 runs = 3 models × 4 targets {t2m, sp, rh, ws} × 10 seeds, univariate regime, N=25, input 12 / horizon 48
**Doc status:** APPROVED (D1–D5 confirmed 2026-08-04); gates in §9 block submission
**Executing agent:** Claude Code
**Last updated:** 2026-08-05 — §3 restructured for measured QOS reality (Correction 2: MaxSubmitPU=5 hard cap → 2 lineages, not 4); §7 corrected finish projection; §13.1 added (HPC-audit-derived facts)

---

## Corrections landed 2026-08-05 (from a QOS + partition audit against a live cluster snapshot)

The initial plan (2026-08-04) rested on three assumptions that the HPC audit falsified. Recorded verbatim so the design record stays honest:

- **CPU cap read wrong.** `MaxTRESPU cpu=32` is a *concurrent* cap, not a submit cap. With the GPU cap holding 2 workers running, we're at 2×8=16 of 32 (or 2×12=24 of 32 under the K=3 config). Not binding — successors get 12 CPUs each safely.
- **MaxSubmitPU=5 hard-blocks the 4-lineage design.** Once a lineage's first successor is queued, the fifth submit slot is exhausted; the second-lineage successor is silently rejected, killing the chain on that lineage. **Two of the initial four workers were cancelled on 2026-08-05** (1181542, 1181543) to run 2 lineages instead. Steady state: 2 running + 2 dependent successors = 4 of 5 submit slots, 1 spare.
- **MaxWall is unset AT QOS AND PARTITION** — but IS set at the **association** level. `sacctmgr show assoc user=jrvpra001` shows `l40sugrd MaxWall=1-00:00:00 = 24h`. The 24 h ceiling was correct after all; only my initial audit path (QOS + partition) missed it. Confirmed empirically 2026-08-05: frozen 1181540/1181541's spooled successor line (which references the on-disk worker.sh) was rejected with `AssocMaxWallDurationPerJobLimit` when I briefly set worker.sh `--time=36:00:00`. Corrected to `--time=24:00:00`.

- **Hardware fault event 2026-08-05 14:07-14:08 SAST**: both frozen jobs 1181540/1181541 landed on **srvrocgpu014** and died in 6-8 seconds with `RuntimeError: CUDA error: uncorrectable ECC error encountered` during the `adj.to(device)` call (before training even started). Neither wrote any state.pt. Chain broken (successor submission was already rejected in the same window by the walltime error above). Recovery: `worker.sh` now includes `--exclude=srvrocgpu014` defensively; can be removed after the node is confirmed healthy. Fresh submissions 1182591/1182592 (2026-08-05 ~14:XX SAST) are the actual campaign start — no frozen K=1 first-link cost applies since the failed frozen jobs produced zero completions, so K=3 runs from link 1 for both lineages.

- **`SLURM_TIMELIMIT` unbound bug 2026-08-05 evening**: my worker.sh added an observability echo `echo "=== walltime=$SLURM_TIMELIMIT ..."`. `$SLURM_TIMELIMIT` is not a real SLURM env var; `set -u` at wrapper top made the reference fatal. Both fresh workers 1182591/1182592 dispatched, ran the DISPATCH marker echo, then died in <10s on this line before reaching the successor-submission block or the singularity exec. Chain broken again. Fix: replaced with literal `24h(sbatch-fixed)` string; local dry-run of the startup block passes; only `$SLURM_JOB_ID` (guaranteed by SLURM) remains as env dependency.

- **`sys.modules['model']` collision bug 2026-08-06 ~10:03 SAST (mine — preventable)**: AGCRN, CLCRN, MTGODE each vendor an upstream repo under a bare `model/` package name and rely on `sys.path` injection to import. Within one Python process, once one adapter loads its `model` package into `sys.modules`, later adapters trying `from model.XXX import ...` hit the WRONG package. `scripts/probe_adjacency.py` had already solved this in a prior session — the pattern is a `_purge_shared_model_module()` helper that deletes `sys.modules['model']` + `model.*` and restores a `sys.path` snapshot. When I wrote `scripts/run_worker.py` I did not carry the pattern across. In production, 4 of 6 procs on link 1 (jobids 1184527/1184528) died at the MTGODE→AGCRN cross-family transition with `ModuleNotFoundError: No module named 'model.AGCRN'; 'model' is not a package`. Process-death, not task-failure — no FAILED_DIVERGED markers written, claims remain held by the still-live parent SLURM job. Fix: applied the `_purge_shared_model_module()` pattern from probe_adjacency.py to `run_worker.py`, called before every `adapter.build_model()`. Deployed and checksum-verified on HPC. Only affects link-1 procs (which had already loaded the old file into memory); successors read the fresh file at dispatch and are clean.

- **Container-`squeue` bug + fix design 2026-08-06 SAST**: `orchestration.claim.live_squeue_jobids()` originally invoked `squeue` via subprocess. `squeue` is **not in PATH inside the singularity container** (Slurm client tools are host-side, not baked into the pytorch container image). Every sweep call inside the container therefore raised `FileNotFoundError` → returned the `_SQUEUE_UNKNOWN` sentinel → sweep spared ALL claims. **The automatic stale-claim sweep never once ran in production.** Silently. The failure direction was conservative-by-design (spare rather than release), so nothing was corrupted, but stale claims from crashed procs / cancelled jobs accumulated with no auto-recovery. Bug surfaced when 82 dead-jobid claims (72 parked seeds-43+ AGCRN+MTGODE + 10 orphaned link-1 seed-42 claims) blocked link-2 procs from picking up `clcrn_rh_s42` (state.pt ep=22) and `clcrn_ws_s42` (state.pt ep=20), plus all seed-42 AGCRN → the AGCRN bug-fix verdict couldn't fire. Manual head-node sweep released them at 14:03 SAST. **Fix design**: `worker.sh` (running on the compute node where `squeue` IS available) writes squeue output atomically (per-lineage `tmp.$SLURM_JOB_ID` file + `mv`) to `runs/live_jobids.txt`, refreshed every 5 min in a background process; exports path via `WF_LIVE_JOBIDS_FILE` env var to the container; `live_squeue_jobids()` reads that file (falling back to direct `squeue` for local dev). **Four hardenings**: (1) atomic writes (temp + mv) both at startup and refresh; (2) staleness guard — reader treats file mtime >15 min as `_SQUEUE_UNKNOWN` (dead refresher must degrade to safe sentinel, not to trust); (3) missing/empty file → sentinel; (4) fail-loud env-probe — worker.sh verifies `squeue` in PATH at wrapper startup and exits with error if absent, plus logs a greppable `SNAPSHOT n=<count> jobids` line as proof of success. All four verified via `pytest tests/test_orchestration.py` (18/18) + a targeted staleness-guard unit test. Deployed to HPC; takes effect for link 4+ (links 2/3 spooled with the old wrapper and cannot pick up the fix).

- **Environment-probe rail added 2026-08-06**: any external command or capability the code depends on (`squeue`, signal delivery via apptainer, resource limits via association MaxWall) MUST be probed in its actual execution context before being relied on, and MUST fail loud, not silent. Three prior misses would have been caught by this rail: apptainer's SIGUSR1 non-forwarding (probed retroactively), association MaxWall=24h (probed retroactively), container-`squeue` absence (probed retroactively). The `SNAPSHOT n=<count>` echo in worker.sh is the model — probe + proof-in-log. This rail is now a standing rule alongside the sbatch-only and stop-and-report rails in §13.

- **Manifest / walk mechanism verified 2026-08-06** (unblocked from item open since first MTGODE-first completions):
  - `runs/manifest.json` emits round-robin-by-seed with slow-models-first per round: `[clcrn_t2m_s42, clcrn_sp_s42, clcrn_rh_s42, clcrn_ws_s42, mtgode_t2m_s42, mtgode_sp_s42, mtgode_rh_s42, mtgode_ws_s42, agcrn_t2m_s42, agcrn_sp_s42, agcrn_rh_s42, agcrn_ws_s42, clcrn_t2m_s43, ...]` — matches spec §2.
  - `next_incomplete_task` does **fresh scan from top on every call** — not a per-proc cursor. Released tasks are reachable by any subsequent proc-cycle.
  - Link 1's "MTGODE-first completion" pattern was NOT a walk-order bug — it was a consequence of the CG-race stale claims from failed 1181540/1181541 blocking the first 2 CLCRN slots at link-1 startup, combined with sweep-in-container silently failing to release them. First 6 procs skipped past 2 blocked + 2 just-claimed CLCRN → landed on MTGODE. Perfectly correct fresh-scan behaviour given the state.
  - Link 2's anomalous `clcrn_t2m_s44` claim by a cycling proc at 13:51 was the same pattern one link later: at that moment, 88 tasks were still claim-locked by dead procs (my 14:03 manual release hadn't yet happened) → walk landed on the first truly unclaimed → `clcrn_t2m_s44`.

- **Claim-residue semantics** (confirmed 2026-08-06): `run_worker.py` does not release claims on successful task completion. The `summary.json` marker file is what causes `is_complete()` to return True on future walks, so the completed task is skipped regardless of whether its claim dir persists. Residue is cosmetic (visible in `ls claims/`) but has no functional effect. Cleaned only when the claim's SLURM job dies AND sweep runs successfully.

- **Sanctioned state.pt inspection method** (2026-08-06 codification): `bash results/sync_from_hpc.sh` first (rsync into `results/runs/`), then `torch.load` **locally** on the copied files. Remote `singularity exec ... python3 -c 'torch.load...'` is banned as of now — that pattern was the same shape as the head-node execution vetoed in Step 3 of the standing rails, however innocent each individual command. File `mtime`s via `ls` on the head node remain fine.

- **Monitor policy** (2026-08-06 codification): exactly ONE monitor at any time, running LOCALLY (on Praise's laptop, not HPC head node), 5-minute minimum tick interval, each tick limited to `ssh + squeue + ls/grep -l` on log files. Ten-second-tick and shorter poll patterns are banned going forward (the promotion-waiter's `until` loop was a specific mistake). Historical: 11 monitor instances existed across this campaign (v1 through v3 chain of re-arms + waiters); all now dead by SDK timeout or explicit `TaskStop`. Any future monitor must serve an operational duty (detect state change → alert or take specific pre-agreed action), not observational-noise.

- **Kill-to-promote decision 2026-08-06 ~11:15 SAST**: the sys.modules bug left link 1 running at 2/6 concurrency for the remaining 22h of its walltime. Naïve arithmetic ("a third of a GPU in hand beats none") missed the critical structural fact that successors 1184529/1184530 are `afterany`-dependency-held on their crippled parents — every hour link 1 limped, the fixed 6-process replacement was gated from starting its queue-fight. Correct arithmetic: keeping the wounded link to walltime finishes the campaign **later** than killing it now, by roughly 0.6 × (remaining walltime) ≈ 13 wall-hours at time of decision. `scancel 1184527 1184528` executed (successors 1184529/1184530 preserved — the `afterany` chain promotes them to eligible on parent termination without any new submit). Gate before scancel: sys.modules fix verified present in deployed `run_worker.py` (md5 `0213d251081f258bab1a97f366794e26` matched between HPC and local); the batch script is spooled but `run_worker.py` is read fresh from disk at dispatch, so the successors will execute the fixed code. Forfeited: the 2 surviving CLCRN procs' remaining CLCRN progress (~10-15 completions) and early first-CLCRN signal — under kill, first CLCRN completions land after link-2 queue wait instead of within hours. The lr=0.05 divergence question therefore gets answered by link 2 itself; divergence guard marks-and-moves autonomously; the ≥2-diverged → flip-to-0.01 protocol is pre-agreed per D5. Both mid-flight CLCRN tasks (clcrn_rh_s42 ep=22 best_val=0.1865 patience=12; clcrn_ws_s42 ep=20 best_val=0.4078 patience=14) resume from their per-epoch state.pt when link 2 picks them up — C1 doing its job. Revised completion projection: queue-dependent; provisional 24–48h from link-2 dispatch at K=3 co-tenant, still comfortably within the Aug 12-13 draft window.

---

## 0. Objective and constraints

Complete all 120 training runs on the UCT HPC (l40s, 24 h wall-clock cap, QOS ~5 concurrent jobs, currently busy queue) such that:

- **C1 (hard):** no seed is ever lost to a wall-clock timeout.
- **C2:** far fewer than 120 job submissions; queue-friendly on a busy cluster.
- **C3:** results usable for the draft in days, not weeks — with partial results usable early.
- **C4:** standing rails hold — sbatch only, never head-node execution; skip-if-complete resumability.

**Design answer to C1 — stated plainly:** timeouts are not prevented; they are made *harmless*. Every run checkpoints at every epoch boundary; every job link has an `afterany` successor that resumes from the last completed epoch. A timeout therefore costs at most one repeated epoch, never a seed. The guarantee is structural, not a bet on runtime estimates.

---

## 1. Core mechanism (three parts)

### 1.1 Per-epoch checkpoint-and-resume (the load-bearing piece)

The trainer saves, **atomically** (write `state.pt.tmp` → `os.replace`), at every epoch boundary:
model state_dict, optimizer, scheduler (if kept), epoch index, best-val metric + best checkpoint path, early-stop counter, and RNG states (torch, cuda, numpy, python).

On startup for a given (model, target, seed):
1. `summary.json` exists → run complete → skip.
2. `state.pt` exists → resume from last completed epoch (restore RNG → deterministic continuation; interrupted epochs restart from their boundary, so resumed runs are bit-identical to uninterrupted ones under `cudnn.deterministic=True, benchmark=False`).
3. else → fresh start.

Dataloader ordering: shuffle seed derived at each epoch start as `f(base_seed, epoch)` — no dataloader state needs saving, resume-at-epoch-boundary is trivially correct.

### 1.2 Self-chaining links

Each SLURM job ("link") is a **worker**, not a specific seed's job. At startup, the wrapper submits its own successor with `--dependency=afterany:$SLURM_JOB_ID` — *only if* the manifest still shows incomplete work at that moment, and only while `LINK_INDEX < MAX_LINKS` (runaway cap, default 10). `afterany`, not `afterok`: a crashed or timed-out link still gets its successor, which resumes. When the queue empties, the one extra successor launches, sees a complete manifest, and exits in seconds. The chain self-terminates.

Additionally, `#SBATCH --signal=B:USR1@1200` delivers a warning 20 min before the limit; the runner traps it, finishes the current epoch, checkpoints, and exits 0. Belt (signal) and braces (per-epoch checkpoints): even a hard kill mid-epoch loses nothing but that epoch.

### 1.3 Dynamic work queue with atomic claims

One global manifest of 120 task IDs `{model}_{target}_s{seed}`. Workers claim tasks via **`mkdir claims/<task_id>`** — atomic on POSIX and networked filesystems (the classic pattern; no flock, which is unreliable on shared FS). The claim dir contains `jobid` + timestamp. Stale-claim recovery: at startup, each worker lists claims, checks their jobids against `squeue`, and releases claims whose job no longer exists (crash recovery). Completion = the trainer's own `summary.json`; a task with summary present is never re-claimed.

Workers are identical and model-agnostic (one container, one env), so there is no per-chain specialisation to manage and no load imbalance: fast tasks drain fast, slow tasks share the pool.

---

## 2. Queue ordering — round-robin by seed (deadline insurance)

Tasks are ordered **breadth-first**: all 12 (model, target) cells at seed 42 first, then all at seed 43, and so on. Within a round, slowest models first (CLCRN, MTGODE, then AGCRN) to keep the tail short.

Why this matters more than it looks: within roughly the first day you hold a **complete 3×4 preliminary matrix at n=1**, and every subsequent round increments n uniformly. The draft can be written against real tables whose numbers tighten as seeds accrue — the drip-feed pattern — instead of waiting for any cell to hit n=10. This is the single biggest de-risking move for the Aug 12–13 draft date, and it costs nothing.

---

## 3. Worker layout and sizing (revised 2026-08-05)

**Snapshot vs. successors — read this first.**

**SLURM spool semantics (critical to the design):** `sbatch` spools a copy of the batch script at submission time. A pending job executes that spooled copy, NOT the file on disk. So the two initial jobids submitted 2026-08-04 (**1181540, 1181541**) will run the *original* worker.sh body (K=1, `singularity exec ... python3 -u scripts/run_worker.py &` with one child, no K_PROCS variable). File edits after their submission cannot reach them — the frozen K=1 config is guaranteed by the spool, not by any default in the new file.

**How K=3 propagates from link 2 onward:** the frozen link 1's spooled successor line calls `sbatch --dependency=afterany:$SLURM_JOB_ID --export=ALL,LINK_INDEX=N+1,... worker.sh`. That successor's SBATCH directives are read from the current `worker.sh` on disk (36h/12CPU/32G alloc), and its bash body is spooled at that moment. Link 2 starts with `K_PROCS` UNSET in env (old script didn't export it) and hits the new script's `K_PROCS="${K_PROCS:-3}"` default → runs K=3. Links 3+ inherit `K_PROCS=3` explicitly via the new script's successor export line.

**Historical bug fixed 2026-08-05:** an earlier revision had `K_PROCS="${K_PROCS:-1}"` under the misconception that frozen jobs would read the new body. This would have left link 2 running K=1 in a 36h/12CPU allocation — silent throughput failure invisible until link 2 dispatched. Corrected to `${K_PROCS:-3}`. Redeployed and grep-verified on HPC.

| Parameter | Frozen (1181540, 1181541) | **Successors (K=3)** | Rationale |
|---|---|---|---|
| Lineages | 2 | 2 | MaxSubmitPU=5 hard-caps at 2 running + 2 pending successors + 1 spare — audit correction |
| Link `--time` | 12:00:00 | **36:00:00** | MaxWall unset at QOS + partition; one admission per lineage can plausibly finish the campaign, collapsing inter-link queue-fights |
| `--mem` | 32G | 32G | Node 015 has only 44 GB free right now; 32G keeps every L40S node backfill-eligible. Do not raise. |
| `--cpus-per-task` | 8 | **12** | K=3 × OMP=2 = 6 kernel threads + I/O headroom in remaining 6; 2 workers × 12 = 24 of 32 CPU cap, within QOS |
| `--gres` | gpu:l40s:1 | gpu:l40s:1 | QOS MaxTRESPU=gpu=2 caps concurrent GPUs at 2 regardless |
| `K_PROCS` (procs per GPU) | 1 | **3** | L40S 48 GB VRAM; models ~2–3 GB each; expected compute-underutilisation at N=25 |
| `PROC_STAGGER` | n/a | 20 s | Avoids synchronised cache-load + CUDA-context creation spike |
| Signal handling | wrapper → 1 child | wrapper → K apptainer PIDs → **pgrep-P for python child** → USR1 to python | Apptainer 1.5.0 starter does NOT forward SIGUSR1 to contained Python (probed 2026-08-05); we locate the python child via `pgrep -P $starter_pid` and signal it directly. Bash `wait` returns immediately when trap fires; re-wait poll loop handles remaining children. |
| MAX_LINKS | 10 | 10 | Runaway cap unchanged |
| Signal margin | USR1 @ 1200 s | USR1 @ 1200 s | > longest plausible N=25 epoch (CLCRN ~234 s in pilot 1181304) |

### sbatch template (`slurm/worker.sh`)

```bash
#!/bin/bash
#SBATCH --partition=l40s --account=l40sugrd
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --output=logs/worker_%j.log

LINK=${LINK_INDEX:-1}
if [ "$LINK" -lt "${MAX_LINKS:-10}" ] && python scripts/manifest_incomplete.py; then
  sbatch --dependency=afterany:$SLURM_JOB_ID \
         --export=ALL,LINK_INDEX=$((LINK+1)) slurm/worker.sh
fi

singularity exec --nv "$CONTAINER" \
  python scripts/run_worker.py --wallclock-margin 1500
```

Initial submission: `for i in 1 2 3 4; do sbatch slurm/worker.sh; done` — four independent worker lineages, each self-perpetuating. Total submissions over the whole campaign: ~12–20 links, versus 120 one-per-seed jobs.

**Contingency (if QOS rejects the queued successor at startup):** fall back to submitting the successor at 50% walltime instead of at startup. Note this in `worker.sh` with a comment.

---

## 4. Runner logic (`scripts/run_worker.py`)

```
install SIGUSR1 handler → set stop_requested
loop:
    if stop_requested or remaining_walltime < margin + EMA(seed_time): checkpoint-and-exit 0
    task = claim_next_incomplete(ordered_manifest)     # mkdir claim; stale-claim sweep first
    if task is None: exit 0                            # queue drained
    train(task):                                       # resume-aware per §1.1
        per-epoch: checkpoint; append epoch_time to timing log
        divergence guard (§5)
        between epochs: if stop_requested or walltime short → checkpoint, release nothing (claim
          stays; summary absent) → exit 0; successor resumes this exact task
    on completion: trainer writes summary.json → task permanently done
    update EMA(seed_time) per model from measured epochs   # calibration is free, continuous
```

The EMA check means the worker never *starts* a seed it can't plausibly finish — but even a wrong estimate only costs a resumed epoch, never the seed. Timing estimates affect efficiency, not correctness; that is why this plan can be finalised without depending on measured timings.

**EMA seeding (per D3 refinement):** initial per-model EMA uses **full-cap** times: AGCRN 18 min, MTGODE 2.44 h, CLCRN 2.6 h (§7). Conservative on purpose — self-corrects from real completions; initial conservatism costs nothing.

---

## 5. Divergence guard (protects the chain from one bad config)

CLCRN at lr = 0.05 (Gaibie's boundary optimum) carries a real instability risk — the smoke history already showed `grad_norm=inf` under an aggressive combo. Guard, inside the epoch loop:

- non-finite train loss, or non-finite grad-norm, for **3 consecutive epochs** → write `FAILED_DIVERGED` marker into the run dir, log, **move to the next task**.
- A `FAILED_DIVERGED` task is *not* released for re-claim (no retry loops burning GPU on a broken config). The status script surfaces the diverged list.

**D5 protocol (systematic vs isolated):**
- **Isolated divergence** (any single task): report, leave `FAILED_DIVERGED`, continue. Praise decides.
- **Systematic divergence** (≥2 CLCRN runs at lr=0.05): treat as systematic. Flip lr=0.01 (Gaibie's mid-range search value) for **all** CLCRN tasks. Re-queue any CLCRN runs already completed at 0.05 (archive their outputs to `runs/archived_lr005/` — never overwrite). Report immediately.
- Learning rate stays **uniform per model** across all targets and seeds — the provenance story depends on it. Round-robin ordering means the systematic-divergence trigger fires by round 1–2; redo cost bounded at ~a dozen cheap runs.

---

## 6. Monitoring (`scripts/chain_status.sh`, head-node-safe: reads files + squeue only)

Prints: done / in-progress / unclaimed / diverged counts per (model, target); current n per cell; EMA seed times per model; projected completion at current throughput; active worker jobids and links. Run it any time; it is the drip-feed dashboard.

---

## 7. Timeline projection (measured, not extrapolated)

Per-model timings measured on L40S in **pilot job 1181304** (2026-08-04 evening; 3 epochs, seed 42, target=t2m, N=25, h=48, input=12; all three tasks completed clean):

| Model | Per-epoch (measured) | Overhead per run | Full-cap run wall | Early-stop wall (~60%) |
|---|---|---|---|---|
| AGCRN | 27 s | ~97 s | ~19 min | ~12 min |
| MTGODE | 44 s | ~98 s | ~2.5 h | ~1.5 h |
| CLCRN | 234 s | ~114 s | ~2.6 h | ~1.6 h |

Pilot loss and metric summary (sanity):
- AGCRN: train=0.5567, val=0.5809, mae_pooled_norm=0.4636
- MTGODE: train=0.4938, val=0.4951, mae_pooled_norm=0.4684
- CLCRN: train=0.2962, val=0.3193, mae_pooled_norm=0.3303 — no NaN, no divergence at lr=0.05 through 3 epochs (flag downgraded from "watch" to "monitor first prod run")

**Total campaign cost (full-cap, no early stop):** 4 targets × 10 seeds × (0.32 h AGCRN + 2.5 h MTGODE + 2.6 h CLCRN) ≈ **217 GPU-h serial**. With patience-15 early stopping at ~60%, expected: **135–215 GPU-h** (bounded above by full-cap, below by early-stop).

**Corrected elapsed projection (2026-08-05, post-audit):**
- Frozen 1181540/1181541 (K=1, 12 h) contribute ~24 GPU-h at 1× throughput = ~12 wall-h across the pair.
- Successors run K=3, blended multiplier ~2.1× (MTGODE's serial ODE drags AGCRN/CLCRN's ~3× down). Remaining ~110–190 GPU-h ÷ (2 GPUs × 2.1×) ≈ **26–45 wall-h of running time** across all successor links.
- First admission projected 2026-08-07 04:34 (`squeue --start` snapshot from live scheduler). Successor chain plausibly finishes in one 36 h admission per lineage.
- **Expected finish: 2026-08-09 to 2026-08-10** (worst case 2026-08-10 to 2026-08-11) → analysis window 2026-08-10 to 2026-08-12 → draft 2026-08-12/13 remains within reach.

**K-calibration rule (pre-committed).** On the first K=3 link, compute blended throughput multiplier m from per-epoch times logged by each of the K processes vs pilot 1181304 baselines:
- m ≥ 2.0 → K=3 stands
- m < 1.6 → next link drops to K=2 (env override `K_PROCS=2`)
- MTGODE-specific: per-epoch inflation > 2.5× under co-tenancy → report only; contingency would be a claim-rule capping MTGODE to 2 concurrent per GPU, but build nothing until numbers justify

**Round-robin ordering (§2)** naturally mixes model families per GPU (CLCRN + MTGODE + AGCRN in the same round), which is expected to help co-tenancy (memory-bound MTGODE co-scheduled with compute-bound AGCRN) rather than hurt.

**Signal-margin sanity:** 20 min USR1 margin covers ~5 CLCRN epochs. A 36 h successor link at K=3 holds ~50–70 blended-model runs — plausibly the whole remaining campaign in one admission.

If runtimes are 2× worse than projected, `MAX_LINKS=10` absorbs it; the pressure valve is dropping K to 2 (well-tested) rather than adding a lineage (blocked by MaxSubmitPU).

---

## 8. Failure-mode walkthrough (why C1 holds)

| Event | Consequence |
|---|---|
| Link hits 12 h wall clock mid-epoch | SLURM kills it; `afterany` successor resumes from last epoch checkpoint; ≤1 epoch repeated |
| Link crashes (OOM, node failure) | Same as above — `afterany`, not `afterok`, is why |
| Worker claims a task then dies | Next worker's stale-claim sweep releases it; training resumes from checkpoint |
| Config diverges (NaN) | §5 guard: marked, skipped, surfaced — chain keeps moving |
| Queue drains early | Successor links launch, see complete manifest, exit in seconds |
| Cluster queue much slower than hoped | Elapsed time stretches; nothing is lost; 5th worker + longer links are the levers |
| QOS rejects queued successor at startup | Contingency in `worker.sh`: submit successor at 50% walltime instead |

No path leads to a lost seed. That is the C1 argument in full.

---

## 9. Gates — blocking, in order

**G0 — configs frozen.** ✅ **PASSED 2026-08-04.** All three model configs diffed line-by-line against [specs/HYPERPARAMETERS.md](HYPERPARAMETERS.md); every row matches (AGCRN 13/13, CLCRN 13/13, MTGODE 21/21). CLCRN `hidden_dim=32 → rnn_units` verified at [models/baselines/CLCRN/clcrn_model.py:89](../models/baselines/CLCRN/clcrn_model.py#L89). Input `(B, 12, 25, 6)` → target `(…, 1)` shape flow confirmed: dataset yields (B,48,N,1) when `target_variable_index` is set; trainer slices `pred[..., tv_idx:tv_idx+1]` before loss ([training/trainer.py:295–298](../training/trainer.py#L295-L298)). Gaibie's "# hidden units = 16" row already resolved in [specs/HYPERPARAMETERS.md:84](HYPERPARAMETERS.md#L84) (N/A for our recurrent variant).

**G1 — resume capability verified.** *(building next)* Trainer must checkpoint per epoch atomically: model, optimizer, scheduler, epoch, best-val + path, early-stop counter, RNG states (torch CPU, cuda-all, numpy, python). Regression test: run control and interrupted-resumed under `cudnn.deterministic=True, benchmark=False`; assert **bitwise-equal** final weights + metrics. Smoke both interruption paths: `kill -9` mid-epoch (hard) and SIGUSR1 → checkpoint → exit 0 → restart resumes (graceful). Production keeps whatever determinism setting the existing protocol uses; the test forces determinism only for itself.

**G2 — claim mechanics tested locally.** Two concurrent runner processes against a toy manifest: no double-claims; stale-claim sweep releases a killed claimant's task.

**G3 — first-round sanity.** At T+3–4 h post-submission, run `chain_status.sh` and inspect the first completed CLCRN and MTGODE summaries: finite losses, val improving, physically plausible MAE magnitudes, empty diverged list. Confirm round-robin order is being honoured. Report status to Praise unprompted. Non-blocking: workers continue regardless.

---

## 10. Fallback (if dynamic claiming is judged too much machinery)

Static 5-way partition — workers own disjoint manifest slices (by model, CLCRN and MTGODE split in two) — with a manual sweeper job at the end for stragglers. Loses automatic load balancing and needs the sweeper; keeps everything in §1 (resume, self-chaining, signal handling), so C1 still holds. Choose only if G2 fails on the shared filesystem, which mkdir-claims make unlikely.

---

## 11. Decisions for Praise — ✅ APPROVED 2026-08-04

| # | Decision | Recommendation | Status |
|---|---|---|---|
| D1 | Dynamic mkdir-claim queue vs static partition | Dynamic — better utilisation, no sweeper; fallback exists | ✅ approved |
| D2 | 4 workers + reserve vs 5 workers | 4 + reserve; add the 5th after day 1 if queue waits are low | ✅ approved |
| D3 | Link duration 12 h | Accept; drop to 8 h only if backfill still stalls | ✅ approved |
| D4 | Round-robin-by-seed ordering | Accept — it is the draft-deadline insurance | ✅ approved |
| D5 | Divergence fallback lr for CLCRN if 0.05 blows up | Pre-agree 0.01 (Gaibie's AGCRN value; inside her search range) so the fix is one config flip, disclosed. Systematic trigger: ≥2 CLCRN divergences → flip all-CLCRN, archive lr=0.05 outputs to `runs/archived_lr005/`, re-queue. Isolated: leave marked, report, Praise decides. | ✅ approved |

---

## 12. File map (new)

```
scripts/run_worker.py          # claim → resume-aware train → EMA walltime logic → SIGUSR1 handler
scripts/manifest_incomplete.py # exit 0 iff work remains (wrapper's successor-submission test)
scripts/build_manifest.py      # 120 tasks, round-robin-by-seed order, slow-models-first per round
scripts/chain_status.sh        # dashboard (file reads + squeue only)
slurm/worker.sh                # §3 template
runs/claims/                   # mkdir-claim dirs
runs/manifest.json             # ordered task list
runs/archived_lr005/           # D5 systematic-divergence archive
```

Trainer changes: per-epoch atomic checkpointing + resume (G1) and the divergence guard (§5). Nothing else in the training code path is touched.

---

## 13. Standing rails (restated)

- **sbatch only, never head-node execution.** Even test runs go through the queue on HPC. `srun --overlap --jobid=<id>` inside a live allocation is the one exception (dmon utilisation probe, §7). Remote `singularity exec ... python3 -c '...'` on the head node is banned as of 2026-08-06 — codified sanctioned method for artefact inspection is `bash results/sync_from_hpc.sh` then `torch.load` locally.
- **If the environment can't satisfy any requirement, stop and report the gap — never substitute and reframe.**
- **Environment-probe rail (2026-08-06)**: any external command or capability the code depends on (`squeue`, signal delivery via apptainer, resource limits) MUST be probed in its actual execution context before being relied on, and MUST fail loud, not silent. See §13.1 for the container-`squeue`, apptainer-signal, and association-MaxWall misses that motivated this rail.
- **Skip-if-complete resumability.** Every artefact is idempotent: re-running a completed task must be a no-op.
- **Monitor policy**: exactly ONE monitor at any time, running locally, ≥5-min tick interval, ssh + squeue + ls/grep only. Ten-second-tick patterns banned.

---

## 13.1. HPC audit record (2026-08-05) — the facts we build on

Recorded verbatim so any future revisit knows exactly which knobs were what and when we looked.

**QOS `l40sugrd`:**
```
MaxTRES=(none)
MaxTRESPU=cpu=32,gres/gpu:l40s=2,gres/gpu=2
MaxJobsPU=(none)
MaxSubmitPU=5
MaxWall=(unset)
```

**Partition `l40s`:**
```
Nodes=srvrocgpu[012-016]   (5 nodes × 4 L40S GPUs = 20 GPUs total)
MaxTime=UNLIMITED
DefMemPerCPU=10000  MaxMemPerCPU=10000
TRES=cpu=240,mem=2500G,node=5,gres/gpu=20,gres/gpu:l40s=20
```

**Per-node hardware:** 48 CPU × 500 GB RAM × 4 L40S (48 GB VRAM each).

**Cluster state at 2026-08-05 ~11:41 SAST:** all 20 L40S GPUs allocated (5 nodes MIXED, 0 IDLE). 19 jobs running, 25 pending (15 Priority + 1 Resources + 9 QOSMax-blocked from other users). Our 4 initial jobs at slots 10–13 by effective priority; identical Priority=7519, identical `StartTime=2026-08-07T04:34:21`.

**Priority weights** (`scontrol show config`): Age=10 000, Partition=10 000, FairShare=1000, JobSize=1000, QOS=1000, TRES=1000/unit.
Scheduler = `sched/backfill`; SelectType = `cons_tres` (individually-allocatable GPUs).

**Live-attach probe pattern (Option Q):** `srun --overlap --jobid=<id> nvidia-smi dmon -s u -c 60` — attaches inside the existing allocation, compute runs on the node, standing rail intact.

**Apptainer signal-forwarding note (probed 2026-08-05, apptainer 1.5.0):**
- The starter process (parent apptainer PID visible to bash) **does not forward SIGUSR1** to the contained Python process by default.
- Confirmed locally: `apptainer exec ... python3 sig_echo.py &` + `kill -USR1 $starter_pid` → Python's handler never fires; process exits on 30s timeout.
- Direct signal to the Python child (located via `pgrep -P $starter_pid -f python3`) works within ~90 ms.
- Wrapper's `forward_usr1()` implements the pgrep pattern. Fallback: if no python child visible (race window during launch), it signals the starter anyway (harmless; starter ignores it).
- **C1 does not depend on graceful stop.** Even if USR1 forwarding fails entirely, the walltime hard-kill costs ≤1 epoch per process, absorbed by per-epoch checkpointing (§1.1).
