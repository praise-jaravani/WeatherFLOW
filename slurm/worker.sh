#!/bin/bash
# Run_Plan_Spec §3 (revised 2026-08-05) — SAWS production worker.
#
# Design change from initial submission (jobids 1181540/1181541 run the frozen
# K=1, 12h, 8-CPU snapshot as tested):
#   - 2 surviving lineages after Correction 2 (MaxSubmitPU=5 hard cap)
#   - Successor links go 36h / 12 CPU / 32G / K=3 processes per GPU
#   - Signal forwarding to all K child PIDs, re-wait loop after trap fires
#   - Per-process log files under logs/wf_worker-<jobid>-p<i>.out
#
# Each job is a *worker*, not a specific seed's job. On startup it (a) submits
# its successor with afterany dependency iff manifest still has work AND
# LINK_INDEX < MAX_LINKS; (b) launches K parallel run_worker.py processes,
# each claiming its own tasks from the shared manifest via atomic mkdir-claim.
#
# --signal=B:USR1@1200 delivers SIGUSR1 to the batch script 20 min before
# --time expires. The wrapper trap forwards to every child PID and then
# re-waits — bash's `wait` returns as soon as the trap runs, so we must
# wait again to actually see children exit cleanly after they checkpoint.
#
# Env overrides (all optional):
#   CONTAINER=/scratch/.../container.sif   (required in production)
#   MAX_LINKS=10                            (runaway cap; default 10)
#   WALLCLOCK_MARGIN=1500                   (worker-internal margin, seconds)
#   REPO=/scratch/.../WeatherFLOW           (repo root; default: cwd)
#   K_PROCS=3                               (concurrent trainings per GPU; default 3)
#   PROC_STAGGER=20                         (seconds between process launches; default 20)
#   MANIFEST=runs/manifest.json             (which manifest this chain drains;
#                                            set to runs/manifest_cls.json for
#                                            the Phase-2 classification chain.
#                                            Successors inherit via --export=ALL.)

#SBATCH --partition=l40s
#SBATCH --account=l40sugrd
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --job-name=wf_worker
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/%x-%j.out
# --time=24:00:00 : association MaxWall (l40sugrd) hard cap. Discovered
#   2026-08-05 when frozen 1181540/1181541 submitted successors that were
#   rejected with AssocMaxWallDurationPerJobLimit at the 36h attempt.
# Historical note: srvrocgpu014 was defensively excluded 2026-08-05 after
#   two consecutive jobs hit uncorrectable ECC errors on it. Exclusion
#   removed 2026-08-13 after observing site users running successfully on
#   that node (per squeue), confirming site has since remediated.

set -u
LINK="${LINK_INDEX:-1}"
MAX_LINKS="${MAX_LINKS:-10}"
WALLCLOCK_MARGIN="${WALLCLOCK_MARGIN:-1500}"
REPO="${REPO:-/scratch/jrvpra001/weatherflow/WeatherFLOW}"
CONTAINER="${CONTAINER:-/scratch/jrvpra001/weatherflow/container.sif}"
PYPKGS="${PYPKGS:-/scratch/jrvpra001/pypkgs}"
# K_PROCS default is 3 — the intended co-tenant config for anyone
# executing THIS script body. Critical correction 2026-08-05:
# SLURM spools batch scripts at submission time. The frozen jobs
# 1181540/1181541 run their SPOOLED old-script copy which single-launches
# K=1 in its own 8-CPU/12h allocation — they can NEVER execute this
# body, so we do not need the default to protect them. The default's
# only real consumers are:
#   (a) link 2 dispatched by a frozen job's OLD successor line
#       (which does NOT export K_PROCS — the variable didn't exist yet)
#   (b) manual submissions of this file
# Both should run K=3. The hardcoded K_PROCS=3 on the new successor
# export line below is belt-and-braces for links 3+.
K_PROCS="${K_PROCS:-3}"
PROC_STAGGER="${PROC_STAGGER:-20}"
LOG_DIR="/scratch/jrvpra001/weatherflow/logs"
# Manifest to drain. Regression campaign uses the default; Phase-2
# classification campaign submits with MANIFEST=runs/manifest_cls.json.
# Path is relative to REPO or absolute — resolved just before use.
MANIFEST_REL="${MANIFEST:-runs/manifest.json}"
case "$MANIFEST_REL" in
    /*) MANIFEST_PATH="$MANIFEST_REL" ;;
    *)  MANIFEST_PATH="$REPO/$MANIFEST_REL" ;;
esac

# One-line greppable marker — search logs for "DISPATCH " to confirm K.
# The K_PROCS value in this line is the definitive proof that link 2+ ran
# the intended co-tenant config (K=3), not a silent fallback to K=1.
# MANIFEST is included so cls vs regression chains are distinguishable in logs.
echo "DISPATCH LINK_INDEX=$LINK K_PROCS=$K_PROCS SLURM_JOB_ID=$SLURM_JOB_ID host=$(hostname) MANIFEST=$MANIFEST_PATH"

# -------------------------------------------------------------------------
# live_jobids snapshot (2026-08-06 fix)
# squeue is NOT available inside the singularity container. Without a
# live-jobids source, orchestration.claim.live_squeue_jobids() would return
# _SQUEUE_UNKNOWN forever and sweep would spare all claims — leaving
# dead-jobid claims locked (this actually happened on link 2).
#
# Solution: the wrapper (on the compute node, where squeue IS available)
# writes squeue's output to a shared file every 5 min. run_worker.py inside
# the container reads that file via env var WF_LIVE_JOBIDS_FILE.
# -------------------------------------------------------------------------
LIVE_JOBIDS_FILE="${REPO}/runs/live_jobids.txt"
LIVE_JOBIDS_TMP="${LIVE_JOBIDS_FILE}.tmp.${SLURM_JOB_ID}"   # per-lineage tmp to avoid mv races
mkdir -p "${REPO}/runs"

# Env-probe rail (added per 2026-08-06 review): the code below assumes
# squeue is in PATH on the compute node. Verify it and fail LOUD if missing.
if ! command -v squeue >/dev/null 2>&1; then
    echo "FATAL: squeue not in PATH on compute node — sweep would be silently broken." >&2
    echo "This is a violation of the environment-probe rail. Aborting." >&2
    exit 2
fi

# Initial snapshot — ATOMIC write (temp file then mv)
squeue -h -u "$USER" -o '%A' > "$LIVE_JOBIDS_TMP" 2>/dev/null
mv -f "$LIVE_JOBIDS_TMP" "$LIVE_JOBIDS_FILE"
SNAPSHOT_COUNT=$(wc -l < "$LIVE_JOBIDS_FILE")
# Fail-loud proof marker: log line grep-able as "SNAPSHOT n="
echo "SNAPSHOT n=$SNAPSHOT_COUNT jobids"
if [ "$SNAPSHOT_COUNT" -lt 1 ]; then
    echo "FATAL: initial squeue snapshot is empty. Cannot be right on a running cluster." >&2
    exit 3
fi

# Background refresh every 5 min — atomic per iteration, per-lineage tmp
(
    while true; do
        sleep 300
        squeue -h -u "$USER" -o '%A' > "$LIVE_JOBIDS_TMP" 2>/dev/null && \
            mv -f "$LIVE_JOBIDS_TMP" "$LIVE_JOBIDS_FILE"
    done
) &
REFRESH_PID=$!
echo "background live-jobids refresh started (pid $REFRESH_PID, interval=300s)"
echo "=== worker link=$LINK/$MAX_LINKS jobid=$SLURM_JOB_ID host=$(hostname) ==="
echo "=== repo=$REPO container=$CONTAINER K_PROCS=$K_PROCS stagger=${PROC_STAGGER}s ==="
echo "=== walltime=24h(sbatch-fixed) margin=${WALLCLOCK_MARGIN}s ==="
echo "=== start=$(date) ==="

cd "$REPO" || { echo "ERROR: cd $REPO failed"; exit 1; }

# ---------------------------------------------------------------------------
# Submit successor eagerly (unless capped or manifest drained)
# ---------------------------------------------------------------------------
if [ "$LINK" -lt "$MAX_LINKS" ]; then
    if singularity exec --bind "$REPO":"$REPO" --pwd "$REPO" \
        --env PYTHONPATH="$PYPKGS" "$CONTAINER" \
        python3 scripts/manifest_incomplete.py \
        --manifest "$MANIFEST_PATH" \
        --runs-dir "$REPO/runs" >/dev/null 2>&1; then

        # Successors always run K=3 regardless of parent's K_PROCS.
        # This is what turns the frozen initial K=1 links into K=3 chains.
        # MANIFEST forwarded so cls / regression chains stay on their own manifest.
        NEXT=$(sbatch \
            --dependency=afterany:$SLURM_JOB_ID \
            --export=ALL,LINK_INDEX=$((LINK+1)),MAX_LINKS=$MAX_LINKS,WALLCLOCK_MARGIN=$WALLCLOCK_MARGIN,REPO=$REPO,CONTAINER=$CONTAINER,PYPKGS=$PYPKGS,K_PROCS=3,PROC_STAGGER=$PROC_STAGGER,MANIFEST=$MANIFEST_REL \
            "$REPO/slurm/worker.sh" 2>&1) || echo "WARN: sbatch successor failed: $NEXT"
        echo "successor submission: $NEXT"
    else
        echo "manifest drained — no successor submitted"
    fi
else
    echo "LINK=$LINK reached MAX_LINKS=$MAX_LINKS — no successor"
fi

# ---------------------------------------------------------------------------
# Launch K parallel run_worker.py processes, each pinned to per-process logs.
# Thread caps prevent oversubscription of the 12 allocated CPUs.
# ---------------------------------------------------------------------------
CHILD_PIDS=()

for i in $(seq 1 "$K_PROCS"); do
    proc_log="${LOG_DIR}/wf_worker-${SLURM_JOB_ID}-p${i}.out"
    echo "[$(date +%H:%M:%S)] launching proc $i/$K_PROCS → $proc_log"

    # OMP_NUM_THREADS=2, MKL/torch threads 2 → 3 procs × 2 threads = 6 CPUs
    # for kernels + I/O headroom in the remaining 6 of our 12 CPU allocation.
    # num_workers=0 in the DataLoader (Trainer default) — no forked dataloader
    # workers that would multiply the CPU footprint.
    singularity exec --nv \
        --pwd "$REPO" \
        --bind "$REPO":"$REPO" \
        --bind /scratch/jrvpra001:/scratch/jrvpra001 \
        --env PYTHONPATH="$PYPKGS" \
        --env OMP_NUM_THREADS=2 \
        --env MKL_NUM_THREADS=2 \
        --env OPENBLAS_NUM_THREADS=2 \
        --env NUMEXPR_NUM_THREADS=2 \
        --env WF_LIVE_JOBIDS_FILE="$LIVE_JOBIDS_FILE" \
        "$CONTAINER" \
        python3 -u scripts/run_worker.py \
            --manifest "$MANIFEST_PATH" \
            --claims-dir "$REPO/runs/claims" \
            --runs-dir "$REPO/runs" \
            --wallclock-margin "$WALLCLOCK_MARGIN" \
        >"$proc_log" 2>&1 &
    CHILD_PIDS+=($!)

    # Stagger process starts so they don't collide on the first cache-load
    # + initial CUDA-context creation. K=3 × 20s = 40s spread — trivial vs
    # the multi-hour runtime, but avoids a synchronised OOM/thrash spike.
    if [ "$i" -lt "$K_PROCS" ]; then
        sleep "$PROC_STAGGER"
    fi
done

echo "[$(date +%H:%M:%S)] all $K_PROCS procs launched; pids=${CHILD_PIDS[*]}"

# ---------------------------------------------------------------------------
# Signal forwarding — SIGUSR1 to wrapper → all children.
# `wait` returns immediately when a trap fires; we must re-wait in a loop.
# ---------------------------------------------------------------------------
STOP_REQUESTED=0
forward_usr1() {
    STOP_REQUESTED=1
    echo "[$(date +%H:%M:%S)] TRAP USR1 → forwarding to ${#CHILD_PIDS[@]} apptainer starters"
    # CRITICAL (probed 2026-08-05 with apptainer 1.5.0): the apptainer
    # starter process does NOT forward SIGUSR1 to the contained Python.
    # Signal the Python child of each starter directly.
    for pid in "${CHILD_PIDS[@]}"; do
        py_pid=$(pgrep -P "$pid" -f python3 | head -1)
        if [ -n "$py_pid" ]; then
            if kill -USR1 "$py_pid" 2>/dev/null; then
                echo "  starter=$pid → python=$py_pid signalled"
            else
                echo "  starter=$pid → python=$py_pid signal FAILED (already gone?)"
            fi
        else
            # Fallback: no python child visible — try the starter itself.
            # Harmless (starter ignores unhandled signals) and covers races.
            kill -USR1 "$pid" 2>/dev/null && echo "  starter=$pid (no py child found) signalled" \
                                       || echo "  starter=$pid already gone"
        fi
    done
}
trap forward_usr1 USR1

# Poll-wait loop: wait on each child; on trap, `wait` returns non-zero
# and we loop through the remaining children again. Each child either
# exits normally (task done) or after checkpointing (USR1 caught).
while (( ${#CHILD_PIDS[@]} > 0 )); do
    NEW_LIST=()
    for pid in "${CHILD_PIDS[@]}"; do
        # kill -0 returns success iff pid exists; failure iff pid gone/reaped
        if kill -0 "$pid" 2>/dev/null; then
            NEW_LIST+=("$pid")
        else
            wait "$pid" 2>/dev/null || true
            echo "[$(date +%H:%M:%S)] child $pid exited"
        fi
    done
    CHILD_PIDS=("${NEW_LIST[@]}")
    if (( ${#CHILD_PIDS[@]} > 0 )); then
        # short sleep prevents busy-looping; USR1 still lands via bash trap
        sleep 5
    fi
done

# Stop the background live-jobids refresher (cleanup)
kill "$REFRESH_PID" 2>/dev/null && echo "stopped live-jobids refresher (pid $REFRESH_PID)"

echo "=== worker jobid=$SLURM_JOB_ID all $K_PROCS procs exited at $(date) ==="
exit 0
