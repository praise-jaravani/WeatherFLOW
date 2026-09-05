#!/bin/bash
# HPO worker — Option B / Gaibie protocol.
# Adapted from slurm/worker.sh (Run_Plan_Spec §3, revised 2026-08-05).
# The production `worker.sh` runs the N=32 phase-1 campaign against
# `runs/manifest.json`; this HPO worker runs a pre-sampled random-search
# manifest at `runs/hpo/manifest.json` with `--hpo-configs-dir runs/hpo/configs`
# so `run_worker.py._prepare_and_run_task` deep-merges per-trial hyperparameters
# onto the base mtgode.yaml.
#
# Everything else — K=3 co-tenancy, afterany chain, SIGUSR1 forwarding via
# pgrep, per-lineage live_jobids snapshot, environment-probe rail — is
# identical to worker.sh. See specs/HPC_Parallel_Worker_System.md for the
# full architecture.
#
# Design notes specific to HPO:
#   - Trials live under runs/hpo/{manifest,configs/,runs/,claims/}, isolated
#     from the phase-1 campaign's runs/{manifest,runs/,claims/}.
#   - `manifest_incomplete.py` is called with --manifest and --runs-dir set
#     to the HPO paths so successor chaining sees only HPO trials.
#   - Each trial is fast (~15–40 min per reviewer's Option B protocol),
#     so 60 trials × 6 concurrent (K=3 × 2 lineages) ≈ 2.5–7 h wall-clock
#     of GPU work. One 24 h link usually drains the manifest without a
#     successor needing to fire — chain kept as failure insurance.

#SBATCH --partition=l40s
#SBATCH --account=l40sugrd
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --job-name=wf_hpo
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/%x-%j.out
# Historical note: srvrocgpu014 exclusion removed 2026-08-13 after site
# remediation (other users running on it successfully per squeue).

set -u
LINK="${LINK_INDEX:-1}"
MAX_LINKS="${MAX_LINKS:-10}"
WALLCLOCK_MARGIN="${WALLCLOCK_MARGIN:-1500}"
REPO="${REPO:-/scratch/jrvpra001/weatherflow/WeatherFLOW}"
CONTAINER="${CONTAINER:-/scratch/jrvpra001/weatherflow/container.sif}"
PYPKGS="${PYPKGS:-/scratch/jrvpra001/pypkgs}"
K_PROCS="${K_PROCS:-3}"
PROC_STAGGER="${PROC_STAGGER:-20}"
LOG_DIR="/scratch/jrvpra001/weatherflow/logs"

# HPO-specific paths
HPO_ROOT="${HPO_ROOT:-${REPO}/runs/hpo}"
HPO_MANIFEST="${HPO_ROOT}/manifest.json"
HPO_CONFIGS="${HPO_ROOT}/configs"
HPO_CLAIMS="${HPO_ROOT}/claims"
HPO_RUNS="${HPO_ROOT}/runs"

echo "DISPATCH HPO LINK_INDEX=$LINK K_PROCS=$K_PROCS SLURM_JOB_ID=$SLURM_JOB_ID host=$(hostname)"
echo "HPO_ROOT=$HPO_ROOT"
mkdir -p "$HPO_CLAIMS" "$HPO_RUNS" "$LOG_DIR"

# live_jobids snapshot — same as production worker.sh (squeue not in container).
LIVE_JOBIDS_FILE="${REPO}/runs/live_jobids.txt"
LIVE_JOBIDS_TMP="${LIVE_JOBIDS_FILE}.tmp.${SLURM_JOB_ID}"
mkdir -p "${REPO}/runs"

if ! command -v squeue >/dev/null 2>&1; then
    echo "FATAL: squeue not in PATH on compute node — sweep would be silently broken." >&2
    exit 2
fi

squeue -h -u "$USER" -o '%A' > "$LIVE_JOBIDS_TMP" 2>/dev/null
mv -f "$LIVE_JOBIDS_TMP" "$LIVE_JOBIDS_FILE"
SNAPSHOT_COUNT=$(wc -l < "$LIVE_JOBIDS_FILE")
echo "SNAPSHOT n=$SNAPSHOT_COUNT jobids"
if [ "$SNAPSHOT_COUNT" -lt 1 ]; then
    echo "FATAL: initial squeue snapshot is empty. Cannot be right on a running cluster." >&2
    exit 3
fi

(
    while true; do
        sleep 300
        squeue -h -u "$USER" -o '%A' > "$LIVE_JOBIDS_TMP" 2>/dev/null && \
            mv -f "$LIVE_JOBIDS_TMP" "$LIVE_JOBIDS_FILE"
    done
) &
REFRESH_PID=$!
echo "background live-jobids refresh started (pid $REFRESH_PID, interval=300s)"
echo "=== hpo_worker link=$LINK/$MAX_LINKS jobid=$SLURM_JOB_ID host=$(hostname) ==="
echo "=== repo=$REPO container=$CONTAINER K_PROCS=$K_PROCS stagger=${PROC_STAGGER}s ==="
echo "=== manifest=$HPO_MANIFEST configs=$HPO_CONFIGS runs=$HPO_RUNS ==="
echo "=== start=$(date) ==="

cd "$REPO" || { echo "ERROR: cd $REPO failed"; exit 1; }

# ---------------------------------------------------------------------------
# Submit successor eagerly (unless capped or HPO manifest drained).
# manifest_incomplete.py is scoped to HPO manifest + HPO runs dir so this link
# only chains itself, not the phase-1 campaign.
# ---------------------------------------------------------------------------
if [ "$LINK" -lt "$MAX_LINKS" ]; then
    if singularity exec --bind "$REPO":"$REPO" --pwd "$REPO" \
        --env PYTHONPATH="$PYPKGS" "$CONTAINER" \
        python3 scripts/manifest_incomplete.py \
        --manifest "$HPO_MANIFEST" \
        --runs-dir "$HPO_RUNS" >/dev/null 2>&1; then

        NEXT=$(sbatch \
            --dependency=afterany:$SLURM_JOB_ID \
            --export=ALL,LINK_INDEX=$((LINK+1)),MAX_LINKS=$MAX_LINKS,WALLCLOCK_MARGIN=$WALLCLOCK_MARGIN,REPO=$REPO,CONTAINER=$CONTAINER,PYPKGS=$PYPKGS,K_PROCS=3,PROC_STAGGER=$PROC_STAGGER,HPO_ROOT=$HPO_ROOT \
            "$REPO/slurm/hpo_worker.sh" 2>&1) || echo "WARN: sbatch successor failed: $NEXT"
        echo "HPO successor submission: $NEXT"
    else
        echo "HPO manifest drained — no successor submitted"
    fi
else
    echo "LINK=$LINK reached MAX_LINKS=$MAX_LINKS — no successor"
fi

# ---------------------------------------------------------------------------
# Launch K parallel run_worker.py processes with HPO paths + --hpo-configs-dir.
# ---------------------------------------------------------------------------
CHILD_PIDS=()

for i in $(seq 1 "$K_PROCS"); do
    proc_log="${LOG_DIR}/wf_hpo-${SLURM_JOB_ID}-p${i}.out"
    echo "[$(date +%H:%M:%S)] launching HPO proc $i/$K_PROCS → $proc_log"

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
            --manifest "$HPO_MANIFEST" \
            --claims-dir "$HPO_CLAIMS" \
            --runs-dir "$HPO_RUNS" \
            --hpo-configs-dir "$HPO_CONFIGS" \
            --wallclock-margin "$WALLCLOCK_MARGIN" \
        >"$proc_log" 2>&1 &
    CHILD_PIDS+=($!)

    if [ "$i" -lt "$K_PROCS" ]; then
        sleep "$PROC_STAGGER"
    fi
done

echo "[$(date +%H:%M:%S)] all $K_PROCS HPO procs launched; pids=${CHILD_PIDS[*]}"

# Signal forwarding: apptainer starter does NOT forward SIGUSR1 → signal python child directly.
STOP_REQUESTED=0
forward_usr1() {
    STOP_REQUESTED=1
    echo "[$(date +%H:%M:%S)] TRAP USR1 → forwarding to ${#CHILD_PIDS[@]} apptainer starters"
    for pid in "${CHILD_PIDS[@]}"; do
        py_pid=$(pgrep -P "$pid" -f python3 | head -1)
        if [ -n "$py_pid" ]; then
            if kill -USR1 "$py_pid" 2>/dev/null; then
                echo "  starter=$pid → python=$py_pid signalled"
            else
                echo "  starter=$pid → python=$py_pid signal FAILED (already gone?)"
            fi
        else
            kill -USR1 "$pid" 2>/dev/null && echo "  starter=$pid (no py child found) signalled" \
                                        || echo "  starter=$pid already gone"
        fi
    done
}
trap forward_usr1 USR1

while (( ${#CHILD_PIDS[@]} > 0 )); do
    NEW_LIST=()
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            NEW_LIST+=("$pid")
        else
            wait "$pid" 2>/dev/null || true
            echo "[$(date +%H:%M:%S)] child $pid exited"
        fi
    done
    CHILD_PIDS=("${NEW_LIST[@]}")
    if (( ${#CHILD_PIDS[@]} > 0 )); then
        sleep 5
    fi
done

kill "$REFRESH_PID" 2>/dev/null && echo "stopped live-jobids refresher (pid $REFRESH_PID)"

echo "=== hpo_worker jobid=$SLURM_JOB_ID all $K_PROCS procs exited at $(date) ==="
exit 0
