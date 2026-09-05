#!/bin/bash
# AGCRN HPO worker — K=3 Optuna workers per job, afterany chain for walltime resilience.
# Mirrors hpo_clcrn_worker.sh exactly; only STUDY_NAME, HPO_ROOT, and --model differ.
# All workers share one SQLite study DB; _stop_when_complete(30) callback
# terminates each worker when 30 complete trials exist across all workers.
#
# Submit the first job with --dependency=afterany:<last_clcrn_job>:
#   sbatch --dependency=afterany:1278112 slurm/hpo_agcrn_worker.sh

#SBATCH --partition=l40s
#SBATCH --account=l40sugrd
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --job-name=wf_hpo_agcrn
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/%x-%j.out

set -u
LINK="${LINK_INDEX:-1}"
MAX_LINKS="${MAX_LINKS:-5}"
REPO="${REPO:-/scratch/jrvpra001/weatherflow/WeatherFLOW}"
CONTAINER="${CONTAINER:-/scratch/jrvpra001/weatherflow/container.sif}"
PYPKGS="${PYPKGS:-/scratch/jrvpra001/pypkgs}"
K_PROCS="${K_PROCS:-3}"
PROC_STAGGER="${PROC_STAGGER:-20}"
LOG_DIR="/scratch/jrvpra001/weatherflow/logs"

N_TRIALS="30"
STUDY_NAME="hpo_agcrn_v2"
STORAGE="sqlite:///${HOME}/hpo/${STUDY_NAME}.db"
HPO_ROOT="${REPO}/runs/hpo_optuna_agcrn"

echo "DISPATCH_HPO_AGCRN LINK=$LINK/$MAX_LINKS SLURM_JOB_ID=$SLURM_JOB_ID host=$(hostname) K_PROCS=$K_PROCS"
echo "study=$STUDY_NAME storage=$STORAGE hpo_root=$HPO_ROOT"
echo "start=$(date)"

mkdir -p "${HOME}/hpo" "${HPO_ROOT}" "${HPO_ROOT}/runs" "$LOG_DIR"
cd "$REPO" || { echo "ERROR: cd $REPO failed" >&2; exit 1; }

# Submit successor eagerly for walltime resilience.
if [ "$LINK" -lt "$MAX_LINKS" ]; then
    NEXT=$(sbatch \
        --dependency=afterany:$SLURM_JOB_ID \
        --export=ALL,LINK_INDEX=$((LINK+1)),MAX_LINKS=$MAX_LINKS,REPO=$REPO,CONTAINER=$CONTAINER,PYPKGS=$PYPKGS,K_PROCS=$K_PROCS,PROC_STAGGER=$PROC_STAGGER \
        "$REPO/slurm/hpo_agcrn_worker.sh" 2>&1) || echo "WARN: sbatch successor failed: $NEXT"
    echo "successor: $NEXT"
else
    echo "LINK=$LINK reached MAX_LINKS — no successor"
fi

# Launch K=3 Optuna worker processes (staggered to avoid DB write collisions at startup)
CHILD_PIDS=()

for i in $(seq 1 "$K_PROCS"); do
    proc_log="${LOG_DIR}/wf_hpo_agcrn-${SLURM_JOB_ID}-p${i}.out"
    echo "[$(date +%H:%M:%S)] launching Optuna worker $i/$K_PROCS → $proc_log"

    singularity exec --nv \
        --pwd "$REPO" \
        --bind "$REPO":"$REPO" \
        --bind /scratch/jrvpra001:/scratch/jrvpra001 \
        --bind "${HOME}":"${HOME}" \
        --env PYTHONPATH="$PYPKGS" \
        --env OMP_NUM_THREADS=2 \
        --env MKL_NUM_THREADS=2 \
        --env OPENBLAS_NUM_THREADS=2 \
        --env NUMEXPR_NUM_THREADS=2 \
        "$CONTAINER" \
        python3 -u scripts/hpo_run_optuna.py \
            --model agcrn \
            --n-trials "$N_TRIALS" \
            --study-name "$STUDY_NAME" \
            --storage "$STORAGE" \
            --hpo-root "$HPO_ROOT" \
            --parallel \
        >"$proc_log" 2>&1 &
    CHILD_PIDS+=($!)

    if [ "$i" -lt "$K_PROCS" ]; then
        sleep "$PROC_STAGGER"
    fi
done

echo "[$(date +%H:%M:%S)] all $K_PROCS Optuna workers launched; pids=${CHILD_PIDS[*]}"

# SIGUSR1 forwarding (singularity starter does not forward signals)
forward_usr1() {
    echo "[$(date +%H:%M:%S)] TRAP USR1 → forwarding to ${#CHILD_PIDS[@]} workers"
    for pid in "${CHILD_PIDS[@]}"; do
        py_pid=$(pgrep -P "$pid" -f python3 | head -1)
        if [ -n "$py_pid" ]; then
            kill -USR1 "$py_pid" 2>/dev/null && echo "  starter=$pid → python=$py_pid signalled" \
                || echo "  starter=$pid → python=$py_pid signal FAILED"
        else
            kill -USR1 "$pid" 2>/dev/null && echo "  starter=$pid signalled" || echo "  starter=$pid already gone"
        fi
    done
}
trap forward_usr1 USR1

# Wait for all K processes
while (( ${#CHILD_PIDS[@]} > 0 )); do
    NEW_LIST=()
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            NEW_LIST+=("$pid")
        else
            wait "$pid" 2>/dev/null || true
            echo "[$(date +%H:%M:%S)] worker pid=$pid exited"
        fi
    done
    CHILD_PIDS=("${NEW_LIST[@]}")
    if (( ${#CHILD_PIDS[@]} > 0 )); then
        sleep 5
    fi
done

echo "=== hpo_agcrn_worker jobid=$SLURM_JOB_ID all $K_PROCS workers done at $(date) ==="
exit 0
