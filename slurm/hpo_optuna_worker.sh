#!/bin/bash
# HPO v2 worker — sequential in-process Optuna TPE + MedianPruner.
# Single L40S, single lineage. Chained via afterany for walltime resilience.
#
# Why single-lineage (not K=3, not 2 concurrent):
#   Optuna's TPE + MedianPruner require live coupling between training loop
#   and study (trial.report + trial.should_prune mid-training). Parallelism
#   would need NFS-safe storage (JournalFileStorage) + constant_liar +
#   stale-trial handling — deferred as v3 if ever needed.
#
# Storage: SQLite on /home (backed up + single-writer safe). scripts/hpo_run_optuna.py
# handles load_if_exists + startup RUNNING→FAILED guard, so re-submitting the
# same sbatch after a walltime kill picks up cleanly.

#SBATCH --partition=l40s
#SBATCH --account=l40sugrd
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --job-name=wf_hpo_optuna
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/%x-%j.out
# Historical note: srvrocgpu014 exclusion removed 2026-08-13 after site
# remediation (other users running on it successfully per squeue).

set -u
LINK="${LINK_INDEX:-1}"
MAX_LINKS="${MAX_LINKS:-5}"
REPO="${REPO:-/scratch/jrvpra001/weatherflow/WeatherFLOW}"
CONTAINER="${CONTAINER:-/scratch/jrvpra001/weatherflow/container.sif}"
PYPKGS="${PYPKGS:-/scratch/jrvpra001/pypkgs}"

# HPO settings — MODEL selects which model's SEARCH_SPACE Optuna searches over.
MODEL="${MODEL:-mtgode}"                 # mtgode | stgode | ... (see scripts/hpo_run_optuna.py SEARCH_SPACES)
N_TRIALS="${N_TRIALS:-30}"
STUDY_NAME="${STUDY_NAME:-hpo_${MODEL}_v2}"
# Study DB on /home (backed up + single-writer safe under sequential)
STORAGE="${STORAGE:-sqlite:///${HOME}/hpo/${STUDY_NAME}.db}"
HPO_ROOT="${HPO_ROOT:-${REPO}/runs/hpo_optuna_${MODEL}}"

echo "DISPATCH_HPO_V2 LINK=$LINK/$MAX_LINKS SLURM_JOB_ID=$SLURM_JOB_ID host=$(hostname) model=$MODEL"
echo "n_trials=$N_TRIALS study=$STUDY_NAME storage=$STORAGE hpo_root=$HPO_ROOT model=$MODEL"
echo "start=$(date)"

mkdir -p "${HOME}/hpo" "${HPO_ROOT}" "$(dirname /scratch/jrvpra001/weatherflow/logs/x)"

cd "$REPO" || { echo "ERROR: cd $REPO failed" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Submit successor eagerly (unless capped). Successor's startup guard will
# handle any RUNNING trial we leave in the DB when walltime kills us.
# ---------------------------------------------------------------------------
if [ "$LINK" -lt "$MAX_LINKS" ]; then
    # A successor is worth submitting only if the study hasn't hit n_trials yet.
    # We check that inside the successor's Python script (n_remaining == 0 → exit clean),
    # so we always submit and let the successor decide it has no work.
    NEXT=$(sbatch \
        --dependency=afterany:$SLURM_JOB_ID \
        --export=ALL,LINK_INDEX=$((LINK+1)),MAX_LINKS=$MAX_LINKS,REPO=$REPO,CONTAINER=$CONTAINER,PYPKGS=$PYPKGS,MODEL=$MODEL,N_TRIALS=$N_TRIALS,STUDY_NAME=$STUDY_NAME,STORAGE="$STORAGE",HPO_ROOT=$HPO_ROOT \
        "$REPO/slurm/hpo_optuna_worker.sh" 2>&1) || echo "WARN: sbatch successor failed: $NEXT"
    echo "successor submission: $NEXT"
else
    echo "LINK=$LINK reached MAX_LINKS=$MAX_LINKS — no successor"
fi

# ---------------------------------------------------------------------------
# Signal forwarding — SIGUSR1 to wrapper → forward to child python via pgrep
# (apptainer starter doesn't forward signals — probed 2026-08-05)
# ---------------------------------------------------------------------------
CHILD_PID=""
forward_usr1() {
    echo "[$(date +%H:%M:%S)] TRAP USR1 → forwarding to apptainer starter's python child"
    if [ -n "$CHILD_PID" ]; then
        py_pid=$(pgrep -P "$CHILD_PID" -f python3 | head -1)
        if [ -n "$py_pid" ]; then
            kill -USR1 "$py_pid" 2>/dev/null && echo "  python=$py_pid signalled" || echo "  python=$py_pid signal FAILED"
        else
            kill -USR1 "$CHILD_PID" 2>/dev/null && echo "  starter=$CHILD_PID signalled" || echo "  starter=$CHILD_PID already gone"
        fi
    fi
}
trap forward_usr1 USR1

# ---------------------------------------------------------------------------
# Launch the Optuna driver inside singularity. --nv for GPU, bind /scratch
# AND ${HOME} so the SQLite study DB is visible + writeable inside the container.
# ---------------------------------------------------------------------------
proc_log="/scratch/jrvpra001/weatherflow/logs/wf_hpo_optuna-${SLURM_JOB_ID}-driver.out"
echo "[$(date +%H:%M:%S)] launching Optuna driver → $proc_log"

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
        --model "$MODEL" \
        --n-trials "$N_TRIALS" \
        --study-name "$STUDY_NAME" \
        --storage "$STORAGE" \
        --hpo-root "$HPO_ROOT" \
    >"$proc_log" 2>&1 &
CHILD_PID=$!

echo "[$(date +%H:%M:%S)] driver launched pid=$CHILD_PID"

# Wait for the driver (with re-wait loop pattern for SIGUSR1 handling)
while kill -0 "$CHILD_PID" 2>/dev/null; do
    wait "$CHILD_PID" 2>/dev/null
done
echo "[$(date +%H:%M:%S)] driver exited"

echo "=== hpo_optuna_worker jobid=$SLURM_JOB_ID exited at $(date) ==="
exit 0
