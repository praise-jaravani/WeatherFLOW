#!/bin/sh
# ============================================================
# Optuna HPO sweep — sbatch array over the three production models.
# See specs/OPTUNA_HPO.md.
#
# Per-trial workflow (scripts/hpo_run.py):
#   - TPE sampler proposes a hyperparameter combination
#   - cli.py trains 1 seed for --config.training.epochs 60 with sampled config
#   - Per-trial timeout: 55 min (SBATCH_TRIAL_TIMEOUT_SEC below).
#     A trial exceeding this is killed; if it wrote any val_loss to
#     history.json before the kill, the best of those is reported to
#     Optuna as the objective. Otherwise it is pruned.
#   - Best val_loss becomes the Optuna objective
#   - Trial artefacts persisted under outputs/hpo/{model}/trials/trial_N/
#   - Study state in outputs/hpo/{model}/study.db (resumable across restarts)
#
# Submission (single command; array-style):
#   sbatch scripts/jobs/hpo_run.sbatch
#
# This produces 3 array tasks (agcrn, mtgode, clcrn). The %2 throttle caps
# concurrent tasks to 2 — matching our QOS L40S GPU cap (MaxTRESPU=2).
# The third task queues automatically and starts when one of the first two
# finishes.
#
# To resume a killed task:
#   sbatch --array=<task_id> scripts/jobs/hpo_run.sbatch
# Optuna loads existing study.db and continues from where it left off.
#
# Real HPC constraints (verified 2026-08-04):
#   MaxWall on l40s association = 24h (hard cap, can't extend)
#   MaxTRESPU on l40sugrd QOS  = 2 L40S GPUs at a time
#   MaxSubmit                   = 5 total (queued + running)
# ============================================================

#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=wf_hpo
#SBATCH --array=0-2%2
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/%x-%A_%a.out

# Map array task ID → model name
MODELS=(agcrn mtgode clcrn)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

if [ -z "${MODEL}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range" >&2
  exit 1
fi

# Overrideable via env (e.g. sbatch --export=ALL,N_TRIALS=15 ... for delta-HPO)
N_TRIALS="${N_TRIALS:-25}"
TRIAL_TIMEOUT="${TRIAL_TIMEOUT:-3300}"   # 55 min

SIF=/scratch/jrvpra001/weatherflow/container.sif
CODE=/scratch/jrvpra001/weatherflow/WeatherFLOW

echo "=== HPO sweep: model=${MODEL}, n_trials=${N_TRIALS}, timeout/trial=${TRIAL_TIMEOUT}s ==="
echo "=== Array task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="
echo "=== Resumable via SQLite at outputs/hpo/${MODEL}/study.db ==="

singularity exec --nv \
  --pwd /weatherflow \
  --bind ${CODE}:/weatherflow \
  --bind /scratch/jrvpra001:/scratch/jrvpra001 \
  ${SIF} python /weatherflow/scripts/hpo_run.py \
    --model ${MODEL} \
    --dataset saws \
    --n-trials ${N_TRIALS} \
    --trial-timeout-sec ${TRIAL_TIMEOUT}

echo "=== HPO ${MODEL} done ==="
