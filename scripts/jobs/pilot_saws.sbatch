#!/bin/sh
# ============================================================
# SAWS PILOT — one 3-epoch run per model on L40S.
# Purpose: measure real per-epoch time + confirm CLCRN lr=0.05 stability
# across multiple epochs before committing to the full 120-run production.
#
# Time budget per task at pessimistic local extrapolation:
#   AGCRN  ~3-5 min    (3 epochs + test eval)
#   MTGODE ~15-25 min  (3 epochs + test eval; ODE-heavy)
#   CLCRN  ~30-60 min  (3 epochs + test eval; autoregressive at h=48)
# Total wall clock (2 concurrent): ~1-1.5h from job start
#
# Submission:
#   sbatch scripts/jobs/pilot_saws.sbatch
#
# Array: 0=agcrn, 1=mtgode, 2=clcrn. %2 throttles to our 2-GPU QOS cap.
# ============================================================

#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=wf_pilot
#SBATCH --array=0-2%2
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/%x-%A_%a.out

MODELS=(agcrn mtgode clcrn)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

SIF=/scratch/jrvpra001/weatherflow/container.sif
CODE=/scratch/jrvpra001/weatherflow/WeatherFLOW

echo "=== PILOT: model=${MODEL} (array task ${SLURM_ARRAY_TASK_ID}) ==="
echo "=== 3 epochs, single seed (42), univariate target=t2m ==="
echo "=== Start time: $(date) ==="

START=$(date +%s)

singularity exec --nv \
  --pwd /weatherflow \
  --bind ${CODE}:/weatherflow \
  --bind /scratch/jrvpra001:/scratch/jrvpra001 \
  --env PYTHONPATH=/scratch/jrvpra001/pypkgs \
  ${SIF} python /weatherflow/cli.py \
    --model ${MODEL} \
    --dataset saws \
    --task regression \
    --runs 1 \
    --seed-start 42 \
    --mlflow-off \
    --no-cache \
    --config.training.epochs 3

END=$(date +%s)
echo "=== PILOT ${MODEL} finished in $((END - START))s ==="
