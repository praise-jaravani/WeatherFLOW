#!/bin/sh
# Job array — one task per seed (42-51). Use when a single 10-seed run
# won't fit in 24 h (e.g. ClimODE at 999 nodes).
#
# %2 limits concurrent tasks to 2 so we don't hog the GPU pool.
# Each task writes to its own output file: wf_climode_era5_reg-<ARRAYID>_<TASKID>.out
#
# Usage:
#   sbatch --export=MODEL=climode,TASK=regression scripts/jobs/train_array.sh

#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --job-name="wf_array"
#SBATCH --mail-user=JRVPRA001@myuct.ac.za
#SBATCH --mail-type=END,FAIL
#SBATCH --array=42-51%2
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/%x-%A_%a.out

MODEL=${MODEL:-climode}
DATASET=${DATASET:-era5}
TASK=${TASK:-regression}

SIF=/scratch/jrvpra001/weatherflow/container.sif
CODE=/scratch/jrvpra001/weatherflow/WeatherFLOW

RUN="singularity exec --nv \
  --pwd /weatherflow \
  --bind ${CODE}:/weatherflow \
  --bind /scratch/jrvpra001:/scratch/jrvpra001 \
  ${SIF}"

SEED=${SLURM_ARRAY_TASK_ID}

echo "=== Array task ${SLURM_ARRAY_TASK_ID}: model=${MODEL} task=${TASK} seed=${SEED} ==="

${RUN} python /weatherflow/cli.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --task ${TASK} \
    --runs 1 \
    --seed-start ${SEED}

echo "=== Seed ${SEED} done ==="
