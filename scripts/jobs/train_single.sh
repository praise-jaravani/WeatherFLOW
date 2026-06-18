#!/bin/sh
# Single model full training run — 10 seeds, both tasks.
# Parametrize with sbatch --export when submitting:
#
#   sbatch --export=MODEL=agcrn,TASK=regression scripts/jobs/train_single.sh
#   sbatch --export=MODEL=mtgode,TASK=classification scripts/jobs/train_single.sh
#
# Or edit MODEL/TASK defaults below and submit directly.

#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --job-name="wf_train"
#SBATCH --mail-user=JRVPRA001@myuct.ac.za
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/%x-%j.out

# Override with --export=MODEL=... at submission time
MODEL=${MODEL:-agcrn}
DATASET=${DATASET:-era5}
TASK=${TASK:-regression}
RUNS=${RUNS:-10}

SIF=/scratch/jrvpra001/weatherflow/container.sif
CODE=/scratch/jrvpra001/weatherflow/WeatherFLOW

RUN="singularity exec --nv \
  --bind ${CODE}:/weatherflow \
  --bind /scratch/jrvpra001:/scratch/jrvpra001 \
  ${SIF}"

echo "=== Job: model=${MODEL} dataset=${DATASET} task=${TASK} runs=${RUNS} ==="
echo "=== Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="

${RUN} python /weatherflow/cli.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --task ${TASK} \
    --runs ${RUNS}

echo "=== Done ==="
