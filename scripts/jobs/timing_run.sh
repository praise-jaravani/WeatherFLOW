#!/bin/sh
# Timing run — 1 epoch at full 999 nodes to measure per-epoch wall time.
# Run this for MTGODE and ClimODE before committing to the full matrix.
# Output tells you whether to use train_single.sh (fits in 24 h) or train_array.sh.
#
# Usage:
#   sbatch --export=MODEL=mtgode scripts/jobs/timing_run.sh
#   sbatch --export=MODEL=climode scripts/jobs/timing_run.sh

#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=02:00:00
#SBATCH --job-name="wf_timing"
#SBATCH --mail-user=JRVPRA001@myuct.ac.za
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/timing-%x-%j.out

MODEL=${MODEL:-mtgode}
DATASET=${DATASET:-era5}

SIF=/scratch/jrvpra001/weatherflow/container.sif
CODE=/scratch/jrvpra001/weatherflow/WeatherFLOW

RUN="singularity exec --nv \
  --pwd /weatherflow \
  --bind ${CODE}:/weatherflow \
  --bind /scratch/jrvpra001:/scratch/jrvpra001 \
  ${SIF}"

echo "=== Timing: model=${MODEL} full 999 nodes, 1 epoch ==="
echo "=== Start: $(date) ==="

${RUN} python /weatherflow/cli.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --task regression \
    --runs 1 \
    --mlflow-off \
    --config.training.epochs 1 \
    --config.training.early_stopping_patience 999

echo "=== End: $(date) ==="
echo "=== Divide per-epoch time by ~240 (24 h / epoch_time) to get max seeds per job ==="
