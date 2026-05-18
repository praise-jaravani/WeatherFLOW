#!/bin/bash
#SBATCH --job-name=weatherflow
#SBATCH --partition=a100
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=outputs/slurm_%j.log
#SBATCH --error=outputs/slurm_%j.err

# Parametrised experiment runner.
# Usage: sbatch scripts/jobs/run_experiment.sh <model> <dataset> <task>
#
# Example:
#   sbatch scripts/jobs/run_experiment.sh mtgode saws regression
#
# All three parameters are required. Any additional arguments are forwarded
# to cli.py as-is (e.g. --config.lr 0.001).

set -euo pipefail

MODEL=${1:?"Usage: sbatch run_experiment.sh <model> <dataset> <task>"}
DATASET=${2:?"Usage: sbatch run_experiment.sh <model> <dataset> <task>"}
TASK=${3:?"Usage: sbatch run_experiment.sh <model> <dataset> <task>"}
shift 3
EXTRA_ARGS="$@"

CONTAINER=/home/${USER}/weatherflow/container.sif
PROJECT=/home/${USER}/weatherflow

echo "=================================================="
echo "WEATHER-FLOW Experiment"
echo "  Model:   ${MODEL}"
echo "  Dataset: ${DATASET}"
echo "  Task:    ${TASK}"
echo "  Node:    $(hostname)"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=================================================="

singularity exec --nv \
  --bind "${PROJECT}:${PROJECT}" \
  "${CONTAINER}" \
  python "${PROJECT}/cli.py" \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --task "${TASK}" \
    ${EXTRA_ARGS}
