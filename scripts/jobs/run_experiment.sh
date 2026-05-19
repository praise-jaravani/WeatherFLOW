#!/bin/bash
#
# WEATHER-FLOW — parametrised SLURM experiment runner
#
# Runs one model × dataset × task combination inside the Singularity container
# on an A100 GPU node at UCT HPC.
#
# Usage:
#   sbatch scripts/jobs/run_experiment.sh <model> <dataset> <task> [extra cli args...]
#
# Examples:
#   sbatch scripts/jobs/run_experiment.sh agcrn  era5 regression
#   sbatch scripts/jobs/run_experiment.sh mtgode saws classification
#   sbatch scripts/jobs/run_experiment.sh agcrn  era5 regression --runs 1 --mlflow-off
#
# The SLURM log for this job is written to:
#   outputs/slurm/<model>_<dataset>_<task>_<job_id>.log

#SBATCH --job-name=wf
#SBATCH --partition=a100
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/%u/weatherflow/outputs/slurm/%x_%j.log
#SBATCH --error=/home/%u/weatherflow/outputs/slurm/%x_%j.err

set -euo pipefail

# ── Parameters ────────────────────────────────────────────────────────────────
MODEL=${1:?"Usage: sbatch run_experiment.sh <model> <dataset> <task>"}
DATASET=${2:?"Usage: sbatch run_experiment.sh <model> <dataset> <task>"}
TASK=${3:?"Usage: sbatch run_experiment.sh <model> <dataset> <task>"}
shift 3
EXTRA_ARGS="${*}"

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT="/home/${USER}/weatherflow"
CONTAINER="${PROJECT}/container.sif"

# Create SLURM log directory and outputs directory (in case not present yet)
mkdir -p "${PROJECT}/outputs/slurm"
mkdir -p "${PROJECT}/outputs"

# ── Environment checks ────────────────────────────────────────────────────────
echo "========================================================"
echo "  WEATHER-FLOW Experiment"
echo "  Model:      ${MODEL}"
echo "  Dataset:    ${DATASET}"
echo "  Task:       ${TASK}"
echo "  Extra args: ${EXTRA_ARGS:-none}"
echo "  SLURM job:  ${SLURM_JOB_ID}"
echo "  Node:       $(hostname)"
echo "  Started:    $(date)"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================================"

# Verify container exists
if [[ ! -f "${CONTAINER}" ]]; then
  echo "ERROR: Container not found at ${CONTAINER}"
  echo "       Run: make upload-container HPC_USER=${USER}"
  exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────────────
START=$(date +%s)

singularity exec --nv \
  --bind "${PROJECT}:${PROJECT}" \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH="${PROJECT}" \
  --pwd "${PROJECT}" \
  "${CONTAINER}" \
  python cli.py \
    --model   "${MODEL}" \
    --dataset "${DATASET}" \
    --task    "${TASK}" \
    ${EXTRA_ARGS}

END=$(date +%s)
ELAPSED=$(( END - START ))
echo ""
echo "========================================================"
echo "  Done. Wall time: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "  Finished: $(date)"
echo "========================================================"
