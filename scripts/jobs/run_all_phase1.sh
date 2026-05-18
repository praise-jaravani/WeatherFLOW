#!/bin/bash
# Batch submission: submit one regression experiment job per model for a given dataset.
#
# Usage: bash scripts/jobs/run_all_phase1.sh <dataset>
#
# Example:
#   bash scripts/jobs/run_all_phase1.sh saws
#   bash scripts/jobs/run_all_phase1.sh era5

set -euo pipefail

DATASET=${1:?"Usage: bash run_all_phase1.sh <dataset>"}
TASK="regression"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(agcrn clcrn mtgode climode stlgru litestgnn diffstg ustd)

echo "Submitting all Phase 1 ${TASK} experiments on ${DATASET}..."
for MODEL in "${MODELS[@]}"; do
  JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/run_experiment.sh" "${MODEL}" "${DATASET}" "${TASK}")
  echo "  ${MODEL}: job ${JOB_ID}"
done

echo "All jobs submitted. Monitor with: squeue -u \$USER"
