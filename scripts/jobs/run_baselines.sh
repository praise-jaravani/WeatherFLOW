#!/bin/bash
# Submit AGCRN and CLCRN as two consecutive SLURM jobs with a dependency chain.
# Second job starts only when the first completes successfully.
#
# Usage: bash scripts/jobs/run_baselines.sh <dataset> <task>
#
# Example:
#   bash scripts/jobs/run_baselines.sh saws regression

set -euo pipefail

DATASET=${1:?"Usage: bash run_baselines.sh <dataset> <task>"}
TASK=${2:?"Usage: bash run_baselines.sh <dataset> <task>"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting AGCRN..."
JOB1=$(sbatch --parsable "${SCRIPT_DIR}/run_experiment.sh" agcrn "${DATASET}" "${TASK}")
echo "  AGCRN job ID: ${JOB1}"

echo "Submitting CLCRN (depends on AGCRN)..."
JOB2=$(sbatch --parsable --dependency=afterok:"${JOB1}" \
  "${SCRIPT_DIR}/run_experiment.sh" clcrn "${DATASET}" "${TASK}")
echo "  CLCRN job ID: ${JOB2}"

echo "Submitted. Monitor with: squeue -u \$USER"
