#!/bin/bash
#
# WEATHER-FLOW — submit AGCRN and CLCRN as independent parallel SLURM jobs.
#
# Both baselines run on the same dataset and task simultaneously (no dependency
# chain — we want them to run in parallel to save wall time).
#
# Usage:
#   bash scripts/jobs/run_baselines.sh <dataset> <task> [extra cli args...]
#
# Examples:
#   bash scripts/jobs/run_baselines.sh era5 regression
#   bash scripts/jobs/run_baselines.sh saws classification --runs 1

set -euo pipefail

DATASET=${1:?"Usage: bash run_baselines.sh <dataset> <task>"}
TASK=${2:?"Usage: bash run_baselines.sh <dataset> <task>"}
shift 2
EXTRA_ARGS="${*}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting baseline jobs for dataset=${DATASET} task=${TASK}"
echo ""

JOB_AGCRN=$(sbatch --parsable \
  --job-name="wf_agcrn_${DATASET}" \
  "${SCRIPT_DIR}/run_experiment.sh" agcrn "${DATASET}" "${TASK}" ${EXTRA_ARGS})
echo "  AGCRN  → SLURM job ${JOB_AGCRN}"

JOB_CLCRN=$(sbatch --parsable \
  --job-name="wf_clcrn_${DATASET}" \
  "${SCRIPT_DIR}/run_experiment.sh" clcrn "${DATASET}" "${TASK}" ${EXTRA_ARGS})
echo "  CLCRN  → SLURM job ${JOB_CLCRN}"

echo ""
echo "Both jobs submitted in parallel."
echo "Monitor: squeue -u \$USER"
echo "Logs:    outputs/slurm/"
