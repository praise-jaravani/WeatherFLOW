#!/bin/bash
#
# WEATHER-FLOW — submit all Phase 1 regression experiments in parallel.
#
# Submits one SLURM job per model (all 8) for the specified dataset.
# All jobs run independently in parallel on separate A100 nodes.
#
# Usage:
#   bash scripts/jobs/run_all_phase1.sh <dataset>
#   bash scripts/jobs/run_all_phase1.sh <dataset> --runs 1   # quick test
#
# Examples:
#   bash scripts/jobs/run_all_phase1.sh era5
#   bash scripts/jobs/run_all_phase1.sh saws
#   bash scripts/jobs/run_all_phase1.sh era5 --runs 1 --mlflow-off

set -euo pipefail

DATASET=${1:?"Usage: bash run_all_phase1.sh <dataset>"}
shift
EXTRA_ARGS="${*}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Baselines go first, then each workstream
MODELS=(agcrn clcrn mtgode climode stlgru litestgnn diffstg ustd)
TASK="regression"

echo "========================================================"
echo "  WEATHER-FLOW — Phase 1 batch submission"
echo "  Dataset:  ${DATASET}"
echo "  Task:     ${TASK}"
echo "  Models:   ${MODELS[*]}"
echo "  Extra:    ${EXTRA_ARGS:-none}"
echo "========================================================"
echo ""

declare -A JOB_IDS

for MODEL in "${MODELS[@]}"; do
  JOB_ID=$(sbatch --parsable \
    --job-name="wf_${MODEL}_${DATASET}" \
    "${SCRIPT_DIR}/run_experiment.sh" "${MODEL}" "${DATASET}" "${TASK}" ${EXTRA_ARGS})
  JOB_IDS[${MODEL}]=${JOB_ID}
  echo "  ${MODEL:10}  → SLURM job ${JOB_ID}"
done

echo ""
echo "All ${#MODELS[@]} Phase 1 jobs submitted."
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     outputs/slurm/"
echo ""
echo "After all jobs complete, compare results:"
echo "  python scripts/compare_results.py outputs/"
