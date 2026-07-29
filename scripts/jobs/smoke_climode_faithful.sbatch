#!/bin/sh
# =====================================================================
# climode_faithful — HPC smoke sbatch (PREPARED, NOT SUBMITTED)
#
# For the Aug 5 feasibility gate. Runs one seed at the full 999-node
# grid for ~1 epoch to measure real per-epoch wall time on L40S,
# validate initial-velocity fit at production scale, and confirm the
# NLL loss stays finite.
#
# First run is expected slower than steady state because the RBF kernel
# (999×999 = ~4 MB) is precomputed once at model init but the
# initial-velocity fit runs its full 200-Adam-step optimization every
# forward pass — that's ~200 solver+backward passes per training-step
# just for velocity fitting alone. If GPU wall time exceeds projection,
# reduce vel_fit_steps in the config (upstream uses 200; halving to 100
# is a defensible tuning we'd log in PATCHES.md).
#
# DO NOT `sbatch` this until Praise reviews the local smoke output and
# the Aug 5 gate criteria. Handoff document is docs/REPORT.md.
# =====================================================================

#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --job-name=wf_climode_faithful_smoke
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/%x-%j.out

# Seed 42 by convention (see docs/TEAMMATE_INTEGRATION.md §1.3)
SEED=42

SIF=/scratch/jrvpra001/weatherflow/container.sif
CODE=/scratch/jrvpra001/weatherflow/WeatherFLOW

echo "=== climode_faithful HPC smoke: seed=${SEED} ==="
echo "=== Aug 5 feasibility gate — measure epoch time + verify no OOM ==="

singularity exec --nv \
  --pwd /weatherflow \
  --bind ${CODE}:/weatherflow \
  --bind /scratch/jrvpra001:/scratch/jrvpra001 \
  ${SIF} python /weatherflow/cli.py \
    --model climode_faithful \
    --dataset era5 \
    --task regression \
    --runs 1 \
    --seed-start ${SEED} \
    --mlflow-off

echo "=== smoke done ==="
echo "=== NEXT: read the log — first epoch time (SKIP the initial-velocity"
echo "===       .npy precompute cost if any; measure epoch 2 for steady state)"
