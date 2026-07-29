#!/bin/sh
# =====================================================================
# climode_faithful — HPC smoke sbatch (PREPARED, NOT SUBMITTED)
#
# For the Aug 5 feasibility gate. Runs one seed on the full 999-node
# grid for 2 epochs to measure real per-epoch wall time on L40S, split
# by stage, and confirm the NLL loss stays finite and drops.
#
# CRITICAL fixes vs the earlier draft (per §4 of the 2026-07-28 review):
#
#   - Explicit epochs=2 override (was: run to patience under 300-cap → wall kill).
#   - patience=1 so early stopping does not fire in a 2-epoch smoke.
#   - --uncached-velocity flag: the smoke fits velocities fresh every
#     forward pass (worst-case cost). This sizes the cache win for the
#     REPORT.md projection. Real training will use the cache.
#   - Print per-stage timings (vel-fit / ODE / emission / backward) via
#     the smoke-mode instrumentation in the model.
#   - Assert NLL finite + decreasing across epoch 1 → 2 (via the
#     dedicated smoke script, not the full CLI, so the wall gate is
#     the timing measurement, not a full-run success).
#   - Use scripts/smoke_climode_faithful_hpc.py rather than cli.py so
#     the eval is a small fixed-window subset (~500 windows) — full
#     eval on 46K test windows would be 40+ hours of CPU cached
#     (see docs/REPORT.md §Validation cost).
#
# DO NOT `sbatch` this until Praise reviews docs/REPORT.md and
# docs/smoke_climode_faithful_output.json.
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
echo "=== Aug 5 feasibility gate — 2-epoch measurement (NOT a real training run) ==="
echo "=== Uncached velocity fits by design to measure worst-case cost. ==="

singularity exec --nv \
  --pwd /weatherflow \
  --bind ${CODE}:/weatherflow \
  --bind /scratch/jrvpra001:/scratch/jrvpra001 \
  ${SIF} python /weatherflow/scripts/smoke_climode_faithful_hpc.py \
    --seed ${SEED} \
    --epochs 2 \
    --patience 100 \
    --eval-subsample 500 \
    --uncached-velocity \
    --report-per-stage-timings

echo "=== smoke done ==="
echo "=== Aug 5 GATE CRITERIA (Praise applies): ==="
echo "===   GO if:  epoch-2 steady-state ≤ ~2 h,"
echo "===          projected seed (converged epochs × epoch time + cached eval) ≤ ~22 h,"
echo "===          no OOM,"
echo "===          NLL finite and decreasing across epoch 1→2."
echo "===   NO-GO → measured-infeasibility descope per the standing plan."
