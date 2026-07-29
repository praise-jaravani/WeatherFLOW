#!/bin/sh
# ============================================================
# climode_faithful velocity cache warmer.
#
# Per §4 of the 2026-07-28 review directive: precompute the Eq. 10
# initial-velocity fit once per window per split, store to
# preprocessing/cache/climode_faithful/velocities/<split>/win_*.npy.
# One cache serves all 5 anchor seeds.
#
# Splits and chunking (default; can override on submission):
#   --split train  --n-chunks 2     → 2 jobs, each does half of train
#   --split val    --n-chunks 1     → 1 job, all val
#   --split test   --n-chunks 1     → 1 job, all test
#
# Chunk = i modulo n_chunks; the two chunks are disjoint index sets.
# Rails: submit ≤2 concurrent per §0. Deterministic seed per window
# means both chunks produce cache entries that are compatible with
# any subsequent single-process run.
# ============================================================

#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=wf_climode_warm_cache
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/%x-%j.out

SIF=/scratch/jrvpra001/weatherflow/container.sif
CODE=/scratch/jrvpra001/weatherflow/WeatherFLOW

# Positional / passthrough — bash `${1:-default}` pattern
SPLIT="${SPLIT:-train}"
CHUNK="${CHUNK:-0}"
N_CHUNKS="${N_CHUNKS:-1}"

echo "=== climode_faithful velocity cache warmer ==="
echo "=== split=${SPLIT} chunk=${CHUNK}/${N_CHUNKS} ==="

singularity exec --nv \
  --pwd /weatherflow \
  --bind ${CODE}:/weatherflow \
  --bind /scratch/jrvpra001:/scratch/jrvpra001 \
  ${SIF} python /weatherflow/scripts/warm_velocity_cache.py \
    --split "${SPLIT}" --chunk "${CHUNK}" --n-chunks "${N_CHUNKS}"

echo "=== warmer done ==="
