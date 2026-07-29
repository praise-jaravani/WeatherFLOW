#!/bin/sh
# ============================================================
# climode_faithful ANCHOR MATRIX — one seed per array task.
#
# Per §5 of the 2026-07-28 mission directive:
#   - 5 seeds (42..46), regression ONLY.
#   - Config exactly as landed in experiments/configs/climode_faithful.yaml
#     (native NLL, Euler default, 300-cap patience 15, cached velocities,
#     subsampled val).
#   - Submit sequentially or 2-at-a-time per §0.
#   - Same job template as other models' seeds; write to a
#     climode_faithful-specific output directory the aggregator picks up.
#
# Environment overrides expected on submission:
#   export SEED=42
#   sbatch scripts/jobs/climode_faithful_anchor.sbatch
#
# Or as an array (if QOS allows):
#   sbatch --array=42-46%2 scripts/jobs/climode_faithful_anchor.sbatch
# (%2 caps concurrency to 2 per §0)
# ============================================================

#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --job-name=wf_climode_faithful
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/%x-%A_%a.out

SIF=/scratch/jrvpra001/weatherflow/container.sif
CODE=/scratch/jrvpra001/weatherflow/WeatherFLOW

# When run as an array, $SLURM_ARRAY_TASK_ID gives the seed.
# When run with just `sbatch` + env `SEED=N`, use that.
SEED="${SLURM_ARRAY_TASK_ID:-${SEED:-42}}"

echo "=== climode_faithful anchor: seed=${SEED} ==="
echo "=== Aug 5 gate assumed PASSED; velocity cache assumed warmed ==="
echo "=== Uses native NLL loss via adapter hook; subsampled val for checkpoint selection ==="

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

echo "=== anchor seed ${SEED} done ==="
