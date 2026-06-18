#!/bin/sh
# Quick GPU smoke test — confirms container, GPU, and pipeline work end-to-end.
# 10 nodes, 3 epochs, 1 seed. Run this BEFORE any full production jobs.
#
# Usage:
#   sbatch scripts/jobs/smoke_test.sh
#   tail -f /scratch/jrvpra001/weatherflow/logs/smoke-<JOBID>.out

#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=01:00:00
#SBATCH --job-name="wf_smoke"
#SBATCH --mail-user=JRVPRA001@myuct.ac.za
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/smoke-%j.out

SIF=/scratch/jrvpra001/weatherflow/container.sif
CODE=/scratch/jrvpra001/weatherflow/WeatherFLOW

# Bind the live git clone over the container's embedded /weatherflow so we
# always run the latest code without rebuilding the SIF.
RUN="singularity exec --nv \
  --bind ${CODE}:/weatherflow \
  --bind /scratch/jrvpra001:/scratch/jrvpra001 \
  ${SIF}"

echo "=== GPU check ==="
${RUN} nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo ""
echo "=== PyTorch CUDA check ==="
${RUN} python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
"

echo ""
echo "=== Pipeline smoke test — AGCRN, 10 nodes, 3 epochs ==="
${RUN} python /weatherflow/cli.py \
    --model agcrn \
    --dataset era5 \
    --task regression \
    --runs 1 \
    --max-nodes 10 \
    --mlflow-off \
    --config.training.epochs 3 \
    --config.training.batch_size 512

echo ""
echo "=== Smoke test complete ==="
