#!/bin/bash
#SBATCH --job-name=wf_casestudy
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/wf_casestudy-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=ada

set -euo pipefail
cd /scratch/jrvpra001/weatherflow/WeatherFLOW

echo "[case_study] extract start: $(date)"
apptainer exec --nv \
    --bind /scratch/jrvpra001/weatherflow:/scratch/jrvpra001/weatherflow \
    /scratch/jrvpra001/weatherflow/container.sif \
    python3 scripts/case_study.py extract
echo "[case_study] extract done: $(date)"
