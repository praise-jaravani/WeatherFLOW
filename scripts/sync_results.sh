#!/bin/bash
# Pull completed run results from HPC. Idempotent — re-run anytime.
# Skips checkpoint.pt (large, opaque) but grabs metrics, logs, per-station arrays.

set -e

REMOTE=jrvpra001@hpc.uct.ac.za:/scratch/jrvpra001/weatherflow/WeatherFLOW/outputs/
LOCAL_DIR=$(dirname "$(readlink -f "$0")")/../outputs
mkdir -p "$LOCAL_DIR"

echo "Pulling from $REMOTE"
echo "        to $LOCAL_DIR"
echo

rsync -avh --progress \
  --include='*/' \
  --include='test_metrics.json' \
  --include='train.log' \
  --include='per_station.npz' \
  --include='manifest.json' \
  --exclude='checkpoint.pt' \
  --exclude='*' \
  --prune-empty-dirs \
  "$REMOTE" "$LOCAL_DIR/"

echo
echo "=== Completed runs (have test_metrics.json) ==="
find "$LOCAL_DIR" -name test_metrics.json -printf '%p\n' | sort
