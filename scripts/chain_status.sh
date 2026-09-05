#!/usr/bin/env bash
# Run_Plan_Spec §6 chain-status dashboard — head-node-safe.
#
# Reads files + squeue only. Never runs the trainer, never blocks a worker.
# Prints:
#   - done / in-progress / unclaimed / diverged counts per (model, target)
#   - current n per cell
#   - EMA seed times per model
#   - projected completion at current throughput
#   - active worker jobids and links
#
# Usage:
#   scripts/chain_status.sh                 # default paths (runs/, current dir)
#   REPO=/scratch/.../WeatherFLOW scripts/chain_status.sh
set -u

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
RUNS_DIR="${RUNS_DIR:-$REPO/runs}"
MANIFEST="${MANIFEST:-$RUNS_DIR/manifest.json}"
CLAIMS_DIR="${CLAIMS_DIR:-$RUNS_DIR/claims}"

echo "=== chain_status.sh ==="
echo "repo:      $REPO"
echo "runs_dir:  $RUNS_DIR"
echo "manifest:  $MANIFEST"
echo "claims:    $CLAIMS_DIR"
echo

# ---------------------------------------------------------------------------
# Delegate the file-heavy counting to Python so we can share the manifest
# parsing logic exactly with the worker.
# ---------------------------------------------------------------------------
python3 - <<'PYEOF' "$REPO" "$MANIFEST" "$RUNS_DIR" "$CLAIMS_DIR"
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

repo, manifest_path, runs_dir, claims_dir = sys.argv[1:5]
sys.path.insert(0, repo)

from orchestration.manifest import (
    load, parse_task_id, is_complete, is_failed_diverged,
)
from orchestration.claim import list_claimed_tasks, read_claim_meta

try:
    manifest = load(Path(manifest_path))
except FileNotFoundError:
    print(f"manifest not found: {manifest_path}")
    sys.exit(0)

runs = Path(runs_dir)
claims = Path(claims_dir)
claimed = set(list_claimed_tasks(claims))

# Bucket counts
buckets = {"done": [], "diverged": [], "in_progress": [], "unclaimed": []}
per_cell = defaultdict(lambda: {"done": 0, "diverged": 0, "in_progress": 0, "unclaimed": 0})
per_cell_seeds_done = defaultdict(list)

for t in manifest:
    model, target, seed = parse_task_id(t)
    if is_complete(runs, t):
        buckets["done"].append(t)
        per_cell[(model, target)]["done"] += 1
        per_cell_seeds_done[(model, target)].append(seed)
    elif is_failed_diverged(runs, t):
        buckets["diverged"].append(t)
        per_cell[(model, target)]["diverged"] += 1
    elif t in claimed:
        buckets["in_progress"].append(t)
        per_cell[(model, target)]["in_progress"] += 1
    else:
        buckets["unclaimed"].append(t)
        per_cell[(model, target)]["unclaimed"] += 1

total = len(manifest)
print(f"MANIFEST TOTAL: {total}")
print(f"  done         {len(buckets['done']):3d}  ({100*len(buckets['done'])/total:5.1f}%)")
print(f"  in_progress  {len(buckets['in_progress']):3d}")
print(f"  diverged     {len(buckets['diverged']):3d}")
print(f"  unclaimed    {len(buckets['unclaimed']):3d}")
print()

# Per-cell breakdown table (rows = model, cols = target)
targets = sorted({t for (m, t) in per_cell})
models = sorted({m for (m, t) in per_cell})
if models and targets:
    print("PER-CELL (n done / n diverged / n in_progress):")
    header = f"  {'model':<10}" + "".join(f"{t:>18}" for t in targets)
    print(header)
    for m in models:
        row = f"  {m:<10}"
        for t in targets:
            c = per_cell[(m, t)]
            cell = f"{c['done']}/{c['diverged']}/{c['in_progress']}"
            row += f"{cell:>18}"
        print(row)
    print()

# Diverged detail
if buckets["diverged"]:
    print("DIVERGED tasks (D5 protocol — flip lr if ≥2 CLCRN cases):")
    for t in buckets["diverged"]:
        print(f"  {t}")
    clcrn_div = [t for t in buckets["diverged"] if t.startswith("clcrn_")]
    if len(clcrn_div) >= 2:
        print(f"  !! {len(clcrn_div)} CLCRN divergences → §5 systematic trigger")
    print()

# In-progress with claim meta
if buckets["in_progress"]:
    print("IN-PROGRESS (task ← claim.jobid):")
    for t in buckets["in_progress"]:
        m = read_claim_meta(claims, t) or {}
        print(f"  {t:<25} ← jobid={m.get('jobid','?'):<10} host={m.get('hostname','?')}")
    print()

# EMA
ema_path = runs / "ema_seconds.json"
if ema_path.exists():
    with open(ema_path) as f:
        ema = json.load(f)
    print("EMA seconds per model (running):")
    for m, s in sorted(ema.items()):
        print(f"  {m:<10} {int(s):>6}s = {s/60:.1f} min = {s/3600:.2f} h")
    print()

# Round-robin honoured?
if buckets["done"]:
    print("SEED COVERAGE per cell (max seed completed):")
    for (m, t), seeds in sorted(per_cell_seeds_done.items()):
        print(f"  {m:<10} {t:<6} → n={len(seeds)}  seeds={sorted(seeds)}")
    print()
PYEOF

echo
# Active worker jobids from SLURM
if command -v squeue >/dev/null 2>&1; then
    echo "ACTIVE SLURM JOBS (name wf_worker or wf_pilot):"
    squeue -h -u "${USER:-$LOGNAME}" \
        -o "  %.10i %.9P %.15j %.2t %.10M %.15R" \
        | grep -E "wf_worker|wf_pilot" || echo "  (none)"
else
    echo "(squeue not available on this host)"
fi
