#!/bin/bash
# Manual sweep of dead-jobid claims — bash-only, runs from HPC head node.
#
# Use case: whenever a chain link exits (walltime, cancel, crash), any
# claims held by its (now-dead) SLURM jobid should be released so the
# successor can pick them up. In production this is handled by
# orchestration.claim.stale_claim_sweep, but the automatic sweep depends
# on squeue being in PATH inside the container. Prior to worker.sh's
# WF_LIVE_JOBIDS_FILE hardening (deployed 2026-08-06), sweep silently
# returned _SQUEUE_UNKNOWN inside the container.
#
# Even after the hardening, links 2 and 3 were spooled with the OLD
# worker.sh (no LIVE_JOBIDS write) so their in-container sweeps STILL
# do nothing. This script is the manual bridge until link 4+ takes over.
#
# Usage (from HPC head node):
#     bash /scratch/jrvpra001/weatherflow/WeatherFLOW/scripts/manual_sweep.sh
#
# Trigger: run promptly after a link's jobids fully leave squeue
# (detected by the local monitor on Praise's laptop).

set -eu
CLAIMS_DIR="${CLAIMS_DIR:-/scratch/jrvpra001/weatherflow/WeatherFLOW/runs/claims}"

if ! command -v squeue >/dev/null 2>&1; then
    echo "ERROR: squeue not in PATH — cannot determine live jobids." >&2
    exit 2
fi

# Live jobids for current user
LIVE=$(squeue -h -u "$USER" -o '%A' | sort -u | tr '\n' ' ')
echo "live jobids: $LIVE"

released=0
kept=0
malformed=0
for d in "$CLAIMS_DIR"/*/; do
    [ -d "$d" ] || continue
    task=$(basename "$d")
    meta="$d/meta.json"
    if [ ! -f "$meta" ]; then
        # No meta — being written or torn. Leave for next sweep.
        malformed=$((malformed + 1))
        continue
    fi
    # Extract jobid via grep/sed — bash-only, no python (per Option A bright line)
    jobid=$(grep -oE '"jobid":[[:space:]]*"[0-9]+"' "$meta" | grep -oE '[0-9]+' | head -1)
    if [ -z "$jobid" ]; then
        malformed=$((malformed + 1))
        continue
    fi
    if echo " $LIVE " | grep -q " $jobid "; then
        kept=$((kept + 1))
    else
        rm -f "$meta" && rmdir "$d" && released=$((released + 1))
    fi
done

echo "released: $released"
echo "kept:     $kept  (owned by live jobids)"
echo "malformed:$malformed  (left in place)"
echo "final claim count: $(ls "$CLAIMS_DIR" 2>/dev/null | wc -l)"
