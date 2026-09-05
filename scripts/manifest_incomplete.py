#!/usr/bin/env python3
"""Exit 0 iff the manifest still has incomplete work.

Called by slurm/worker.sh's successor-submission block:
    if python scripts/manifest_incomplete.py; then
      sbatch --dependency=afterany:$SLURM_JOB_ID ...
    fi

A task counts as "incomplete" if it has neither summary.json (Run_Plan_Spec
§1.3 completion marker) nor FAILED_DIVERGED (§5 skip marker).

Exit codes:
    0 → incomplete tasks remain; keep the chain alive
    1 → all done; the successor at LINK_INDEX+1 would exit in seconds anyway,
        so save the queue slot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestration.manifest import incomplete_tasks, load


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/manifest.json")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    try:
        manifest = load(Path(args.manifest))
    except FileNotFoundError:
        # No manifest yet — treat as "no work to submit for" so we exit 1.
        # (The user must run build_manifest.py before workers start.)
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    incomplete = incomplete_tasks(manifest, Path(args.runs_dir))
    if args.verbose:
        print(f"{len(incomplete)}/{len(manifest)} incomplete")
        for t in incomplete[:5]:
            print(f"  {t}")
        if len(incomplete) > 5:
            print(f"  ... and {len(incomplete) - 5} more")

    return 0 if incomplete else 1


if __name__ == "__main__":
    sys.exit(main())
