#!/usr/bin/env python3
"""Build a manifest at:
  - runs/manifest.json      (regression, default)
  - runs/manifest_cls.json  (classification)
  - runs/manifest_all.json  (combined Phase 1 + Phase 2 for one-chain campaign)

Regression task ordering per Run_Plan_Spec §2: round-robin by seed (breadth-
first), slow models first within each round (CLCRN, STGODE, MTGODE, AGCRN).
Default: 4 models × 4 targets × 10 seeds = 160 tasks.

Classification task ordering: round-robin by seed, slow models first (CLCRN,
MTGODE, AGCRN — STGODE not classification-compatible per allowlist guard).
Default: 3 models × 10 seeds = 30 tasks (no target axis — one classification
run predicts all Phase-2 label variables jointly).

Combined ordering: interleaved-by-seed. For each seed, drops all Phase-1
tasks (models × targets) then all Phase-2 tasks (classification-compatible
models). Preserves the deadline-insurance property from Run_Plan_Spec §2 —
partial results include coverage of both phases at every seed round.
Default: 190 tasks total.

Usage:
    python scripts/build_manifest.py                              # 160 regression tasks → runs/manifest.json
    python scripts/build_manifest.py --task classification        # 30 classification tasks → runs/manifest_cls.json
    python scripts/build_manifest.py --task both                  # 190 combined tasks → runs/manifest_all.json
    python scripts/build_manifest.py --out foo.json               # custom output path
    python scripts/build_manifest.py --seeds 42 43                # limit to seeds 42, 43
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestration.manifest import (
    DEFAULT_SEEDS,
    MODEL_ORDER,
    PHASE1_MODEL_ORDER,
    TARGET_ORDER,
    build_manifest,
    build_manifest_classification,
    build_manifest_combined,
    dump,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["regression", "classification", "both"],
                    default="regression",
                    help="which manifest to build (default: regression). "
                         "'both' emits Phase 1 + Phase 2 interleaved-by-seed "
                         "for a single-chain campaign.")
    ap.add_argument("--out", default=None,
                    help="output path (default: runs/manifest.json for regression, "
                         "runs/manifest_cls.json for classification, "
                         "runs/manifest_all.json for both)")
    ap.add_argument("--models", nargs="+", default=None,
                    help=f"models in slow→fast order. Defaults per --task: "
                         f"regression uses PHASE1_MODEL_ORDER={list(PHASE1_MODEL_ORDER)}, "
                         f"classification uses MODEL_ORDER={list(MODEL_ORDER)} "
                         f"(classification-compatible only). Ignored for --task both.")
    ap.add_argument("--targets", nargs="+", default=list(TARGET_ORDER),
                    help=f"targets (regression + both; default: {list(TARGET_ORDER)})")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                    help=f"seeds (default: {list(DEFAULT_SEEDS)})")
    args = ap.parse_args()

    if args.task == "regression":
        models = args.models or list(PHASE1_MODEL_ORDER)
        tasks = build_manifest(models, args.targets, args.seeds)
        out_path = Path(args.out) if args.out else Path("runs/manifest.json")
    elif args.task == "classification":
        models = args.models or list(MODEL_ORDER)
        tasks = build_manifest_classification(models, args.seeds)
        out_path = Path(args.out) if args.out else Path("runs/manifest_cls.json")
    else:  # both
        tasks = build_manifest_combined(
            phase1_models=list(PHASE1_MODEL_ORDER),
            phase2_models=list(MODEL_ORDER),
            targets=args.targets,
            seeds=args.seeds,
        )
        out_path = Path(args.out) if args.out else Path("runs/manifest_all.json")

    dump(tasks, out_path)
    print(f"wrote {len(tasks)} {args.task} tasks → {out_path}")
    print(f"first 6: {tasks[:6]}")
    print(f"last  6: {tasks[-6:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
