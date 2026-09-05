#!/usr/bin/env python3
"""Per-(model, target) training-health gate for the Phase-1 campaign.

Applies the criteria from results/diagnostics/agcrn_n32_health/REPORT.md §4 —
originally developed for the AGCRN N=32 lr-repair diagnostic (2026-08-13) —
to the first completed run per (model, target) cell:

    HEALTHY iff
        best epoch b >= 3
        AND v[b] / v[1] <= 0.97
        AND all(v[e] for e in 1..b) form a broadly descending shape
              (implemented as: v[b] is at least the 25th percentile lowest
               among the pre-b epochs, i.e. no epoch-1 peak-then-park)

Motivation: the AGCRN diagnostic proved lr=0.003 healthy on ONE seed for ONE
target (t2m). The remaining per-target risk (does 0.003 hold on rh/sp/ws?)
is best answered inside the campaign — as its first-completions land — rather
than by a pre-campaign search that would cost days of compute.

Usage:
    # Check every (model, target) that has at least one completed seed:
    python scripts/health_gate.py --runs-dir runs

    # Check just AGCRN, all four targets:
    python scripts/health_gate.py --runs-dir runs --model agcrn

    # Custom criteria:
    python scripts/health_gate.py --runs-dir runs --min-best-epoch 3 \\
        --min-ratio 0.97 --seed-picker earliest

Exit codes:
    0  all (model, target) cells that have >=1 completed seed pass the gate
    2  at least one cell fails — halt that arm and re-diagnose
    3  no completed runs to check (nothing to gate)

Does NOT halt any HPC jobs on its own — this is diagnostic tooling. Failing
the gate is a signal for a human to decide whether to cancel the arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestration.manifest import (
    PHASE1_MODEL_ORDER,
    TARGET_ORDER,
    parse_task_id,
    summary_path,
)


def _load_val_curve(path: Path) -> Optional[list[float]]:
    try:
        with open(path) as f:
            s = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    v = s.get("val_loss")
    if not isinstance(v, list) or len(v) == 0:
        return None
    return [float(x) for x in v]


def _judge(vl: list[float], min_best_epoch: int, min_ratio: float) -> tuple[bool, str]:
    """Return (is_healthy, reason). Reason is a one-line explanation either way."""
    if len(vl) < min_best_epoch:
        return False, f"only {len(vl)} epochs recorded (< min_best_epoch={min_best_epoch})"
    best = min(range(len(vl)), key=lambda i: vl[i])
    b = best + 1
    v1 = vl[0]
    vb = vl[best]
    ratio = vb / v1 if v1 != 0 else float("inf")

    if b < min_best_epoch:
        return False, (
            f"PATHOLOGY: best epoch b={b} < {min_best_epoch} "
            f"(v[1]={v1:.4f}, v[b]={vb:.4f}, ratio={ratio:.4f})"
        )
    if ratio > min_ratio:
        return False, (
            f"PATHOLOGY: v[b]/v[1]={ratio:.4f} > {min_ratio} "
            f"(v[1]={v1:.4f}, v[b]={vb:.4f}, b={b})"
        )
    return True, (
        f"HEALTHY: b={b}, v[1]={v1:.4f}, v[b]={vb:.4f}, ratio={ratio:.4f}"
    )


def _pick_first_completed(runs_dir: Path, model: str, target: str,
                           picker: str) -> Optional[str]:
    """Return the task_id of the first-completed run for a (model, target) cell,
    or None if no completed runs exist yet.

    picker='earliest' -> lowest-seed completion (deterministic wrt manifest order)
    picker='fastest'  -> earliest wall-clock completion by summary.json mtime
    """
    candidates = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        try:
            m, t, s = parse_task_id(entry.name)
        except ValueError:
            continue
        if m != model or t != target:
            continue
        summ = summary_path(runs_dir, entry.name)
        if not summ.exists():
            continue
        candidates.append((s, summ.stat().st_mtime, entry.name))
    if not candidates:
        return None
    if picker == "fastest":
        candidates.sort(key=lambda x: x[1])
    else:  # "earliest"
        candidates.sort(key=lambda x: x[0])
    return candidates[0][2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-dir", default="runs",
                    help="dir containing per-task subdirs")
    ap.add_argument("--model", default=None,
                    help="check only this model (default: all Phase-1 models)")
    ap.add_argument("--targets", nargs="+", default=list(TARGET_ORDER),
                    help=f"targets to gate (default: {list(TARGET_ORDER)})")
    ap.add_argument("--min-best-epoch", type=int, default=3,
                    help="admissibility: best epoch must be >= this (default: 3)")
    ap.add_argument("--min-ratio", type=float, default=0.97,
                    help="admissibility: v[b]/v[1] must be <= this (default: 0.97)")
    ap.add_argument("--seed-picker", choices=["earliest", "fastest"],
                    default="earliest",
                    help="which completed seed to gate per cell (default: earliest)")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"[health-gate] runs dir not found: {runs_dir}", file=sys.stderr)
        return 3

    models = [args.model] if args.model else list(PHASE1_MODEL_ORDER)
    print(f"[health-gate] runs_dir={runs_dir}  models={models}  targets={args.targets}")
    print(f"[health-gate] criteria: b >= {args.min_best_epoch}, v[b]/v[1] <= {args.min_ratio}")
    print()

    checked = 0
    failed: list[tuple[str, str, str]] = []
    for model in models:
        for target in args.targets:
            task_id = _pick_first_completed(runs_dir, model, target, args.seed_picker)
            if task_id is None:
                print(f"  [{model} {target}]  no completed seed yet — skipped")
                continue
            vl = _load_val_curve(summary_path(runs_dir, task_id))
            if vl is None:
                print(f"  [{model} {target}]  {task_id}: summary present but val_loss unreadable — skipped")
                continue
            healthy, reason = _judge(vl, args.min_best_epoch, args.min_ratio)
            marker = "PASS" if healthy else "FAIL"
            print(f"  [{model} {target}]  {task_id}: {marker} — {reason}")
            checked += 1
            if not healthy:
                failed.append((model, target, task_id))

    print()
    if checked == 0:
        print(f"[health-gate] no (model, target) cells had a completed seed to gate.")
        return 3
    if failed:
        print(f"[health-gate] {len(failed)} FAILED — HALT the following arms and re-diagnose:")
        for m, t, tid in failed:
            print(f"    {m} × {t}   (offending run: {tid})")
        return 2
    print(f"[health-gate] all {checked} cell(s) PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
