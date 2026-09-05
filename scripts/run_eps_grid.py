"""
Deliverable A of specs/VPT_Extended_Spec.md — ε-sensitivity grid.

Reads per-seed test_metrics.json files (already produced by cli.py runs),
extracts pooled `nrmse_per_horizon` curves per variable per seed, and
computes VFH at every ε in the grid. Pure Python + JSON I/O — no model
execution, no cache load. Runs in seconds.

Usage:
    python scripts/run_eps_grid.py \
        --outputs-dir outputs \
        --model mtgode agcrn clcrn \
        --dataset saws \
        --out-dir outputs/vpt_extended/eps_grid

Discovers per-seed runs by matching `outputs/{model}_{dataset}_regression_*/` directories.
"""

from __future__ import annotations

import os
import sys
_ROOT = "/home/praise/Computer_Science_Resources/Weatherflow"
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import argparse
import json
import logging
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from evaluation.vpt_extended import (
    DEFAULT_EPS_GRID,
    aggregate_eps_grid_across_seeds,
    eps_grid_from_pooled_curves,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _find_seed_runs(outputs_dir: Path, model: str, dataset: str) -> List[Path]:
    """Return sorted list of run directories for a given (model, dataset)."""
    pattern = str(outputs_dir / f"{model}_{dataset}_regression_*" )
    dirs = sorted(Path(d) for d in glob(pattern) if Path(d).is_dir())
    return dirs


def _extract_pooled_curves(run_dir: Path) -> Optional[Dict[str, List[float]]]:
    """Pull pooled nrmse_per_horizon dict from a run's test_metrics.json."""
    # test_metrics.json usually lives under seed_<N>/ subdirs or at run root
    candidates = list(run_dir.rglob("test_metrics.json"))
    if not candidates:
        return None
    metrics = json.loads(candidates[0].read_text())
    return metrics.get("nrmse_per_horizon")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    p.add_argument("--model", nargs="+", default=["mtgode", "agcrn", "clcrn"])
    p.add_argument("--dataset", default="saws")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/vpt_extended/eps_grid"))
    p.add_argument("--eps", nargs="+", type=float, default=list(DEFAULT_EPS_GRID))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.model:
        run_dirs = _find_seed_runs(args.outputs_dir, model, args.dataset)
        if not run_dirs:
            log.warning(f"No runs found for {model}_{args.dataset}_regression_*")
            continue
        log.info(f"{model}: discovered {len(run_dirs)} run dir(s)")

        per_seed = []
        for d in run_dirs:
            curves = _extract_pooled_curves(d)
            if curves is None:
                log.warning(f"  {d.name}: no test_metrics.json found; skipping")
                continue
            per_seed.append(eps_grid_from_pooled_curves(curves, args.eps))

        if not per_seed:
            log.warning(f"{model}: no valid runs")
            continue

        summary = aggregate_eps_grid_across_seeds(per_seed, args.eps)
        out_path = args.out_dir / f"{model}_eps_grid.json"
        out_path.write_text(json.dumps(summary, indent=2))
        log.info(f"  Wrote {out_path}")

    # Also emit a compact cross-model table for the deck
    combined = {}
    for f in sorted(args.out_dir.glob("*_eps_grid.json")):
        model = f.stem.replace("_eps_grid", "")
        combined[model] = json.loads(f.read_text())
    (args.out_dir / "all_models_eps_grid.json").write_text(
        json.dumps(combined, indent=2)
    )
    log.info(f"Wrote combined table: {args.out_dir / 'all_models_eps_grid.json'}")


if __name__ == "__main__":
    main()
