"""Run_Plan_Spec §2 manifest — regression + classification builders.

Per §2 (regression): "Tasks are ordered breadth-first: all 12 (model, target)
cells at seed 42 first, then all at seed 43, and so on. Within a round,
slowest models first (CLCRN, MTGODE, then AGCRN) to keep the tail short."

This ordering is the deadline insurance — inside ~day 1 we have a complete
3×4 preliminary matrix at n=1, and every subsequent round increments n
uniformly, so partial results are always usable for the draft.

Task ID formats:
    regression:     "{model}_{target}_s{seed}"   e.g. "clcrn_t2m_s42"
    classification: "{model}_cls_s{seed}"        e.g. "clcrn_cls_s42"

Classification tasks are per-(model, seed) — one run trains a single model
that predicts all Phase-2 label variables jointly (t2m/tp/rh/ws). The "cls"
token in the target slot is the type discriminator that run_worker.py uses
to branch on when preparing the config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence


# Slowest-first per pilot 1181304 measured on L40S:
#   CLCRN ~234 s/epoch (autoregressive at h=48)
#   MTGODE ~44 s/epoch (200-epoch cap dominates)
#   AGCRN ~27 s/epoch
MODEL_ORDER: tuple[str, ...] = ("clcrn", "mtgode", "agcrn")

# Gaibie (2024) target set.
TARGET_ORDER: tuple[str, ...] = ("t2m", "sp", "rh", "ws")

# 10 seeds per (model, target) cell. Starts at 42 by project convention.
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(42, 52))  # 42..51


def make_task_id(model: str, target: str, seed: int) -> str:
    return f"{model}_{target}_s{int(seed)}"


def parse_task_id(task_id: str) -> tuple[str, str, int]:
    """Inverse of make_task_id. Raises ValueError on malformed IDs."""
    parts = task_id.split("_")
    if len(parts) < 3 or not parts[-1].startswith("s"):
        raise ValueError(f"malformed task id: {task_id!r}")
    seed = int(parts[-1][1:])
    # Everything before the last segment splits model_target — but model
    # names may contain no underscores in our 3-model set. Keep it simple:
    # model is parts[0], target is "_".join(parts[1:-1]).
    model = parts[0]
    target = "_".join(parts[1:-1])
    if not model or not target:
        raise ValueError(f"malformed task id: {task_id!r}")
    return model, target, seed


def build_manifest(
    models: Sequence[str] = MODEL_ORDER,
    targets: Sequence[str] = TARGET_ORDER,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> List[str]:
    """Return the 120-task manifest in §2 order.

    Round-robin by seed: outer loop over seeds, inner loop over (model, target)
    in the (slow-first, target-cycle) order.

    With defaults: 3 models × 4 targets × 10 seeds = 120 tasks. Round 1 is
    the first 12 IDs; round 10 the last 12.
    """
    manifest: List[str] = []
    for seed in seeds:
        for model in models:
            for target in targets:
                manifest.append(make_task_id(model, target, seed))
    return manifest


# Task-id token for classification tasks. Occupies the "target" slot in the
# {model}_{target}_s{seed} grammar and is what run_worker.py branches on to
# switch config["task"] regression→classification.
CLASSIFICATION_TOKEN = "cls"

# Models whose adapter architecture is compatible with Phase-2 multi-label
# classification (i.e., emit `(B, H, N, D_full)` so `_subset_pred_to_label_channels`
# can select the label variables). STGODE upstream's `pred` head emits a
# single scalar per (station, horizon step) — the trainer's index_select would
# raise IndexError. See specs/STGODE_INTEGRATION_AUDIT.md §12 for the full
# analysis. To include STGODE in Phase 2, monkey-patch its output head to emit
# `T_out × D_full` channels (~30 lines of adapter surgery) — deferred.
CLASSIFICATION_COMPATIBLE_MODELS = frozenset({"clcrn", "mtgode", "agcrn"})

# Phase-1 model order — includes STGODE (regression only). Slow-first per
# empirical epoch time on L40S:
#   CLCRN  ~234 s/epoch (autoregressive at H=48)
#   STGODE ~ 50 s/epoch (1.66M params, dominates the mid-tier)  — projected from local smoke
#   MTGODE ~ 44 s/epoch
#   AGCRN  ~ 27 s/epoch
# STGODE's position may need revision after we see actual L40S per-epoch times
# from the HPO trials (jobs 1221029/1221030); see specs/STGODE_INTEGRATION_AUDIT.md.
PHASE1_MODEL_ORDER: tuple[str, ...] = ("clcrn", "stgode", "mtgode", "agcrn")


def build_manifest_classification(
    models: Sequence[str] = MODEL_ORDER,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> List[str]:
    """Return the Phase-2 manifest — one task per (model, seed) tuple.

    Classification is multi-label per run (t2m/tp/rh/ws jointly), so there is
    no per-target dimension. Same round-robin-by-seed + slow-first ordering
    as the regression manifest so early completions cover all 3 models.

    With defaults: 3 models × 10 seeds = 30 tasks.

    Raises:
        ValueError if any requested model is not in
        CLASSIFICATION_COMPATIBLE_MODELS. This fail-loud guard replaces the
        cryptic `IndexError: index out of range` that the trainer would
        otherwise throw at first classification loss step. STGODE is the
        current known-incompatible model (univariate output head — see
        specs/STGODE_INTEGRATION_AUDIT.md §12).
    """
    incompatible = [m for m in models if m not in CLASSIFICATION_COMPATIBLE_MODELS]
    if incompatible:
        raise ValueError(
            f"models {incompatible} are not classification-compatible. "
            f"Their adapters emit univariate output (B,H,N,1) rather than the "
            f"per-variable (B,H,N,D_full) needed by _subset_pred_to_label_channels. "
            f"Compatible models: {sorted(CLASSIFICATION_COMPATIBLE_MODELS)}. "
            f"See specs/STGODE_INTEGRATION_AUDIT.md §12 for STGODE's case."
        )
    manifest: List[str] = []
    for seed in seeds:
        for model in models:
            manifest.append(make_task_id(model, CLASSIFICATION_TOKEN, seed))
    return manifest


def is_classification_task(task_id: str) -> bool:
    """True iff the task id encodes a classification run (target == 'cls')."""
    _, target, _ = parse_task_id(task_id)
    return target == CLASSIFICATION_TOKEN


def build_manifest_combined(
    phase1_models: Sequence[str] = PHASE1_MODEL_ORDER,
    phase2_models: Sequence[str] = MODEL_ORDER,
    targets: Sequence[str] = TARGET_ORDER,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> List[str]:
    """Return a single manifest that drains Phase 1 + Phase 2 in one chain.

    Ordering: interleaved-by-seed. For each seed, we emit all Phase-1 tasks
    (slow-first models × targets) THEN all Phase-2 tasks (slow-first models,
    no target axis). This preserves the deadline-insurance property from
    Run_Plan_Spec §2 — partial results at any point cover both phases at the
    seeds completed so far, rather than making Phase 2 wait until Phase 1 is
    100% done.

    With PHASE1_MODEL_ORDER (4 models incl. STGODE) + TARGET_ORDER (4) +
    MODEL_ORDER for Phase 2 (3 classification-compatible models) +
    DEFAULT_SEEDS (10):
      Phase 1: 4 × 4 × 10 = 160 tasks
      Phase 2: 3 ×     10 =  30 tasks
      Combined:            190 tasks

    Raises:
        ValueError if any phase2_models entry is not in
        CLASSIFICATION_COMPATIBLE_MODELS (same guard as
        build_manifest_classification).
    """
    incompatible = [m for m in phase2_models if m not in CLASSIFICATION_COMPATIBLE_MODELS]
    if incompatible:
        raise ValueError(
            f"phase2_models {incompatible} are not classification-compatible. "
            f"Compatible models: {sorted(CLASSIFICATION_COMPATIBLE_MODELS)}. "
            f"See specs/STGODE_INTEGRATION_AUDIT.md §12."
        )

    manifest: List[str] = []
    for seed in seeds:
        # Phase 1 block for this seed (models × targets, slow-first)
        for model in phase1_models:
            for target in targets:
                manifest.append(make_task_id(model, target, seed))
        # Phase 2 block for this seed (classification-compatible models only)
        for model in phase2_models:
            manifest.append(make_task_id(model, CLASSIFICATION_TOKEN, seed))
    return manifest


def dump(manifest: Sequence[str], path: Path) -> None:
    """Persist manifest to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(list(manifest), f, indent=2)


def load(path: Path) -> List[str]:
    """Load manifest from JSON."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(t, str) for t in data):
        raise ValueError(f"manifest at {path} is not a list of task IDs")
    return data


def summary_path(runs_dir: Path, task_id: str) -> Path:
    return Path(runs_dir) / task_id / "summary.json"


def diverged_marker_path(runs_dir: Path, task_id: str) -> Path:
    return Path(runs_dir) / task_id / "FAILED_DIVERGED"


def is_complete(runs_dir: Path, task_id: str) -> bool:
    """A task is complete iff its summary.json exists.

    Per Run_Plan_Spec §1.3: "Completion = the trainer's own summary.json;
    a task with summary present is never re-claimed."
    """
    return summary_path(runs_dir, task_id).exists()


def is_failed_diverged(runs_dir: Path, task_id: str) -> bool:
    """§5: a FAILED_DIVERGED task is not released for re-claim."""
    return diverged_marker_path(runs_dir, task_id).exists()


def incomplete_tasks(
    manifest: Sequence[str], runs_dir: Path
) -> List[str]:
    """Return tasks that have neither summary.json nor FAILED_DIVERGED."""
    runs_dir = Path(runs_dir)
    return [
        t for t in manifest
        if not is_complete(runs_dir, t) and not is_failed_diverged(runs_dir, t)
    ]
