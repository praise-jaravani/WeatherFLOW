"""G2 — mkdir-claim + manifest primitives.

Covers:
  - Manifest ordering per Run_Plan_Spec §2 (round-robin-by-seed, slow-first)
  - Manifest round-trip through JSON (build → dump → load equality)
  - task_id ↔ (model, target, seed) round-trip
  - Atomic claim: two processes race, exactly one wins
  - Stale-claim sweep: released only when jobid is not live
  - Sweep refuses to release when squeue result is unknown (safety valve)
  - next_incomplete respects completion, divergence, and claims
  - Skip-if-complete: completed tasks never re-claimed
"""

import concurrent.futures
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestration import claim as claim_mod
from orchestration import manifest as manifest_mod
from orchestration.claim import (
    _SQUEUE_UNKNOWN,
    list_claimed_tasks,
    next_incomplete_task,
    read_claim_meta,
    release_claim,
    stale_claim_sweep,
    try_claim,
)
from orchestration.manifest import (
    CLASSIFICATION_TOKEN,
    DEFAULT_SEEDS,
    MODEL_ORDER,
    TARGET_ORDER,
    build_manifest,
    build_manifest_classification,
    dump,
    incomplete_tasks,
    is_classification_task,
    is_complete,
    is_failed_diverged,
    load,
    make_task_id,
    parse_task_id,
    summary_path,
    diverged_marker_path,
)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_manifest_default_120_tasks():
    m = build_manifest()
    assert len(m) == len(MODEL_ORDER) * len(TARGET_ORDER) * len(DEFAULT_SEEDS)
    assert len(m) == 120
    # No duplicates
    assert len(set(m)) == 120


def test_manifest_round_robin_by_seed():
    """Per §2: outer loop = seed, inner = (model, target). First 12 must be seed 42."""
    m = build_manifest()
    first_round = m[:12]
    for tid in first_round:
        _, _, seed = parse_task_id(tid)
        assert seed == 42, f"expected seed 42 in first round, got {seed} in {tid}"
    last_round = m[-12:]
    for tid in last_round:
        _, _, seed = parse_task_id(tid)
        assert seed == 51, f"expected seed 51 in last round, got {seed} in {tid}"


def test_manifest_slow_models_first_within_round():
    """Per §2: within a round the slow-first order (CLCRN, MTGODE, AGCRN) is honoured."""
    m = build_manifest()
    first_round = m[:12]
    # first_round is [clcrn_t2m, clcrn_sp, clcrn_rh, clcrn_ws, mtgode_..., agcrn_...]
    models_in_order = [parse_task_id(t)[0] for t in first_round]
    assert models_in_order[:4] == ["clcrn"] * 4
    assert models_in_order[4:8] == ["mtgode"] * 4
    assert models_in_order[8:] == ["agcrn"] * 4


def test_task_id_round_trip():
    for m in MODEL_ORDER:
        for t in TARGET_ORDER:
            for s in DEFAULT_SEEDS:
                tid = make_task_id(m, t, s)
                m2, t2, s2 = parse_task_id(tid)
                assert (m, t, s) == (m2, t2, s2)


def test_parse_task_id_rejects_malformed():
    for bad in ("no_seed", "agcrn_t2m_42", "agcrn_s42", "_t2m_s42", ""):
        with pytest.raises(ValueError):
            parse_task_id(bad)


def test_manifest_dump_load_round_trip(tmp_path):
    m = build_manifest(seeds=(42, 43))
    path = tmp_path / "runs" / "manifest.json"
    dump(m, path)
    assert path.exists()
    loaded = load(path)
    assert loaded == m


# ---------------------------------------------------------------------------
# Classification manifest (Phase 2)
# ---------------------------------------------------------------------------

def test_classification_manifest_default_30_tasks():
    m = build_manifest_classification()
    assert len(m) == len(MODEL_ORDER) * len(DEFAULT_SEEDS) == 30
    assert len(set(m)) == 30, "no duplicates"


def test_classification_manifest_round_robin_by_seed():
    m = build_manifest_classification()
    # First round: 3 tasks at seed 42 in slow-first (clcrn, mtgode, agcrn) order.
    first_round = m[:3]
    for tid in first_round:
        _, target, seed = parse_task_id(tid)
        assert seed == 42
        assert target == CLASSIFICATION_TOKEN
    assert [parse_task_id(t)[0] for t in first_round] == list(MODEL_ORDER)


def test_classification_task_id_format():
    tid = make_task_id("mtgode", CLASSIFICATION_TOKEN, 42)
    assert tid == "mtgode_cls_s42"
    m2, t2, s2 = parse_task_id(tid)
    assert (m2, t2, s2) == ("mtgode", "cls", 42)
    assert is_classification_task(tid)


def test_is_classification_task_discriminates():
    assert is_classification_task("mtgode_cls_s42")
    assert not is_classification_task("mtgode_t2m_s42")
    assert not is_classification_task("clcrn_sp_s51")


def test_classification_manifest_dump_load_round_trip(tmp_path):
    m = build_manifest_classification(seeds=(42, 43))
    path = tmp_path / "runs" / "manifest_cls.json"
    dump(m, path)
    assert path.exists()
    assert load(path) == m


def test_regression_and_classification_task_ids_never_collide():
    """A shared runs/claims/ directory holds both; task IDs must be unique."""
    reg = set(build_manifest())
    cls = set(build_manifest_classification())
    assert reg.isdisjoint(cls)


def test_completion_and_divergence_markers(tmp_path):
    runs = tmp_path / "runs"
    (runs / "agcrn_t2m_s42").mkdir(parents=True)
    (summary_path(runs, "agcrn_t2m_s42")).write_text("{}")
    assert is_complete(runs, "agcrn_t2m_s42")
    assert not is_failed_diverged(runs, "agcrn_t2m_s42")

    (runs / "clcrn_ws_s45").mkdir(parents=True)
    (diverged_marker_path(runs, "clcrn_ws_s45")).write_text("streak=3")
    assert not is_complete(runs, "clcrn_ws_s45")
    assert is_failed_diverged(runs, "clcrn_ws_s45")

    assert not is_complete(runs, "mtgode_rh_s50")
    assert not is_failed_diverged(runs, "mtgode_rh_s50")


def test_incomplete_tasks_filter(tmp_path):
    m = ["a", "b", "c", "d"]
    runs = tmp_path / "runs"
    (runs / "a").mkdir(parents=True)
    (summary_path(runs, "a")).write_text("{}")
    (runs / "c").mkdir(parents=True)
    (diverged_marker_path(runs, "c")).write_text("nope")
    assert incomplete_tasks(m, runs) == ["b", "d"]


# ---------------------------------------------------------------------------
# Claim: atomicity
# ---------------------------------------------------------------------------

def test_try_claim_atomicity_sequential(tmp_path):
    claims = tmp_path / "claims"
    claims.mkdir()
    assert try_claim(claims, "task1", jobid="1000") is True
    assert try_claim(claims, "task1", jobid="1001") is False  # second attempt blocked


def test_claim_writes_meta_json(tmp_path):
    claims = tmp_path / "claims"
    claims.mkdir()
    assert try_claim(claims, "task1", jobid="42") is True
    meta = read_claim_meta(claims, "task1")
    assert meta is not None
    assert meta["jobid"] == "42"
    assert "hostname" in meta and "pid" in meta and "ts" in meta


def test_release_is_idempotent(tmp_path):
    claims = tmp_path / "claims"
    claims.mkdir()
    try_claim(claims, "task1", jobid="1")
    release_claim(claims, "task1")
    release_claim(claims, "task1")  # no error on second call
    # Now re-claimable
    assert try_claim(claims, "task1", jobid="2") is True


def _claim_worker(claims_dir_str, task_id, jobid):
    """Run in a subprocess (or process pool) to attempt a claim."""
    from orchestration.claim import try_claim as _tc  # re-import in worker
    return _tc(Path(claims_dir_str), task_id, jobid)


def test_try_claim_atomicity_concurrent(tmp_path):
    """Two processes racing for the same task: exactly one wins.

    Uses ProcessPoolExecutor so we get real OS-level mkdir contention,
    not thread-level (which would be identical under GIL).
    """
    claims = tmp_path / "claims"
    claims.mkdir()
    N = 8
    with concurrent.futures.ProcessPoolExecutor(max_workers=N) as pool:
        futures = [
            pool.submit(_claim_worker, str(claims), "hot_task", f"job_{i}")
            for i in range(N)
        ]
        results = [f.result() for f in futures]
    assert sum(results) == 1, f"expected exactly 1 successful claim, got {sum(results)}"


# ---------------------------------------------------------------------------
# Stale-claim sweep
# ---------------------------------------------------------------------------

def test_stale_sweep_releases_dead_jobids(tmp_path):
    claims = tmp_path / "claims"
    claims.mkdir()
    try_claim(claims, "task_alive", jobid="1000")
    try_claim(claims, "task_dead", jobid="9999")
    released = stale_claim_sweep(claims, live_jobids={"1000"})
    assert released == ["task_dead"]
    # Alive claim untouched
    assert read_claim_meta(claims, "task_alive") is not None
    # Dead claim gone
    assert read_claim_meta(claims, "task_dead") is None
    assert "task_dead" not in list_claimed_tasks(claims)


def test_stale_sweep_refuses_when_squeue_unknown(tmp_path):
    """Safety valve — if squeue result is unavailable we must NOT release claims."""
    claims = tmp_path / "claims"
    claims.mkdir()
    try_claim(claims, "task1", jobid="1000")
    try_claim(claims, "task2", jobid="1001")
    released = stale_claim_sweep(claims, live_jobids=_SQUEUE_UNKNOWN)
    assert released == []
    assert set(list_claimed_tasks(claims)) == {"task1", "task2"}


def test_stale_sweep_skips_meta_less_claims(tmp_path):
    """A claim dir with no meta.json (torn write) is left alone for next sweep."""
    claims = tmp_path / "claims"
    claims.mkdir()
    (claims / "orphan").mkdir()  # no meta.json
    released = stale_claim_sweep(claims, live_jobids=set())
    assert released == []  # can't verify — leave for next cycle
    assert "orphan" in list_claimed_tasks(claims)


# ---------------------------------------------------------------------------
# next_incomplete_task integration
# ---------------------------------------------------------------------------

def test_next_incomplete_skips_completed_and_diverged(tmp_path):
    claims = tmp_path / "claims"
    runs = tmp_path / "runs"
    manifest = ["a", "b", "c", "d"]

    # a is complete, b is diverged, c is claimed by a live worker, d is free
    (runs / "a").mkdir(parents=True)
    (summary_path(runs, "a")).write_text("{}")
    (runs / "b").mkdir(parents=True)
    (diverged_marker_path(runs, "b")).write_text("nope")
    claims.mkdir()
    try_claim(claims, "c", jobid="1000")

    picked = next_incomplete_task(
        manifest, claims, runs, my_jobid="9999", live_jobids={"1000", "9999"}
    )
    assert picked == "d"
    # After the pick, d is claimed by us
    meta = read_claim_meta(claims, "d")
    assert meta["jobid"] == "9999"


def test_next_incomplete_releases_dead_claim_then_picks_it(tmp_path):
    """A crashed worker's claim on task X should be released and X returned to us."""
    claims = tmp_path / "claims"
    runs = tmp_path / "runs"
    manifest = ["a", "b"]

    claims.mkdir()
    try_claim(claims, "a", jobid="9999_dead")   # dead worker owns 'a'
    try_claim(claims, "b", jobid="1000_alive")  # live worker owns 'b'

    picked = next_incomplete_task(
        manifest, claims, runs, my_jobid="7777", live_jobids={"1000_alive", "7777"}
    )
    assert picked == "a"  # 'a' released by sweep, we grabbed it
    assert read_claim_meta(claims, "a")["jobid"] == "7777"
    # 'b' still owned by live worker
    assert read_claim_meta(claims, "b")["jobid"] == "1000_alive"


def test_next_incomplete_returns_none_when_drained(tmp_path):
    claims = tmp_path / "claims"
    runs = tmp_path / "runs"
    manifest = ["a", "b"]
    (runs / "a").mkdir(parents=True)
    (summary_path(runs, "a")).write_text("{}")
    (runs / "b").mkdir(parents=True)
    (summary_path(runs, "b")).write_text("{}")
    claims.mkdir()

    picked = next_incomplete_task(
        manifest, claims, runs, my_jobid="7777", live_jobids={"7777"}
    )
    assert picked is None
