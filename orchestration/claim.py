"""Run_Plan_Spec §1.3 mkdir-claim primitives.

Atomic on POSIX and cluster filesystems (Lustre, GPFS) because mkdir
creates the directory-inode entry atomically — if two workers race,
exactly one succeeds and the other's mkdir raises FileExistsError.
No flock, which is unreliable on shared filesystems.

Claim dir layout:
    runs/claims/<task_id>/
        meta.json    { "jobid": "<slurm_jobid>", "hostname": "...", "ts": <epoch>, "pid": <pid> }

Stale-claim recovery: on worker startup, list all claim dirs, extract each
jobid, and release any whose jobid is not in the live_jobids set.
"""

from __future__ import annotations

import errno
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

# Time is used by live_squeue_jobids for the staleness guard.


CLAIM_META_FILENAME = "meta.json"


def _meta_path(claims_dir: Path, task_id: str) -> Path:
    return Path(claims_dir) / task_id / CLAIM_META_FILENAME


def try_claim(claims_dir: Path, task_id: str, jobid: str) -> bool:
    """Atomically claim `task_id` by mkdir-ing its claim dir.

    Returns True iff this call created the claim (i.e. this worker owns
    the task). Returns False if another worker already claimed it.

    On success writes meta.json with jobid/hostname/ts/pid so the
    stale-claim sweep can identify the owning process.
    """
    claim_dir = Path(claims_dir) / task_id
    try:
        # mkdir is atomic — this is the load-bearing atomicity guarantee.
        os.mkdir(claim_dir)
    except FileExistsError:
        return False
    except OSError as e:
        # mkdir may fail for reasons other than "already exists" — permission,
        # missing parent, network hiccup. Propagate so the worker surfaces it.
        if e.errno == errno.EEXIST:
            return False
        raise

    # We own the claim. Write meta.json (best-effort — a race with a stale
    # sweep isn't possible because the sweep only touches dirs whose meta
    # references a dead jobid, which ours cannot until we die).
    meta = {
        "jobid": str(jobid),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "ts": time.time(),
    }
    with open(_meta_path(claims_dir, task_id), "w") as f:
        json.dump(meta, f)
    return True


def release_claim(claims_dir: Path, task_id: str) -> None:
    """Remove the claim dir so another worker can pick this task up.

    Called by the stale-claim sweep (never by a live claimant — a live
    worker that wants to stop mid-task keeps its claim so the successor
    knows to resume from state.pt rather than restart from scratch).
    """
    claim_dir = Path(claims_dir) / task_id
    try:
        # Remove meta.json first, then rmdir the empty dir.
        meta = claim_dir / CLAIM_META_FILENAME
        if meta.exists():
            meta.unlink()
        claim_dir.rmdir()
    except FileNotFoundError:
        pass  # already released — idempotent


def read_claim_meta(claims_dir: Path, task_id: str) -> Optional[dict]:
    """Return claim metadata dict, or None if not claimed / meta missing."""
    p = _meta_path(claims_dir, task_id)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        # Torn write during a claim attempt — treat as no meta; sweep may
        # release. Extremely unlikely because open()+write is buffered
        # and closed before another read, but be robust.
        return None


def list_claimed_tasks(claims_dir: Path) -> List[str]:
    """Return the task IDs currently claimed (i.e. claim dirs that exist)."""
    p = Path(claims_dir)
    if not p.exists():
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir())


def live_squeue_jobids() -> Set[str]:
    """Return the set of live SLURM jobids for the current user.

    Two-source lookup (2026-08-06 fix — squeue is not available inside the
    singularity container, so direct invocation always failed and sweep
    silently returned _SQUEUE_UNKNOWN forever, leaving dead-jobid claims
    locked in place):

      1. WF_LIVE_JOBIDS_FILE env var: path to a file written by the
         SLURM wrapper (which has squeue in PATH) at worker startup and
         refreshed every 5 min in a background process. One jobid per line.
         Preferred — bypasses the container's missing squeue.
      2. Direct `squeue` subprocess call — fallback for local tests where
         no wrapper is involved.

    If both fail, return _SQUEUE_UNKNOWN (sweep spares all claims — safe).
    """
    # 1. Env-var-pointed file (production path via worker.sh)
    jobids_file = os.environ.get("WF_LIVE_JOBIDS_FILE")
    if jobids_file and os.path.exists(jobids_file):
        try:
            # Staleness guard: file mtime older than 15 min → the wrapper's
            # background refresher died or is hung. Degrade to sentinel
            # (release nothing) rather than trust a stale snapshot.
            # 15 min = 3× the 5-min refresh interval — comfortable margin
            # for occasional load spikes without false alarms.
            STALE_SECS = 15 * 60
            age = time.time() - os.path.getmtime(jobids_file)
            if age > STALE_SECS:
                return _SQUEUE_UNKNOWN
            with open(jobids_file) as f:
                out = f.read()
            jobids = {line.strip() for line in out.splitlines() if line.strip()}
            if jobids:
                return jobids
            # empty file — squeue returned nothing? Unlikely on a live cluster.
            # Treat as "unknown" rather than "queue is empty" to avoid mass release.
            return _SQUEUE_UNKNOWN
        except OSError:
            pass  # fall through to direct-squeue

    # 2. Direct squeue call (works on head node / local dev)
    try:
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
        if not user:
            return _SQUEUE_UNKNOWN
        out = subprocess.check_output(
            ["squeue", "-h", "-u", user, "-o", "%A"],
            stderr=subprocess.DEVNULL, timeout=30,
        )
        jobids = {line.strip() for line in out.decode().splitlines() if line.strip()}
        return jobids if jobids else _SQUEUE_UNKNOWN
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        return _SQUEUE_UNKNOWN


class _SqueueUnknown(frozenset):
    """Sentinel: squeue result unavailable — do not perform destructive sweep."""


_SQUEUE_UNKNOWN = _SqueueUnknown()


def stale_claim_sweep(
    claims_dir: Path,
    live_jobids: Set[str],
) -> List[str]:
    """Release claims whose jobid is not in `live_jobids`.

    Returns the list of task IDs whose claim was released. Skips claims
    with unreadable meta.json (they're either being written now or torn —
    let the next sweep pick them up).

    If `live_jobids` is the _SQUEUE_UNKNOWN sentinel, does nothing and
    returns []: refusing to release claims when we can't verify liveness
    is safer than false-positive releases.
    """
    if live_jobids is _SQUEUE_UNKNOWN:
        return []
    released: List[str] = []
    for task_id in list_claimed_tasks(claims_dir):
        meta = read_claim_meta(claims_dir, task_id)
        if meta is None:
            continue
        jobid = str(meta.get("jobid", ""))
        if jobid and jobid not in live_jobids:
            release_claim(claims_dir, task_id)
            released.append(task_id)
    return released


def next_incomplete_task(
    manifest: Sequence[str],
    claims_dir: Path,
    runs_dir: Path,
    my_jobid: str,
    live_jobids: Optional[Set[str]] = None,
) -> Optional[str]:
    """Return the next task this worker should claim, or None if drained.

    Walks the manifest in order (round-robin-by-seed per §2) and returns
    the first task that is:
      - not complete (no summary.json)
      - not FAILED_DIVERGED
      - not currently claimed (mkdir-claim succeeds)

    Runs a stale-claim sweep first so previously-crashed workers' tasks
    are re-available. `live_jobids=None` triggers a fresh squeue call.
    """
    from orchestration.manifest import is_complete, is_failed_diverged

    if live_jobids is None:
        live_jobids = live_squeue_jobids()
    stale_claim_sweep(claims_dir, live_jobids)

    runs_dir = Path(runs_dir)
    Path(claims_dir).mkdir(parents=True, exist_ok=True)
    for task_id in manifest:
        if is_complete(runs_dir, task_id):
            continue
        if is_failed_diverged(runs_dir, task_id):
            continue
        if try_claim(claims_dir, task_id, my_jobid):
            return task_id
    return None
