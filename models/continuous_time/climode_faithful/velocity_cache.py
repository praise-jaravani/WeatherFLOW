"""
Velocity cache — precompute-and-cache the Eq. 10 fit per (split, window_index).

Per V1 in docs/REPORT.md, the Verma et al. Eq. 10 fit is DATA-ONLY: no
gradients from main training flow into it, no model parameters are shared.
That means for any fixed window set, the fit can be computed once and
cached — mirroring upstream's own `fit_velocity → .npy` design, transplanted
from their (year, day-of-year) grid to our sliding-window grid.

Key: (split_tag, window_start_index).  Value: (2K, H, W) velocity tensor.

**Deterministic seeding**: each per-window Adam fit is seeded from
`hash(f"{split_tag}:{window_start_index}") & 0xFFFF_FFFF` so re-running
produces bit-identical caches (up to CUDA nondeterminism at fit time,
which is why fits should be done on CPU or with deterministic algorithms
flagged if reproducibility down to the last bit is required).

Cache lives under `preprocessing/cache/climode_faithful/velocities/`
so it survives across model runs and is shared across seeds. One cache
serves all 10 seeds because the estimator is deterministic given
(data, seed).

Fallback: if the cache is missing for a requested window, fit-and-store
on-demand. This is how the HPC smoke can run uncached by design (see
§4 of the 2026-07-28 review) — the smoke does not warm the cache; the
first real training run does.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .initial_velocity import fit_initial_velocity


CACHE_ROOT = Path("preprocessing/cache/climode_faithful/velocities")


def _cache_path(split_tag: str, window_index: int) -> Path:
    """Resolve the on-disk .npy path for one cache entry."""
    subdir = CACHE_ROOT / split_tag
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"win_{window_index:07d}.npy"


def _window_seed(split_tag: str, window_index: int) -> int:
    """Deterministic per-window seed for the Adam fit's parameter init.

    Uses SHA-256 truncated to 32 bits to avoid Python's hash randomization
    (PYTHONHASHSEED is unseeded by default across processes).
    """
    key = f"{split_tag}:{window_index}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF   # positive int32


def load_or_fit_velocity(
    split_tag: str,
    window_index: int,
    past_frames: torch.Tensor,
    past_time_hours: torch.Tensor,
    kernel_inv: torch.Tensor,
    steps: int = 200,
    lr: float = 2.0,
    prior_weight: float = 1e-7,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    Return the initial-velocity tensor for one window, using the cache
    when possible.

    Args:
        split_tag: "train" / "val" / "test" / "smoke" — namespaces the cache
        window_index: sliding-window start index within the split
        past_frames: (B, T_past, K, H, W) input window — B must be 1
            (one fit per window per this API; batch at the caller level
            by concatenating over batch dim outside)
        past_time_hours: (T_past,) hour offsets
        kernel_inv: (H*W, H*W) precomputed RBF K^{-1}
        steps, lr, prior_weight: mirror upstream `optimize_vel` defaults
        use_cache: if False, always fit fresh (used by the smoke)

    Returns:
        (2*K, H, W) velocity tensor (v_x concatenated with v_y).
    """
    assert past_frames.shape[0] == 1, (
        f"load_or_fit_velocity expects one window at a time; got B={past_frames.shape[0]}. "
        "Call it in a loop over the batch dim if you need multiple."
    )
    path = _cache_path(split_tag, window_index)

    if use_cache and path.exists():
        arr = np.load(path)
        v = torch.from_numpy(arr).to(past_frames.device, past_frames.dtype)
        assert v.dim() == 3, f"cache entry shape {v.shape} — expected (2K, H, W)"
        return v

    # Deterministic seed for the Adam fit
    seed = _window_seed(split_tag, window_index)
    with _torch_seed(seed):
        v = fit_initial_velocity(
            past_frames=past_frames,
            past_time_hours=past_time_hours,
            kernel_inv=kernel_inv,
            steps=steps,
            lr=lr,
            prior_weight=prior_weight,
        )                                        # (1, 2K, H, W)
    v = v.squeeze(0)                             # → (2K, H, W)

    if use_cache:
        # Save as float32 numpy for stability across runs
        np.save(path, v.detach().cpu().float().numpy())

    return v


class _torch_seed:
    """Context manager: seed torch for the duration, restore state on exit.

    Preserves the caller's RNG state so the cache-warming side effect
    doesn't perturb outer randomness (main-model init, dataloader shuffle).
    """
    def __init__(self, seed: int):
        self.seed = seed

    def __enter__(self):
        self._cpu_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self._cuda_state = torch.cuda.get_rng_state()
        torch.manual_seed(self.seed)
        return self

    def __exit__(self, *exc):
        torch.set_rng_state(self._cpu_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(self._cuda_state)
        return False


def cache_stats(split_tag: Optional[str] = None) -> dict:
    """Report cache coverage (for logging + REPORT.md)."""
    if not CACHE_ROOT.exists():
        return {"exists": False, "split_counts": {}}
    counts = {}
    total_mb = 0.0
    for sub in CACHE_ROOT.iterdir():
        if not sub.is_dir():
            continue
        if split_tag and sub.name != split_tag:
            continue
        files = list(sub.glob("win_*.npy"))
        counts[sub.name] = len(files)
        total_mb += sum(f.stat().st_size for f in files) / (1024 * 1024)
    return {"exists": True, "split_counts": counts, "total_mb": total_mb}
