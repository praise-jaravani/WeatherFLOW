"""
Orchestrates the full preprocessing sequence with hash-based cache invalidation.

Pipeline order (leakage-safe):
  1. Parse   — saws_parser / era5_parser → raw DataFrames
  2. Impute  — IDW missing value imputation, 5% exclusion rule
  3. Split   — chronological 60/20/20 split (MUST precede label/normalise)
  4. Label   — 95th-percentile thresholds computed on train only
  5. Normalise — z-score stats computed on train only
  6. Graph   — adjacency matrix from station coordinates
  7. DataLoaders — sliding-window Dataset + DataLoader factory

Hash check: before running, a SHA-256 fingerprint of all raw data files and
the preprocessing config is compared against preprocessing/cache/preprocessing_hash.json.
If they match, cached .npz files are loaded and returned immediately. If they
differ (or no cache exists), the full pipeline runs, results are saved to
preprocessing/cache/, and the hash file is updated.
"""

import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader


CACHE_DIR = Path(__file__).parent / "cache"
HASH_FILE = CACHE_DIR / "preprocessing_hash.json"


def compute_data_hash(raw_data_dirs: list, config: Dict[str, Any]) -> str:
    """Compute a SHA-256 fingerprint of raw data file mtimes + config dict."""
    raise NotImplementedError


def load_cache() -> Dict[str, np.ndarray]:
    """Load preprocessed arrays from .npz cache files."""
    raise NotImplementedError


def save_cache(arrays: Dict[str, np.ndarray], data_hash: str) -> None:
    """Save preprocessed arrays to .npz files and write the hash file."""
    raise NotImplementedError


def run_pipeline(
    config: Dict[str, Any], force_rerun: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, Dict[str, Any]]:
    """
    Entry point called by cli.py. Returns (train_loader, val_loader,
    test_loader, adjacency_matrix, preprocessing_metadata).

    preprocessing_metadata includes normalisation stats and label thresholds
    needed for inverse-normalisation and evaluation.

    Args:
        config: merged config dict from YAML + CLI overrides.
        force_rerun: skip hash check and always rerun (--no-cache flag).
    """
    raise NotImplementedError
