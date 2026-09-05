#!/bin/bash
# HPC preflight — head-node-safe verification of the runtime environment
# BEFORE submitting HPO jobs. Import-only + filesystem-only checks.
#
# EXPLICITLY head-node-safe: no model construction, no forward passes, no
# training, no torch.zeros of any meaningful size, no Optuna study lifecycle.
# The heaviest thing this does is `import torch` and check `torch.tensor([1.0])
# .numpy()` — validates the exact class of failure that killed our first HPO
# submission (`ModuleNotFoundError`, `_ARRAY_API`) without doing any training-
# like compute on the head node.
#
# What this verifies:
#   1. Container exists + is executable via singularity
#   2. All required Python modules import inside the container with our PYPKGS
#   3. Our repo code imports cleanly (Phase 1 + Phase 2 wiring both present)
#   4. numpy/torch versions don't have the ABI mismatch that broke the earlier attempt
#   5. Cache files exist (all Phase-1 + STGODE DTW matrices)
#   6. Study DB directory (~/hpo/) is writeable
#
# What this deliberately does NOT verify (both need a compute node):
#   - CUDA kernel execution — known-good on L40S
#   - Actual model construction (deferred to the sbatch's first trial)
#   - Actual training loop
#
# Rationale: 95%+ of the failures we've seen this session were at import or
# module-load time. Those we catch cheaply. Model-arch failures we can only
# catch on-device — that's the sbatch's job.
#
# Usage on HPC head node, from repo root:
#   bash scripts/hpc_preflight.sh
#
# Exit code 0 = safe to submit. Non-zero = fix the reported failure first.

set -u
REPO="${REPO:-/scratch/jrvpra001/weatherflow/WeatherFLOW}"
CONTAINER="${CONTAINER:-/scratch/jrvpra001/weatherflow/container.sif}"
PYPKGS="${PYPKGS:-/scratch/jrvpra001/pypkgs}"

echo "=== HPC preflight (head-node-safe) ==="
echo "repo:      $REPO"
echo "container: $CONTAINER"
echo "pypkgs:    $PYPKGS"
echo

[ -f "$CONTAINER" ] || { echo "[FAIL] container not found: $CONTAINER" >&2; exit 1; }
echo "[ok] container exists"
[ -d "$PYPKGS" ]    || { echo "[FAIL] pypkgs dir not found: $PYPKGS" >&2; exit 1; }
echo "[ok] pypkgs dir exists"

singularity exec --bind /scratch/jrvpra001:/scratch/jrvpra001 --bind "${HOME}:${HOME}" \
    --env PYTHONPATH="$PYPKGS" \
    "$CONTAINER" \
    python3 - <<PYEOF
"""Import + FS checks only. NO model construction. NO forward pass. NO training."""
import sys, os
from pathlib import Path

REPO = "${REPO}"
os.chdir(REPO)
sys.path.insert(0, REPO)

# --- core third-party imports ---
try:
    import numpy, torch, yaml, pandas, optuna
    print(f"[ok] core imports:  numpy={numpy.__version__}  torch={torch.__version__}  "
          f"optuna={optuna.__version__}  yaml={yaml.__version__}  pandas={pandas.__version__}")
except Exception as e:
    print(f"[FAIL] core imports: {type(e).__name__}: {e}"); sys.exit(2)

if not numpy.__version__.startswith("1."):
    print(f"[FAIL] numpy is {numpy.__version__} — container's C-compiled modules need numpy 1.x. "
          "Purge pypkgs numpy so the container's numpy 1.x wins."); sys.exit(2)

# ABI check — smallest possible sanity that torch's numpy interop isn't broken
try:
    x = torch.tensor([1.0])
    _ = x.numpy()
    print(f"[ok] torch↔numpy ABI clean (tensor.numpy() works)")
except Exception as e:
    print(f"[FAIL] torch↔numpy ABI broken (likely numpy version conflict): {e}"); sys.exit(2)

# --- repo code imports (module-load only; no construction) ---
try:
    from scripts.hpo_run_optuna import SEARCH_SPACES, TRIAL_SEED
    from cli import load_config, load_adapter, ADAPTER_REGISTRY
    from preprocessing.pipeline import load_cache, run_pipeline, compute_data_hash
    from preprocessing.labels_ch import (
        DAY_WIN, HOUR_WIN, apply_ch_labels, compute_ch_thresholds,
        label_all_splits_ch,
    )
    from training.trainer import Trainer, TrainingDiverged
    from training.seed import seed_everything
    from training.losses import mae_loss
    from evaluation.regression import ERA5_VAR_NAMES
    from orchestration.manifest import parse_task_id, is_classification_task, \
                                         CLASSIFICATION_TOKEN, build_manifest, \
                                         build_manifest_classification, \
                                         build_manifest_combined, \
                                         PHASE1_MODEL_ORDER
    print(f"[ok] repo imports:  hpo_run_optuna, cli, pipeline, trainer, adapters, "
          f"labels_ch (day_win={DAY_WIN} hour_win={HOUR_WIN}), "
          f"orchestration.manifest (Phase-1 + Phase-2 + combined)")
    print(f"[ok] SEARCH_SPACES models = {list(SEARCH_SPACES)}")
    print(f"[ok] adapter registry has = {sorted(ADAPTER_REGISTRY.keys())}")
except Exception as e:
    print(f"[FAIL] repo imports: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc(); sys.exit(2)

# --- adapter class attribute check (no construction — just verify the class loads) ---
for name in ("mtgode", "stgode"):
    try:
        cls = load_adapter(name)
        assert hasattr(cls, "build_model") and hasattr(cls, "forward") and hasattr(cls, "predict"), \
            f"adapter {name} missing required BaseAdapter methods"
        print(f"[ok] adapter class '{name}' loads: {cls.__module__}.{cls.__name__}")
    except Exception as e:
        print(f"[FAIL] adapter class '{name}': {type(e).__name__}: {e}"); sys.exit(2)

# --- config yaml loads (data only; no adapter build) ---
for name in ("mtgode", "stgode"):
    try:
        cfg = load_config(name, {})
        assert "data" in cfg and "model" in cfg and "training" in cfg, \
            f"{name}.yaml missing standard sections"
        print(f"[ok] config '{name}.yaml' loads: input_len={cfg['data'].get('input_len')} "
              f"horizon={cfg['data'].get('horizon')}")
    except Exception as e:
        print(f"[FAIL] config '{name}': {type(e).__name__}: {e}"); sys.exit(2)

# --- cache files exist ---
cache_dir = Path("preprocessing/cache/saws")
required = ["X_train.npy", "X_val.npy", "X_test.npy", "adj.npy", "coords.npy",
            "mean.npy", "std.npy", "thresholds.npy", "meta.json"]
missing = [f for f in required if not (cache_dir / f).exists()]
if missing: print(f"[FAIL] cache missing: {missing}"); sys.exit(2)
print(f"[ok] cache: {len(required)} files present")

dtw = ["t2m", "sp", "rh", "ws"]
missing_dtw = [t for t in dtw if not (cache_dir / f"dtw_matrix_{t}.npy").exists()]
if missing_dtw: print(f"[FAIL] STGODE DTW matrices missing for: {missing_dtw}"); sys.exit(2)
print(f"[ok] STGODE DTW matrices present for {dtw}")

# --- Phase-2 calendar-hour scheme artefacts ---
import json as _json_early                                            # local alias — the top-level `import json` happens later
ch_required = ["thresholds_ch.npy", "y_train_ch.npy", "y_val_ch.npy",
               "y_test_ch.npy", "tp_wet_trigger_ch.npy", "labels_ch_meta.json"]
missing_ch = [f for f in ch_required if not (cache_dir / f).exists()]
if missing_ch: print(f"[FAIL] calendar-hour cache artefacts missing: {missing_ch}"); sys.exit(2)
with open(cache_dir / "labels_ch_meta.json") as f: ch_meta = _json_early.load(f)
print(f"[ok] Phase-2 CH cache: {len(ch_required)} artefacts present, "
      f"window=(day={ch_meta.get('day_win')}, hour={ch_meta.get('hour_win')}), "
      f"scheme={ch_meta.get('scheme')} v{ch_meta.get('scheme_version')}")
if ch_meta.get('day_win') != 7 or ch_meta.get('hour_win') != 1:
    print(f"[FAIL] CH cache built with wrong window (expected day=7, hour=1)"); sys.exit(2)

# --- combined manifest present ---
mpath = Path("runs/manifest_all.json")
if not mpath.exists(): print(f"[FAIL] {mpath} missing"); sys.exit(2)
with open(mpath) as f: manifest = _json_early.load(f)
if len(manifest) != 190: print(f"[FAIL] manifest_all.json has {len(manifest)} tasks, expected 190"); sys.exit(2)
print(f"[ok] runs/manifest_all.json: {len(manifest)} tasks (160 reg + 30 cls, interleaved-by-seed)")

# --- cache meta.json parseable ---
import json
with open(cache_dir / "meta.json") as f: meta = json.load(f)
print(f"[ok] cache meta.json: n_nodes={meta.get('n_nodes')} var_names={meta.get('var_names')}")

# --- study DB dir writeable (no actual DB creation) ---
hpo_home = Path.home() / "hpo"
hpo_home.mkdir(parents=True, exist_ok=True)
probe = hpo_home / f".preflight_write_test_{os.getpid()}"
try:
    probe.write_text("ok"); probe.unlink()
    print(f"[ok] {hpo_home} is writeable")
except Exception as e:
    print(f"[FAIL] {hpo_home} not writeable: {e}"); sys.exit(2)

# --- sbatch wrapper syntax (bash -n) ---
import subprocess
r = subprocess.run(["bash", "-n", "slurm/hpo_optuna_worker.sh"],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(f"[FAIL] slurm/hpo_optuna_worker.sh syntax: {r.stderr}"); sys.exit(2)
print(f"[ok] slurm/hpo_optuna_worker.sh syntax OK")

print("\n[PASS] preflight complete — safe to submit HPO jobs.")
PYEOF

rc=$?
if [ "$rc" != "0" ]; then
    echo
    echo "[FAIL] preflight exit $rc — DO NOT submit HPO jobs until this passes."
    exit "$rc"
fi
