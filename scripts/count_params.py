"""
Patch n_params, n_train_samples, n_val_samples, n_test_samples into every
runs/{task_id}/summary.json.

No inference, no checkpoints, no GPU.  Two minutes to patch all 190 runs.

n_params:   adapter.get_param_count() after build_model — architecture only.
n_*_samples: len(STGNNDataset(X_{split})) — same windowing used at training time.

Usage:
    python scripts/count_params.py [--dry-run] [--runs-dir runs]
"""

import argparse
import json
import sys
import yaml
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ADAPTERS = {
    "agcrn":  "models.baselines.agcrn_adapter.AGCRNAdapter",
    "clcrn":  "models.baselines.clcrn_adapter.CLCRNAdapter",
    "mtgode": "models.continuous_time.mtgode_adapter.MTGODEAdapter",
    "stgode": "models.continuous_time.stgode_adapter.STGODEAdapter",
}


def _load_adapter_cls(model_name: str):
    spec = ADAPTERS[model_name]
    mod_path, cls_name = spec.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def _get_config(model_name: str, cached_meta: dict) -> dict:
    cfg_path = ROOT / "experiments" / "configs" / f"{model_name}.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    run_meta = dict(cached_meta["meta"])
    run_meta.update({
        "coords":            cached_meta["coords"],
        "mean":              cached_meta["mean"],
        "std":               cached_meta["std"],
        "thresholds":        cached_meta["thresholds"],
        "static_thresholds": cached_meta["thresholds"],
    })
    cfg["meta"] = run_meta
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--runs-dir",  default="runs")
    args = parser.parse_args()

    from preprocessing.pipeline import load_cache
    from preprocessing.dataset import STGNNDataset

    print("Loading cache …")
    cached = load_cache("saws")

    # Dataset window counts — constant across all tasks and models
    # (input_len=12, horizon=48 fixed for all campaign runs)
    meta_json   = cached["meta"]
    input_len   = meta_json.get("input_len", 12)
    horizon     = meta_json.get("horizon",   48)

    n_samples = {}
    for split in ("train", "val", "test"):
        X  = cached[f"X_{split}"]
        ds = STGNNDataset(X, labels=None, input_len=input_len, horizon=horizon,
                          task="regression", target_variable_index=0)
        n_samples[f"n_{split}_samples"] = len(ds)
    print(f"Dataset windows: {n_samples}")

    # Build each model once (config-only) to get param count
    import torch
    # STGODE vendors 'model.py' in its own subdir; add that dir for its import to resolve
    STGODE_VENDOR = str(ROOT / "models" / "continuous_time" / "STGODE")
    param_counts: dict[str, int] = {}
    for model_name in ADAPTERS:
        try:
            if model_name == "stgode":
                if STGODE_VENDOR not in sys.path:
                    sys.path.insert(0, STGODE_VENDOR)
                # Remove any stale 'model' from sys.modules (different vendor package)
                sys.modules.pop("model", None)
            cfg = _get_config(model_name, cached)
            cls = _load_adapter_cls(model_name)
            adapter = cls()
            adapter.build_model(cfg, cached["adj"])
            n = adapter.get_param_count()
            param_counts[model_name] = n
            print(f"  {model_name:8s}  trainable params = {n:>10,}")
        except Exception as e:
            print(f"  {model_name:8s}  ERROR: {e}")

    # Save standalone reference file
    counts_out = ROOT / "results" / "PARAM_COUNTS.json"
    counts_out.parent.mkdir(exist_ok=True)
    if not args.dry_run:
        with open(counts_out, "w") as f:
            json.dump({"param_counts": param_counts, "dataset_windows": n_samples}, f, indent=2)
        print(f"\nWritten: {counts_out}")

    # Patch every run summary.json
    runs_dir = ROOT / args.runs_dir
    patched = skipped = errors = 0

    for run_dir in sorted(runs_dir.iterdir()):
        sj = run_dir / "summary.json"
        if not sj.exists():
            continue
        try:
            s = json.load(open(sj))
        except Exception as e:
            print(f"  ERROR reading {run_dir.name}: {e}")
            errors += 1
            continue

        model_name = s.get("model")
        if model_name not in param_counts:
            continue  # not a campaign model (e.g. resmoke dirs)

        already = (
            "n_params"         in s and
            "n_train_samples"  in s and
            "n_val_samples"    in s and
            "n_test_samples"   in s
        )
        if already:
            skipped += 1
            continue

        s["n_params"]        = param_counts[model_name]
        s["n_train_samples"] = n_samples["n_train_samples"]
        s["n_val_samples"]   = n_samples["n_val_samples"]
        s["n_test_samples"]  = n_samples["n_test_samples"]

        if not args.dry_run:
            with open(sj, "w") as f:
                json.dump(s, f, indent=2)
        patched += 1

    print(f"\nPatched: {patched}  Already done: {skipped}  Errors: {errors}")
    if args.dry_run:
        print("(dry-run — nothing written)")


if __name__ == "__main__":
    main()
