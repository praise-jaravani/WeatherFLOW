"""
Adjacency-consumption probe (Deliverable B of specs/Adjacency_Audit_Spec.md).

For each model in {agcrn, mtgode, clcrn}, empirically determine whether it
consumes the static adjacency matrix supplied by preprocessing.graph, by
substituting variants of A and measuring output changes on a fixed test batch.

Probes (spec §4.1):
  P1: real A, run twice → nondeterminism floor
  P2: identity A (row-normalised) → call-time consumption
  P3: seeded random row-normalised A → call-time, weight-sensitivity
  P4: row-permuted real A (structure preserved but shuffled) → subtle use
  P5: rebuild adapter with identity A, then forward with real x → construction-time
  P7: state_dict + buffer inspection for (N, N) shaped tensors → weights baked-in

CLCRN-specific P6 (rebuild with jittered coords) omitted here because it
requires the CLCRN model to accept a `coords` kwarg through the adapter build
path — see specs/ADJACENCY_AUDIT.md §5 for the manual coord-source check.

Adapted from the spec's ERA5-N=999 formulation to SAWS-N=25. Uses untrained
models with a fixed random seed for deterministic forward passes — the question
is architectural (does the adapter read A?), not weight-dependent. Runs
locally in <1 min per model on CPU or GPU.

Tolerance: max |Δ| ≤ 5×10⁻⁵ in output space, matching the 964078 CPU/GPU
equivalence precedent (2.23×10⁻⁵). Borderline results (10× tolerance) get
a float64 CPU re-check flag.

Output: outputs/adjacency_probe/{model}_probe.json + probe_summary.md
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("probe")

TOLERANCE = 5e-5
BORDERLINE_MULT = 10.0

ADAPTER_REGISTRY = {
    "agcrn":  "models.baselines.agcrn_adapter.AGCRNAdapter",
    "mtgode": "models.continuous_time.mtgode_adapter.MTGODEAdapter",
    "clcrn":  "models.baselines.clcrn_adapter.CLCRNAdapter",
}

# Default probe order: CLCRN first. AGCRN's adapter pollutes sys.modules['model']
# to point at its own vendored code, hiding CLCRN's `model/clcnn/`. Running
# CLCRN first sidesteps the collision.
DEFAULT_PROBE_ORDER = ["clcrn", "mtgode", "agcrn"]


_INITIAL_SYS_PATH = None


def _snapshot_sys_path():
    """Capture the base sys.path before any adapter pollutes it."""
    global _INITIAL_SYS_PATH
    if _INITIAL_SYS_PATH is None:
        _INITIAL_SYS_PATH = list(sys.path)


def _purge_shared_model_module():
    """AGCRN, CLCRN and MTGODE each vendor an upstream repo under a bare
    'model' package name and rely on `sys.path` injection to import it.
    Between probes we must both:
      1. Delete `sys.modules['model']` (and submodules) so the next adapter
         reloads its own vendored copy instead of the previous vendor's;
      2. Restore `sys.path` to its pre-probe snapshot so imports resolve
         from the CURRENT adapter's repo path, not a previous one that
         still happens to sit earlier in the path.
    MTGODE uses a namespaced sys.modules key already and isn't itself
    affected, but its REPO_PATH still lands in sys.path and can interfere
    with adapters run after it.
    """
    # 1. Purge shared 'model' module and its submodules from cache
    for key in list(sys.modules.keys()):
        if key == "model" or key.startswith("model."):
            del sys.modules[key]
    # 2. Restore sys.path to the initial (pre-adapter) snapshot
    if _INITIAL_SYS_PATH is not None:
        sys.path[:] = list(_INITIAL_SYS_PATH)


def _load_adapter(model_name: str):
    """Instantiate the adapter class."""
    dotted = ADAPTER_REGISTRY[model_name]
    mod_path, cls_name = dotted.rsplit(".", 1)
    module = importlib.import_module(mod_path)
    return getattr(module, cls_name)()


def _load_model_config(model_name: str) -> Dict[str, Any]:
    """Load experiments/configs/<model>.yaml minus training bits."""
    import yaml
    path = Path("experiments/configs") / f"{model_name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _fresh_adapter(model_name: str, adj: torch.Tensor, coords_np: np.ndarray,
                   input_len: int, horizon: int, n_vars: int, seed: int = 42):
    """Build a fresh adapter+model with reproducible weights.

    `adj` is passed to build_model (its shape and values may or may not be
    consumed by the adapter — that's what the probe measures).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    adapter = _load_adapter(model_name)
    config = _load_model_config(model_name)
    config["dataset"] = "saws"
    config["task"] = "regression"
    config["meta"] = {
        "n_vars":  n_vars,
        "coords":  coords_np,  # CLCRN needs [lat, lon]
    }
    config.setdefault("data", {})
    config["data"]["input_len"] = input_len
    config["data"]["horizon"] = horizon

    adapter.build_model(config, adj)
    return adapter


def _make_probe_A(kind: str, real_A: torch.Tensor, seed: int = 1337) -> torch.Tensor:
    """Construct a substituted adjacency matrix per probe kind."""
    N = real_A.shape[0]
    if kind == "real":
        return real_A.clone()
    if kind == "identity":
        A = torch.eye(N, dtype=real_A.dtype, device=real_A.device)
        # row-normalised identity is still identity (each row sums to 1)
        return A
    if kind == "random":
        g = torch.Generator(device=real_A.device).manual_seed(seed)
        A = torch.rand(N, N, generator=g, dtype=real_A.dtype, device=real_A.device)
        A.fill_diagonal_(0.0)
        # row-normalise
        row_sums = A.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return A / row_sums
    if kind == "permuted":
        g = torch.Generator(device=real_A.device).manual_seed(seed)
        perm = torch.randperm(N, generator=g, device=real_A.device)
        return real_A[perm][:, perm]
    raise ValueError(f"unknown probe kind: {kind}")


@torch.no_grad()
def _forward_pass(adapter, x: torch.Tensor, adj: torch.Tensor,
                  rng_seed: int = 42) -> torch.Tensor:
    """Deterministic evaluation forward pass. Reseeds torch's global RNG on
    every call because some models (notably MTGODE's adaptive-graph builder)
    call torch.rand/randn during forward, giving different outputs across
    consecutive calls if the RNG isn't reset."""
    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)
    adapter.model.eval()
    return adapter.predict(x, adj)


def _delta(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    """Max |a-b| and mean |a-b|."""
    diff = (a - b).abs()
    return float(diff.max().item()), float(diff.mean().item())


def _run_probes_for_model(
    model_name: str, real_A: torch.Tensor, coords_np: np.ndarray,
    x: torch.Tensor, input_len: int, horizon: int, n_vars: int, seed: int,
) -> Dict[str, Any]:
    """Execute P1-P7 for one model. Returns a dict of per-probe results."""
    log.info(f"=== {model_name.upper()} probe ===")
    device = x.device

    results: Dict[str, Any] = {"model": model_name, "n_nodes": real_A.shape[0]}

    # ---- P1: nondeterminism floor (same adapter, real A run twice) ----
    # Must measure same-adapter-twice, not fresh-then-fresh — the latter is 0.0
    # by construction (seeded init), which hides forward-call nondeterminism
    # (e.g. MTGODE's adaptive-graph builder calls torch.rand without a seeded
    # generator). The _forward_pass helper reseeds torch RNG before each call.
    adapter_a = _fresh_adapter(model_name, real_A, coords_np, input_len, horizon, n_vars, seed=seed)
    adapter_a.model.to(device)
    ref_out = _forward_pass(adapter_a, x, real_A)
    rerun_out = _forward_pass(adapter_a, x, real_A)

    p1_max, p1_mean = _delta(ref_out, rerun_out)
    log.info(f"  P1 nondeterminism floor:  max={p1_max:.2e}  mean={p1_mean:.2e}")
    results["P1_nondeterminism_floor"] = {"max_abs_delta": p1_max, "mean_abs_delta": p1_mean}

    if p1_max > TOLERANCE:
        log.warning(f"  ⚠ P1 exceeds tolerance {TOLERANCE:.0e} — determinism must be fixed first")

    # ---- P2, P3, P4: call-time A substitution ----
    for kind, seed_A in [("identity", None), ("random", 1337), ("permuted", 1337)]:
        A_sub = _make_probe_A(kind, real_A, seed=seed_A or 0)
        sub_out = _forward_pass(adapter_a, x, A_sub)
        max_d, mean_d = _delta(ref_out, sub_out)
        verdict = "CONSUMED" if max_d > TOLERANCE else "IGNORED"
        borderline = TOLERANCE < max_d <= TOLERANCE * BORDERLINE_MULT
        log.info(f"  P2/3/4 [{kind:9s}]:      max={max_d:.2e}  mean={mean_d:.2e}  → {verdict}"
                 + ("  (borderline)" if borderline else ""))
        results[f"P_{kind}"] = {
            "max_abs_delta": max_d, "mean_abs_delta": mean_d,
            "verdict": verdict, "borderline": borderline,
        }

    # ---- P5: construction-time A substitution (rebuild with identity, forward with x) ----
    identity_A = torch.eye(real_A.shape[0], dtype=real_A.dtype, device=real_A.device)
    adapter_p5 = _fresh_adapter(model_name, identity_A, coords_np, input_len, horizon, n_vars, seed=seed)
    adapter_p5.model.to(device)
    p5_out = _forward_pass(adapter_p5, x, real_A)   # use REAL A at forward time to isolate build-time effect
    max_d, mean_d = _delta(ref_out, p5_out)
    verdict = "CONSUMED" if max_d > TOLERANCE else "IGNORED"
    borderline = TOLERANCE < max_d <= TOLERANCE * BORDERLINE_MULT
    log.info(f"  P5 [construct with identity]: max={max_d:.2e}  mean={mean_d:.2e}  → {verdict}"
             + ("  (borderline)" if borderline else ""))
    results["P5_construction_time"] = {
        "max_abs_delta": max_d, "mean_abs_delta": mean_d,
        "verdict": verdict, "borderline": borderline,
    }

    # ---- P7: state_dict + buffer inspection for (N, N) tensors ----
    N = real_A.shape[0]
    buffers_matching = []
    for name, buf in adapter_a.model.named_buffers():
        if buf.dim() == 2 and buf.shape[0] == N and buf.shape[1] == N:
            buffers_matching.append({"name": name, "shape": list(buf.shape)})
    params_matching = []
    for name, param in adapter_a.model.named_parameters():
        if param.dim() == 2 and param.shape[0] == N and param.shape[1] == N:
            params_matching.append({"name": name, "shape": list(param.shape)})
    log.info(f"  P7 (N,N) buffers: {len(buffers_matching)}  parameters: {len(params_matching)}")
    results["P7_nn_tensor_scan"] = {
        "buffers": buffers_matching, "parameters": params_matching,
    }

    # ---- Overall verdict ----
    consumed = any(results[k]["verdict"] == "CONSUMED"
                   for k in ("P_identity", "P_random", "P_permuted", "P5_construction_time"))
    overall = "CONSUMED" if consumed else "IGNORED"
    results["verdict"] = overall
    results["consumed_by"] = [k for k in ("P_identity", "P_random", "P_permuted", "P5_construction_time")
                              if results[k]["verdict"] == "CONSUMED"]
    log.info(f"  → {model_name.upper()}: {overall}")
    if consumed:
        log.info(f"    triggered by: {', '.join(results['consumed_by'])}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Adjacency-consumption probe (spec §4)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for the probe. 4 is enough at N=25 for CPU.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Weight-init seed. Fixed so probes are reproducible.")
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--models", nargs="+", default=DEFAULT_PROBE_ORDER)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="outputs/adjacency_probe")
    args = parser.parse_args()

    import os
    os.chdir("/home/praise/Computer_Science_Resources/Weatherflow")
    sys.path.insert(0, "/home/praise/Computer_Science_Resources/Weatherflow")
    _snapshot_sys_path()

    # Load SAWS cache: coords + real adjacency + one real batch of features
    cache = Path("preprocessing/cache/saws")
    adj_np = np.load(cache / "adj.npy")
    coords_np = np.load(cache / "coords.npy")   # (N, 2) [lat, lon]
    X_test = np.load(cache / "X_test.npy")

    N, D = adj_np.shape[0], X_test.shape[-1]
    log.info(f"SAWS cache: N={N}, D={D}, X_test.shape={X_test.shape}")

    device = torch.device(args.device)
    real_A = torch.from_numpy(adj_np).float().to(device)
    x_batch = torch.from_numpy(X_test[:args.batch_size * args.input_len]).float()
    # reshape to (B, T, N, D)
    x_batch = x_batch[:args.batch_size * args.input_len].view(args.batch_size, args.input_len, N, D).to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for model_name in args.models:
        _purge_shared_model_module()   # avoid AGCRN/CLCRN sys.modules['model'] collision
        try:
            res = _run_probes_for_model(
                model_name, real_A, coords_np, x_batch,
                args.input_len, args.horizon, D, args.seed,
            )
            (out_dir / f"{model_name}_probe.json").write_text(json.dumps(res, indent=2))
            all_results.append(res)
        except Exception as e:
            log.exception(f"{model_name} probe FAILED: {type(e).__name__}: {e}")
            all_results.append({"model": model_name, "error": f"{type(e).__name__}: {e}"})

    # Summary
    lines = ["# Adjacency Probe Summary\n",
             f"**Batch:** B={args.batch_size}, T={args.input_len}, N={N}, D={D}, seed={args.seed}",
             f"**Tolerance:** max |Δ| ≤ {TOLERANCE:.0e}",
             "",
             "| Model | P1 floor | P2 identity | P3 random | P4 permuted | P5 construct | Verdict |",
             "|---|---|---|---|---|---|---|"]
    for r in all_results:
        if "error" in r:
            lines.append(f"| {r['model']} | ERROR | — | — | — | — | ERROR: {r['error']} |")
            continue
        def cell(key):
            d = r.get(key, {})
            m = d.get("max_abs_delta", None)
            if m is None: return "—"
            v = d.get("verdict", "")
            tag = f" ({v})" if v else ""
            return f"{m:.2e}{tag}"
        lines.append(
            f"| {r['model']} | {cell('P1_nondeterminism_floor')} | "
            f"{cell('P_identity')} | {cell('P_random')} | {cell('P_permuted')} | "
            f"{cell('P5_construction_time')} | **{r['verdict']}** |"
        )
    (out_dir / "probe_summary.md").write_text("\n".join(lines))
    log.info(f"Wrote probe_summary.md to {out_dir}/")

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
