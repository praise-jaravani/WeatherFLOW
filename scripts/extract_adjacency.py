"""
Extract learned and constructed dependency structures for Section 4.6.

Weights only: no forward pass, no dataset, no GPU. Loads each checkpoint, asks the
model's OWN code path for its adjacency, and saves everything to one .npz.

    python scripts/extract_adjacency.py --probe          # inspect checkpoint keys, exit
    python scripts/extract_adjacency.py                  # t2m, seeds 42-46
    python scripts/extract_adjacency.py --targets t2m sp rh ws

WHY THE MODEL'S OWN CODE PATH
-----------------------------
The MTGODE audit (Section 3.3.3) found the adjoint backward pass had excluded the graph
learner from gradient flow, leaving the adjacency at its random initialisation. A
reimplementation of the published formula would not have caught that: it would have
produced a plausible-looking matrix from whatever the embeddings happened to be. So this
script calls the model's own graph-construction method wherever one exists, and falls
back to an explicit formula only when it must -- flagging loudly when it does.

OUTPUT
------
results/ADJACENCY_all.npz with keys:
    learned/{model}/{target}/s{seed}   (N, N) float32   AGCRN, MTGODE
    clcrn_knn                          (N, N) uint8     spherical k-NN mask, k=25
    stgode_spatial, stgode_semantic    (N, N) float32
    geometric                          (N, N) float32   cached["adj"], the pipeline reference
    station_names, lat, lon            (N,)
"""

import argparse, sys, warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEEDS = [42, 43, 44, 45, 46]
LEARNED = {
    "agcrn":  ("models.baselines.agcrn_adapter",        "AGCRNAdapter"),
    "mtgode": ("models.continuous_time.mtgode_adapter", "MTGODEAdapter"),
}
OUT = ROOT / "results" / "ADJACENCY_all.npz"


def load_state(path, device="cpu"):
    import torch
    st = torch.load(path, map_location=device, weights_only=True)
    if isinstance(st, dict) and "model_state_dict" in st:
        st = st["model_state_dict"]
    return st


def probe(cached):
    """Print checkpoint parameter names so the embedding tensors can be identified."""
    for model in LEARNED:
        p = ROOT / "runs" / f"{model}_t2m_s42" / "checkpoint.pt"
        print(f"\n{'='*70}\n{model}  {p}\n{'='*70}")
        if not p.exists():
            print("  MISSING"); continue
        for k, v in load_state(p).items():
            if hasattr(v, "shape") and v.ndim <= 2:
                print(f"  {k:60s} {tuple(v.shape)}")
    print(f"\n{'='*70}\ncache keys\n{'='*70}")
    for k, v in sorted(cached.items()):
        shape = getattr(v, "shape", f"{type(v).__name__}[{len(v)}]" if hasattr(v, "__len__") else "")
        print(f"  {k:28s} {shape}")


# ── learned adjacencies, via each model's own code ───────────────────────────

def adjacency_from_model(model_name, adapter, state):
    """
    Ask the built model for its adjacency. Tries, in order:
      1. a method or submodule the adapter or model exposes (including 'gc')
      2. the published formula on the loaded embeddings, with a warning
    Returns (A, how) where `how` records which route was taken -- this goes in the
    .npz so the paper can state how the matrices were obtained.

    Frozen-graph test (MTGODE): we call model.gc BEFORE loading the checkpoint
    (random-init weights) and again AFTER. If the matrices are identical the graph
    learner was never updated -- confirming the adjoint-mode gradient-exclusion
    concern from the Section 3.3.3 audit. The result is printed; it does not change
    the returned matrix (which is always the post-load value).
    """
    import torch
    m = adapter.model
    N = m_nodes(m)
    idx = torch.arange(N)

    # frozen-graph test: snapshot random-init adjacency before loading checkpoint.
    # The state dict key prefix net.gc.* tells us gc lives on m.net (the inner MTGODE),
    # NOT on m directly. We probe both; whichever has gc wins.
    A_pre = None
    gc_owner_pre = None
    for _o in (m, getattr(m, "net", None)):
        if _o is not None and hasattr(_o, "gc"):
            try:
                _o.eval()
                with torch.no_grad():
                    A_pre = _o.gc(idx).detach().cpu().numpy().astype(np.float32)
                gc_owner_pre = _o
            except Exception as e:                       # noqa: BLE001
                print(f"    frozen-graph pre-load failed: {e}")
            break

    adapter.model.load_state_dict(state)
    adapter.model.eval()

    # 'gc' is MTGODE's graph-constructor submodule (MTGNN-style). It lives on the
    # inner MTGODE instance (m.net), so include that as a search owner.
    _inner = getattr(m, "net", None)
    owners = (adapter, m) if _inner is None else (adapter, m, _inner)
    for owner in owners:
        for name in ("get_adjacency", "learned_adjacency", "graph_learner",
                     "get_graph", "compute_adj", "_build_adj", "gc"):
            fn = getattr(owner, name, None)
            if fn is None:
                continue
            try:
                with torch.no_grad():
                    A = fn() if not hasattr(fn, "forward") else fn(idx)
                A = np.asarray(A.detach().cpu() if hasattr(A, "detach") else A, dtype=np.float32)
                if A.ndim == 2 and A.shape[0] == A.shape[1]:
                    if A_pre is not None and name == "gc":
                        max_diff = float(np.abs(A_pre - A).max())
                        frozen = max_diff < 1e-6
                        status = "FROZEN — embeddings unchanged" if frozen else "TRAINED — embeddings differ"
                        print(f"    frozen-graph test: {status}  max_diff={max_diff:.6f}")
                    return A, f"model.{name}(idx)"
            except Exception as e:                       # noqa: BLE001
                print(f"    {name}() raised {type(e).__name__}: {e}")

    warnings.warn(f"{model_name}: no adjacency method found; using explicit formula. "
                  "VERIFY against the model source before trusting the result.")
    return adjacency_fallback(model_name, state), "explicit formula (UNVERIFIED)"


def m_nodes(model):
    for attr in ("num_nodes", "n_nodes", "num_node"):
        if hasattr(model, attr):
            return int(getattr(model, attr))
    return 32


def adjacency_fallback(model_name, state):
    """Published formulae. Used only if the model exposes no accessor."""
    import torch
    def find(*needles):
        hits = [(k, v) for k, v in state.items()
                if any(n in k.lower() for n in needles) and v.ndim == 2]
        if not hits:
            raise KeyError(f"no tensor matching {needles}; run --probe")
        return hits

    if model_name == "agcrn":
        # AGCRN DAGG: A = softmax(relu(E @ E.T))
        k, E = find("node_embeddings", "node_emb")[0]
        print(f"    using {k} {tuple(E.shape)}")
        A = torch.softmax(torch.relu(E @ E.T), dim=1)
        return A.detach().cpu().numpy().astype(np.float32)

    if model_name == "mtgode":
        # MTGNN-style graph learner: two embeddings, antisymmetric tanh, top-k
        hits = find("emb", "nodevec")
        if len(hits) < 2:
            raise KeyError(f"expected two embedding tensors, found {[h[0] for h in hits]}")
        (k1, E1), (k2, E2) = hits[0], hits[1]
        print(f"    using {k1} {tuple(E1.shape)} and {k2} {tuple(E2.shape)}")
        alpha, topk = 3.0, 20
        M1, M2 = torch.tanh(alpha * E1), torch.tanh(alpha * E2)
        A = torch.relu(torch.tanh(alpha * (M1 @ M2.T - M2 @ M1.T)))
        mask = torch.zeros_like(A)
        idx = A.topk(min(topk, A.shape[1]), dim=1).indices
        mask.scatter_(1, idx, 1.0)
        return (A * mask).detach().cpu().numpy().astype(np.float32)

    raise KeyError(model_name)


# ── constructed graphs ───────────────────────────────────────────────────────

def haversine_km(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def constructed(cached, lat, lon):
    out = {}
    D = haversine_km(lat, lon)
    out["distance_km"] = D.astype(np.float32)

    # CLCRN: spherical k-NN neighbourhood, k = 25 (Table 10). A mask, not a weight
    # matrix -- CLCRN generates kernel weights from displacement at run time.
    k = 25
    nn = np.argsort(D, axis=1)[:, 1:k + 1]
    knn = np.zeros_like(D, dtype=np.uint8)
    np.put_along_axis(knn, nn, 1, axis=1)
    out["clcrn_knn"] = knn
    print(f"  clcrn_knn: k={k}, density {100*knn.mean():.0f}% of all edges")

    # STGODE spatial: Gaussian sigma = 500 km, cutoff 500 km
    sigma = 500.0
    S = np.exp(-(D ** 2) / (sigma ** 2))
    S[D > 500.0] = 0.0
    out["stgode_spatial"] = S.astype(np.float32)

    # geometric reference used by the pipeline (consumed by no model, Section 3.3.1)
    if "adj" in cached:
        out["geometric"] = np.asarray(cached["adj"], dtype=np.float32)
    else:
        print("  WARNING: cache has no 'adj'; geometric reference omitted")

    print("  NOTE: stgode_semantic (DTW over 24-h diurnal profiles) is not rebuilt here. "
          "If STGODE caches it, copy that file in; otherwise rebuild with the pipeline's "
          "own DTW routine rather than a reimplementation.")
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["t2m"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    import yaml
    from preprocessing.pipeline import load_cache
    cached = load_cache("saws")

    if args.probe:
        probe(cached); return

    # station metadata
    import pandas as pd
    meta = pd.read_csv(ROOT / "data/saws/stations_32node.csv").sort_values("node_id")
    lat, lon = meta["latitude"].to_numpy(), meta["longitude"].to_numpy()
    names = meta["station"].astype(str).to_numpy()
    store = {"station_names": names, "lat": lat, "lon": lon}

    print("constructed graphs:")
    store.update(constructed(cached, lat, lon))

    print("\nlearned graphs:")
    how = {}
    for model, (mod, cls) in LEARNED.items():
        cfg = yaml.safe_load(open(ROOT / f"experiments/configs/{model}.yaml"))
        run_meta = dict(cached["meta"])
        run_meta.update({
            "coords":            cached["coords"],
            "mean":              cached["mean"],
            "std":               cached["std"],
            "thresholds":        cached["thresholds"],
            "static_thresholds": cached["thresholds"],
        })
        cfg["meta"] = run_meta
        for target in args.targets:
            for seed in args.seeds:
                ck = ROOT / "runs" / f"{model}_{target}_s{seed}" / "checkpoint.pt"
                if not ck.exists():
                    print(f"  {model}/{target}/s{seed}: MISSING"); continue
                adapter = getattr(__import__(mod, fromlist=[cls]), cls)()
                adapter.build_model(cfg, np.asarray(cached["adj"]))
                A, route = adjacency_from_model(model, adapter, load_state(ck))
                store[f"learned/{model}/{target}/s{seed}"] = A
                how[model] = route
                print(f"  {model}/{target}/s{seed}: {A.shape} via {route}  "
                      f"diag_mass={np.trace(A)/A.sum():.3f}")

    store["extraction_route"] = np.array([f"{k}={v}" for k, v in how.items()])
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **store)
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {len(store)} arrays)")
    if any("UNVERIFIED" in v for v in how.values()):
        print("\nWARNING: at least one adjacency came from the fallback formula. "
              "Check it against the model source before using it in the paper.")


if __name__ == "__main__":
    main()
