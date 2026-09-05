# ADJACENCY AUDIT — Verdict, Evidence, Follow-ups

**Spec:** [specs/Adjacency_Audit_Spec.md](Adjacency_Audit_Spec.md)
**Executed:** 2026-08-03
**Scope adjustment vs spec:** the spec was written for the ERA5 999-node grid; probes here run against the current production SAWS 25-node network at seed 42 with reproducible-init untrained models. The **architectural question — does adapter X read A — is dataset-agnostic**; using untrained models is valid because we're measuring whether the code path exists, not what weights would learn to do with it.

---

## Headline

**All three production models — AGCRN, MTGODE, CLCRN — empirically IGNORE the static adjacency matrix produced by `preprocessing/graph.py`.** Every probe (P1–P5 + P7) returns exactly 0.00e+00 max |Δ| on every model — the strongest possible verdict, at the float32 precision floor. **D-0 branch fires** per spec §6.

Practical implication: `graph.py`'s A is inert plumbing for our three-model matrix. Kept in the pipeline because teammate adapters (STLGRU, DiffSTG-family, USTD-spirit) likely do consume it as buffers.

Methods-section sentence (paste-ready):

> The preprocessing pipeline constructs a static thresholded-Gaussian adjacency on haversine distance following Davidson & Moodley (2022) for pipeline generality; probe tests confirm none of the evaluated architectures consume it — AGCRN and MTGODE generate adjacency from learned embeddings, and CLCRN constructs K-nearest-neighbour spherical neighbourhoods from node coordinates — matching their original implementations and the per-model graph handling in Gaibie et al. (2024).

---

## Deliverable A — static audit

Per-adapter code trace of the `adj` argument:

| Adapter | Where adj enters | What happens to it | Docstring claim | Verified |
|---|---|---|---|---|
| [agcrn_adapter.py](../models/baselines/agcrn_adapter.py) L23 | `build_model(config, graph)` L32 | Only `graph.shape[0]` read for `N`; adj **not passed to model**; forward is `self.model(x)` L52 | "ignored by AGCRN (learns its own)" L27, L47 | ✅ Confirmed by probe |
| [mtgode_adapter.py](../models/continuous_time/mtgode_adapter.py) L24 | `build_model(config, graph)` L36 | Only `graph.shape[0]` read for `N`; adj not passed; forward is `self.model(x)` L63 | "ignored (learns its own)" L5, L31, L57 | ✅ Confirmed by probe |
| [clcrn_adapter.py](../models/baselines/clcrn_adapter.py) L38 | `build_model(config, graph)` L48 | Only `graph.shape[0]` read; coords passed via `config["meta"]["coords"]` L57–60; adj not passed at forward L63 | "not used by CLCRN — CLCRN uses its own spherical k-NN graph" L4-5, L43, L69 | ✅ Confirmed by probe |
| ClimODE-inspired (archived) | — | Out of live matrix | — | Not probed (documented in `results/archive/climode_inspired/PROVENANCE.md`) |

**Config-level grep** (from spec §3.2): no `predefined_A`, `buildA_true=False`, `supports=A`, `graph_input`, or `use_graph` flag in the model YAMLs or upstream configs that would activate an A-consumption path. MTGODE's model wrapper explicitly sets `buildA_true=True, predefined_A=None` at [mtgode_model.py](../models/continuous_time/mtgode_model.py#L83-86).

**CLCRN coordinate path** (spec §3.3): [clcrn_model.py](../models/baselines/clcrn_model.py#L42-58) `_build_kernel_inputs()` runs `KernelGenerator(coords_lonlat, k_neighbors=k, local_map="fast")` and derives `sparse_idx`, `geodesic`, `angle_ratio`, `loc_info` **from coordinates only**. Never touches adj. The `coords_lonlat` passed at build time is a `[:, [1, 0]]` column-swap of `config["meta"]["coords"]` (which the pipeline attaches from the SAWS station-coordinates CSV).

Static audit alone does not close the question — this is what B is for.

---

## Deliverable B — empirical probe

### Harness

[scripts/probe_adjacency.py](../scripts/probe_adjacency.py) — reproducible, ~1 minute on CPU at N=25. Implements P1, P2, P3, P4, P5, P7 from spec §4.1. P6 (CLCRN coord jitter) is omitted here because it would require touching the CLCRN adapter's build path to accept a `coords` kwarg through the probe — the coord-source has been verified statically instead (§A above).

Key harness design notes learned during implementation (both would have produced false-CONSUMED verdicts if unaddressed):

- **P1 must measure same-adapter-twice, not fresh-then-fresh.** Fresh-then-fresh is deterministic by construction (seeded init) and returns 0.0, hiding forward-call nondeterminism. MTGODE's adaptive-graph builder calls `torch.rand` during forward without a seeded generator, giving 3–5 × 10⁻³ variance across consecutive calls — big enough to falsely trigger the 5×10⁻⁵ CONSUMED threshold. The harness now reseeds `torch.manual_seed(42)` before every forward pass.
- **Adapters vendored under bare `model.*` names collide across probes.** AGCRN, CLCRN and MTGODE each import from a package literally called `model`, resolved via `sys.path` injection. After one adapter loads, `sys.modules['model']` and `sys.path` prefix are polluted, causing the next adapter to import the previous vendor's package (with different submodules). Fix: snapshot `sys.path` at startup, restore between probes; purge `sys.modules['model']` and any `model.*` keys.

### Results

Batch: B=4, T=12, N=25, D=6, weights seed 42, tolerance 5×10⁻⁵, CPU.

| Model | P1 floor | P2 identity | P3 random | P4 permuted | P5 construct | Verdict |
|---|---|---|---|---|---|---|
| **CLCRN**  | 0.00e+00 | 0.00e+00 (IGNORED) | 0.00e+00 (IGNORED) | 0.00e+00 (IGNORED) | 0.00e+00 (IGNORED) | **IGNORED** |
| **MTGODE** | 0.00e+00 | 0.00e+00 (IGNORED) | 0.00e+00 (IGNORED) | 0.00e+00 (IGNORED) | 0.00e+00 (IGNORED) | **IGNORED** |
| **AGCRN**  | 0.00e+00 | 0.00e+00 (IGNORED) | 0.00e+00 (IGNORED) | 0.00e+00 (IGNORED) | 0.00e+00 (IGNORED) | **IGNORED** |

P7 (checkpoint scan for (N,N) tensors): 0 buffers, 0 parameters shaped (25, 25) in any of the three models. Nothing structurally A-shaped is baked in.

Verdict per model: **IGNORED at both call time and construction time on every substitution.** No probe distinguished substituted A from real A at any resolution above the float32 precision floor. Result is unambiguous — no borderline flags, no float64 re-check needed.

Raw per-probe JSONs at [outputs/adjacency_probe/{clcrn,mtgode,agcrn}_probe.json](../outputs/adjacency_probe/); summary table at [outputs/adjacency_probe/probe_summary.md](../outputs/adjacency_probe/probe_summary.md).

### What we learned along the way (worth preserving)

1. **MTGODE has forward-call nondeterminism at ~5×10⁻³.** Not a defect for training (loss backprop averages over the noise), but future eval-time reproducibility work needs to know: repeated forward passes on the same model + input can drift ~5e-3 per output cell unless `torch.manual_seed` is set before each call. Documented in the probe harness so a future audit doesn't re-hit this.
2. **Probe artifacts caught by design.** The probe design (compare against a same-adapter same-adj baseline; run at strict tolerance) correctly refused to trust the initial "MTGODE CONSUMED" reading and forced diagnosis of the RNG source. This is exactly the spec's intent: static docstrings aren't evidence, and neither is a single empirical measurement without a well-calibrated noise floor.

---

## Deliverable C — `graph.py` hardening & characterisation

### C.1 Hygiene fixes (both applied)

1. **Scalar `haversine_distance` clip guard** ([graph.py:17-27](../preprocessing/graph.py#L17-L27)): antipodal / float-overshoot values of `sqrt(a)` are now clipped to `[0, 1]` before `arcsin`, matching the vectorised path's behaviour. Test coverage: `TestClipGuard::test_scalar_antipodal_no_nan`, `test_identical_points_zero_distance`.
2. **Provenance docstring** ([graph.py:1-33](../preprocessing/graph.py#L1-L33)): expanded to record the Davidson (2022) / DCRNN (Li et al. 2018) lineage, the two convention deltas (diagonal zeroed, row-normalised), and the "A is a shared interface, not a shared method" caveat with a pointer to this audit.

Note: the SAWS override in `saws_lekuba.yaml` is **not** deprecated — it's actively used and now baked into pipeline defaults ([preprocessing/pipeline.py](../preprocessing/pipeline.py) dataset-aware branch). The spec's C.1 item 2 was written under the ERA5-only assumption and doesn't apply post-SAWS-pivot.

### C.2 Unit tests

[tests/test_graph.py](../tests/test_graph.py) — 18 tests, all passing:

- Cape Town ↔ Joburg haversine matches the 1265 km published value
- Distance matrix symmetric, zero diagonal, non-negative, scalar↔vectorised agree
- Antipodal-point clip guard (scalar + vectorised); identical-point zero distance
- Row sums = 1 for connected rows; isolated-node row is all-zero (no div-by-zero)
- Threshold monotonicity: sparsity(200 km) ⊆ sparsity(300 km)
- Determinism + float32 dtype
- SAWS network structure (adapted from spec's ERA5-grid tests): every station has ≥1 neighbour at SAWS defaults; degrees vary (not all identical); strongest edge is to nearest connected neighbour
- Weight sanity: at 100 km separation vs 300 km, raw-weight ratio matches the Gaussian exp(4) ≈ 55× prediction

The two ERA5-specific tests from spec §5.2 (interior-node degree on the 27×37 lattice; band-diagonal structure) are replaced by three structural tests appropriate for a scattered station network — the interior/edge/corner distinction doesn't apply to a point cloud.

### C.3 Characterisation artifact

[outputs/adjacency_probe/graph_characterisation.json](../outputs/adjacency_probe/graph_characterisation.json) + [figure](../outputs/adjacency_probe/graph_characterisation.png).

SAWS 25-node network at defaults (500 km / 200 km / row-normalised / zero-diagonal):
- 128 undirected edges, sparsity 41.0%
- Degree range [3, 16], mean 10.2, std computed
- Row sums all = 1.0 (no isolated nodes)
- Pairwise distance median 559 km, max 1422 km

The one-line methods contribution if anyone asks: even if a future model DID consume A on this network, the graph is well-behaved — connected, degree-heterogeneous, weights decay smoothly across ~1.5 orders of magnitude before hitting the cutoff.

---

## Deliverable D — outcome-contingent refinement

**Branch fired: D-0 (all three IGNORED).** Per spec §6:

- `specs/ADJACENCY_AUDIT.md` finalised — this document
- Methods sentence prepared (see Headline)
- **No code change beyond Deliverable C.** Audit closed.

D-1 (CLCRN CONSUMED) and D-2 (AGCRN/MTGODE CONSUMED) did not fire — no PATCHES.md entry, no methods deviation to disclose, no freeze-and-escalate protocol, no 1-seed sensitivity retrain required.

### D-3 cross-cutting note

**Teammate courtesy note:** the probe harness is dataset-agnostic and adapter-generic (add adapter to `ADAPTER_REGISTRY` in `scripts/probe_adjacency.py` and run). Recommend Zoleka and Génevieve rerun it on STLGRU / Lite-STGNN / DiffSTG / USTD-spirit adapters — for their models the answer is likely CONSUMED (upstream DiffSTG-family and USTD-spirit expect a graph input), which then makes the same-question-different-answer disclosure explicit rather than assumed. Their upstream-faithfulness question, our tooling.

---

## Reproducibility

```
# Reproduce all three deliverables from scratch
pytest tests/test_graph.py -v                                # C: unit tests
python scripts/probe_adjacency.py --device cpu               # B: probe (~1 min)
python scripts/graph_characterise.py                          # C.3: characterisation
```

Outputs land in [outputs/adjacency_probe/](../outputs/adjacency_probe/).

## Files touched by the audit

**New:**
- `scripts/probe_adjacency.py` — P1–P5 + P7 probe harness
- `scripts/graph_characterise.py` — degree/weight characterisation
- `tests/test_graph.py` — 18 unit tests
- `specs/ADJACENCY_AUDIT.md` — this document
- `outputs/adjacency_probe/{agcrn,mtgode,clcrn}_probe.json` — per-model probe results
- `outputs/adjacency_probe/probe_summary.md` — verdict table
- `outputs/adjacency_probe/graph_characterisation.{json,png}` — network artifact

**Edited (minimal, non-destructive):**
- `preprocessing/graph.py` — scalar haversine clip guard + provenance docstring

Zero adapter code changed. Zero model config changed. Zero retraining triggered. Audit closes cleanly.
