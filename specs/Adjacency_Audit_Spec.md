# Adjacency Matrix — Audit, Test & Refinement Specification

**Project:** WeatherFLOW (CSC4019Z) · Praise Jaravani
**Doc status:** Draft for review — decisions marked ⚠ require Praise sign-off; one branch requires Dr. Moodley
**Executing agent:** Claude Code, under the rails in this document
**Last updated:** 2026-08-03

---

## 0. Purpose and scope

`preprocessing/graph.py` builds a single static adjacency `A ∈ ℝ^{999×999}` (Gaussian kernel on haversine distance, 300 km cutoff, σ = 100 km, row-normalised) and the trainer contract hands it to every model as `forward(x, adj)`. Gaibie et al. (2024) handled adjacency **per model**: AGCRN random-init/learned (no prior graph), GWN random self-adaptive, CLCRN KNN on induced spherical distance from station coordinates. Upstream AGCRN and MTGODE generate their own graphs from learned embeddings; upstream CLCRN builds KNN spherical neighbourhoods from coordinates.

**The audit question:** does any production model in Praise's 3-model matrix actually *consume* the static A? The answer determines whether the pipeline is (a) simultaneously upstream-faithful and Gaibie-aligned with A as inert shared-interface plumbing, or (b) carrying an undisclosed deviation — in the worst case one that injects geography into the "no map given" cells and breaks the single-factor structure of the headline results.

**Deliverables:**
- **A — Static audit:** file-level trace of `adj` through each adapter + config grep. Produces a per-model hypothesis table.
- **B — Empirical probe (decisive):** forward-pass A-substitution tests on production checkpoints at N=999, covering call-time *and* construction-time consumption, plus a CLCRN coordinate-source check. Settles the question regardless of adapter code complexity.
- **C — `graph.py` hardening & characterisation:** unit tests, two hygiene fixes, degree/weight characterisation artifact, provenance doc.
- **D — Outcome-contingent refinement:** pre-committed decision tree per probe outcome, from a one-sentence methods disclosure up to a stop-the-line escalation.

**Hard scope limits:** Praise's three production models (AGCRN, MTGODE, CLCRN) + archived ClimODE-inspired noted only. Teammate adapters (STLGRU, Lite-STGNN, DiffSTG, USTD-spirit) are **out of audit scope but in blast radius**: the archive shows several of them register the supplied adjacency as buffers — i.e. they very likely *do* consume A. Therefore **no destructive change to `graph.py`'s output semantics is permitted under any outcome**; refinements must be additive/opt-in, and a courtesy note goes to Zoleka and Génevieve (their upstream-faithfulness question, not ours). No retraining is authorised by this spec; at most one 1-seed sensitivity run under an explicit gate.

---

## 1. Background facts the spec relies on

1. **Code correctness:** `build_distance_matrix` and `build_adjacency_matrix` are correct as written (vectorised haversine with `arcsin` clip guard; diagonal zeroed; hard cutoff; row-normalise with zero-row guard). The post-slice re-normalisation at `cli.py:340` is **required, not a bug** — slicing breaks row sums.
2. **Provenance:** thresholded-Gaussian-on-distance is the Davidson & Moodley (2022) construction (§3.5, Euclidean), itself the DCRNN convention (Li et al. 2018). Ours upgrades Euclidean → haversine. It is the *other* predecessor's protocol, not an invention — citable if consumed, one-line explicable if inert.
3. **Convention deltas vs DCRNN lineage:** we zero the diagonal (DCRNN keeps w_ii = 1) and row-normalise (DCRNN typically uses the kernel weights directly or symmetric normalisation). Legitimate choices; must be documented if A is consumed anywhere.
4. **Geometry at defaults:** 300 km ≈ 6-cell disc radius → interior degree ≈ 110 (exact value computed in C); weights ~0.88 at 1 cell (50 km) decaying to ~0.011 at cutoff; variable degree at domain edges (this is the KNN-vs-Gaussian topological difference flagged in the audit).
5. **Known adapter facts (archive):** CLCRN's `ConvAttrs` stores `sparse_idx`, `local_graph_coor`, `geodesic`, `angle_ratio` as tensor attributes built **at construction** — the CLCRN question is precisely whether those derive from coordinates (upstream-faithful) or from the static A's sparsity pattern (hybrid deviation). A forward-pass CPU/GPU equivalence test precedent exists (job 964078, tolerance ~2.2×10⁻⁵) — reuse its harness shape and tolerance.
6. **Gaibie separation:** IDW in Gaibie is time-series imputation (§3.2), unrelated to adjacency. "Random" for AGCRN means random *embedding* init feeding DAGG — there is no random fixed graph.

---

## 2. Definitions

**CONSUMED (call-time):** substituting A at `forward()` changes outputs beyond tolerance.
**CONSUMED (construction-time):** rebuilding the adapter with a substituted A (then loading the same checkpoint weights) changes outputs beyond tolerance, even if call-time substitution does not.
**IGNORED:** neither substitution changes outputs beyond tolerance → A is inert for that model.

**Tolerance:** max |Δ| ≤ 5×10⁻⁵ in z-space outputs on the probe batch (float32 forward noise ceiling, anchored to the 964078 precedent of 2.23×10⁻⁵ max). Report max and mean |Δ| regardless of verdict. Borderline (within 10× of tolerance) → rerun probe batch at float64 on CPU before classifying.

---

## 3. Deliverable A — static audit (local, ~30 min)

For each of `agcrn_adapter`, `mtgode_adapter`, `clcrn_adapter` (+ the archived ClimODE-inspired adapter, documentation only):

1. Trace the `adj` argument: consumed by upstream module / stored as buffer and read in forward / stored and never read / dropped.
2. Grep adapter + upstream + production config for predefined-graph switches: `predefined_A`, `buildA_true`, `supports`, `static_feat`, `adj_mx`, `graph_input`, `use_graph`, `sparse_idx` provenance.
3. CLCRN specifically: identify the inputs to `ConvAttrs` construction — coordinates path vs adjacency path — and record which config keys carry lat/lon into the adapter.
4. Output: `specs/ADJACENCY_AUDIT.md` §A — a per-model hypothesis table (`expected: IGNORED/CONSUMED`, evidence lines, file:line citations).

A does **not** close the question. Its job is to make B's results interpretable and catch config-level surprises (e.g. an MTGODE predefined-graph flag accidentally on).

---

## 4. Deliverable B — empirical probe (decisive)

### 4.1 Probe matrix

For each model, checkpoint = production seed 42, `model.eval()`, `torch.no_grad()`, one fixed test batch (B = 8) at N = 999, all comparisons in z-space:

| Probe | Substitution | Catches |
|---|---|---|
| P1 | real A (reference forward, run twice) | run-to-run nondeterminism floor — must be ≤ tolerance or fix determinism first |
| P2 | identity matrix, row-normalised | call-time consumption |
| P3 | seeded random row-normalised A (rng seed 1337) | call-time consumption, weight-sensitivity |
| P4 | row-permuted real A (fixed permutation π applied to rows+cols of A only, **not** to x) | subtle structural use that P2/P3 might mask |
| P5 | **construction-time**: rebuild adapter with identity A, load seed-42 `state_dict`, forward with real x | construction-time consumption (the CLCRN `ConvAttrs` case) |
| P6 | CLCRN only — rebuild with true coords jittered by +0.25° lat | confirms coordinates (not A) are the live geometric input: outputs **should** change |
| P7 | checkpoint inspection — scan `state_dict` + named buffers for tensors of shape (999, 999) or KNN-style index tensors; fingerprint against static A's values/sparsity | whether A's values were baked into saved state |

Verdict per model: **IGNORED** iff P2–P5 all within tolerance. **CONSUMED** otherwise, with the failing probe(s) identifying the pathway. P6 failing to change CLCRN outputs is itself a red flag (would mean geometry enters some other way — investigate before classifying).

### 4.2 Impact quantification (conditional — only for CONSUMED models)

B2: rerun 50 test batches under real A vs identity A; report per-variable MAE under each. Converts "outputs differ by 3×10⁻³ in z-space" into "MAE moves by X% " — the number the refinement decision actually needs.

### 4.3 Execution rails

- Harness: `scripts/probe_adjacency.py` — pure inference, streams batches, never accumulates preds. Loads the cache subset needed for one/50 batches only if the loader allows; otherwise accepts the ~22 GB cache load.
- **sbatch only** (standing head-node policy — the 2026-06-24 incident is the binding precedent). Single job covers all models: `--partition=l40s --account=l40sugrd --mem=64G --time=02:00:00`; bump to 128G only if the loader forces a full cache load. CLCRN requires GPU.
- Local smoke first at N = 10 with untrained models: asserts the harness itself (P1 floor ≈ 0, P2 ≠ reference for a stub model known to consume A, P5 rebuild-and-load roundtrip works).
- Output: `outputs/adjacency_probe/{model}_probe.json` — per probe: `{max_abs_delta, mean_abs_delta, verdict, notes}`; plus `probe_summary.md` table.

---

## 5. Deliverable C — `graph.py` hardening & characterisation

Independent of probe outcome (the module stays in the pipeline for teammate models regardless).

### 5.1 Hygiene fixes (two, both one-liners)

1. Scalar `haversine_distance`: add the same `np.clip(√a, 0, 1)` guard the vectorised path has (antipodal/fp-overshoot). Add a test.
2. Dead config: the SAWS override (500 km / 200 km in `saws_lekuba.yaml`) is unreachable under ERA5-only scope — mark deprecated in-place with a comment referencing the ERA5 descope; do **not** delete (history value).

Plus a docstring block in `build_adjacency_matrix` recording: Davidson/DCRNN provenance, the diagonal-zeroed + row-normalised convention deltas, and the "shared-interface artifact; consumption is per-adapter" caveat.

### 5.2 Unit tests (`tests/test_graph.py`)

1. Haversine known value: Cape Town (−33.925, 18.424) ↔ Johannesburg (−26.204, 28.047) ≈ 1,265 km ± 10.
2. `D` symmetric, zero diagonal, non-negative; scalar vs vectorised agreement on random pairs.
3. Clip guard: antipodal points do not NaN (scalar + vectorised).
4. Row sums = 1 for rows with any edge; isolated-node row (synthetic coords beyond threshold) stays all-zero without div-by-zero.
5. Threshold monotonicity: sparsity pattern at 200 km ⊆ pattern at 300 km.
6. Determinism + dtype float32.
7. Production-grid structure: on the real 27×37 coords — interior-node degree equals the analytic disc count (compute once, assert exact); corner degree < edge degree < interior degree; nonzeros confined to |Δrow| ≤ 6 and |Δcol| ≤ 7 band offsets (6-cell disc + latitude narrowing margin).
8. Weight sanity: w at 1-cell spacing ∈ (0.85, 0.92) pre-normalisation; w just under cutoff < 0.02.

### 5.3 Characterisation artifact

`outputs/adjacency_probe/graph_characterisation.json` + one figure: degree histogram (interior/edge/corner), weight-vs-distance curve with threshold marked, and the exact interior degree. Feeds the methods section if any model consumes A; feeds one sentence if none do. Runs locally (coords only, no cache).

---

## 6. Deliverable D — outcome-contingent refinement

Pre-committed branches. Do not improvise past them.

### D-0 · All three IGNORED *(expected outcome)*

- `specs/ADJACENCY_AUDIT.md` finalised with probe evidence.
- Methods sentence (paste-ready): *“The preprocessing pipeline constructs a static thresholded-Gaussian adjacency on haversine distance following Davidson & Moodley (2022) for pipeline generality; probe tests confirm none of the evaluated architectures consume it — AGCRN and MTGODE generate adjacency from learned embeddings, and CLCRN constructs K-nearest-neighbour spherical neighbourhoods from node coordinates — matching their original implementations and the per-model graph handling in Gaibie et al. (2024).”*
- No code change beyond Deliverable C. Close the audit.

### D-1 · CLCRN CONSUMED (construction- or call-time)

Severity: **moderate** — still a geometric, distance-derived prior, so the 2×2 cell assignment and the −59% *interpretation* survive; but it is an upstream-fidelity deviation of exactly the class the ClimODE standard was set on (silent substitution of Gaussian-disc/variable-degree for KNN/k-regular), and it weakens the "closer to Lin et al.'s original regime than Gaibie" argument.

1. `PATCHES.md` entry, class-labelled (graph-construction substitution), citing probe evidence and B2 impact numbers.
2. Explicit methods disclosure paragraph (what upstream does, what ours does, measured impact).
3. ⚠ **Decision (Praise, Moodley consulted): disclose-and-keep vs align-to-upstream-KNN.** Recommendation: disclose-and-keep, **bounded by a 1-seed sensitivity run** — retrain CLCRN seed 42 with the upstream KNN graph path and compare MAE (the ClimODE fixed-step validation pattern). If the 1-seed delta is within seed noise (±1 CLCRN std per variable), disclosure suffices; if larger, escalate to a rerun decision with Moodley. No 10-seed rerun is authorised by this spec.

### D-2 · AGCRN or MTGODE CONSUMED

Severity: **stop-the-line.** Geography injected into the learned-graph cells breaks single-factor status: AGCRN-consuming contaminates the −59% spatial contrast; MTGODE-consuming makes AGCRN↔MTGODE a two-difference comparison and voids the clean −23% CT claim as stated.

1. Freeze deck/paper claims touching the affected contrast. No edits before analysis.
2. Run B2 immediately for the affected model(s) — the identity-vs-real MAE delta bounds the contamination.
3. Escalate to Praise + analytical review (this conversation) with probe JSONs and B2 numbers before any reframing. Possible outcomes range from "Δ negligible → footnote + disclosure" to "claims restructured around a hybrid-prior reading." That call is not made inside this spec.
4. Moodley informed at the next check-in with the measured delta, per the raise-early-with-evidence standard.

### D-3 · Cross-cutting (any outcome)

- Courtesy note to Zoleka & Génevieve: static A is live infrastructure their adapters buffer; whether *their* upstreams intend an external graph is their faithfulness question — attach the probe harness so they can rerun it on their models in minutes.
- The archived ClimODE-inspired model's A-dependence is documented in `PROVENANCE.md` only (it is out of the live matrix; no probe needed).

---

## 7. Sequencing & gates

| Stage | What | Where | Gate |
|---|---|---|---|
| G0 | Deliverable A hypothesis table | local | file:line evidence per model |
| G1 | Harness smoke (N=10, stub models incl. one known consumer) | local CPU | P1 floor ≈ 0; stub consumer correctly flagged; P5 roundtrip loads |
| B | Probe job, 3 models × P1–P7 | sbatch, l40s | JSONs + verdict table |
| C | graph.py tests + hygiene + characterisation | local | tests green |
| D | Branch per §6 | — | Praise sign-off on branch taken |

Schedule fit: A + G1 + C are an afternoon; the probe job is minutes of compute inside a 2 h allocation. Entire audit closable **before the check-in**, which is the point — walk in with a verified statement, not an assumption. No conflict with the ClimODE spike (different partition footprint, trivial GPU time) or the VPT spec (whose Stage-0 adapter-input audit shares Deliverable A's file reading — do them in one pass).

---

## 8. Required tests (`tests/test_adjacency_probe.py` + `tests/test_graph.py`)

Probe harness: (1) P1 determinism floor on a fixed stub; (2) stub-consumer flagged by P2/P3; (3) stub-ignorer passes P2–P5; (4) P4 permutation applied to A only, x untouched (assert x hash unchanged); (5) P5 rebuild-load-forward roundtrip equals direct forward for an ignorer; (6) JSON schema of probe outputs. `graph.py`: the eight tests in §5.2. All green locally before sbatch.

---

## 9. Reporting rules

1. The audit's outcome statement always names the mechanism, not just the verdict: *which* probe distinguished, at *what* magnitude.
2. Every consumed-path disclosure carries the Davidson/DCRNN provenance line — the construction has pedigree; the issue (if any) is silence, not the design.
3. My earlier conversational explanations of CLCRN's graph (KNN band-diagonal, "handed exactly its true neighbours") were made under the upstream-default assumption; if D-1 fires, those explanations get re-cut to the measured reality (still frozen, still geometric, still banded — k-regular vs variable-degree changes).
4. No table in the paper attributes graph provenance to a model without probe evidence behind it.

---

## 10. Decisions required

| # | Decision | Owner | Recommendation |
|---|---|---|---|
| D1 | Run the audit now, before the check-in | Praise | Yes — minutes of compute, converts an assumption into a verified methods sentence |
| D2 | If CLCRN consumes: disclose-and-keep + 1-seed sensitivity vs full KNN alignment | Praise (Moodley consulted) | Disclose-and-keep unless the 1-seed delta exceeds seed noise |
| D3 | If AGCRN/MTGODE consumes: freeze-and-escalate protocol | Praise | Accept as written — no unilateral claim edits |
| D4 | Keep `graph.py` in pipeline regardless of verdict | Praise | Yes — teammate adapters buffer it; changes must stay additive |
| D5 | Share probe harness with teammates | Praise | Yes — their faithfulness question, our tooling |

---

## 11. File map (new code only)

```
scripts/probe_adjacency.py          # P1–P7 harness; per-model, checkpoint seed 42; streams; JSON out
slurm/probe_adjacency.sh            # --partition=l40s --account=l40sugrd --mem=64G --time=02:00:00
tests/test_adjacency_probe.py
tests/test_graph.py
specs/ADJACENCY_AUDIT.md            # Deliverable A table → finalised with probe verdicts
outputs/adjacency_probe/            # {model}_probe.json, probe_summary.md, graph_characterisation.json + figure
```

Existing files touched: `preprocessing/graph.py` (clip guard + docstring only), `experiments/configs/saws_lekuba.yaml` (deprecation comment only). Nothing else.
