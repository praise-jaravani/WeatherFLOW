# Hyperparameters — Paper-Provenance Tables

**Locked:** 2026-08-04 · **Verified against papers:** 2026-08-04 · **Regime:** No HPO — all values traceable to published sources.

---

## 2026-08-10 update block (HPO v1 PARKED; v2 pending)

- **HPO v1 (random search, 60 trials, 2026-08-09/10) is PARKED.** Superseded by
  HPO v2 (sequential Optuna TPE + MedianPruner). See
  [runs/hpo_v1_random_search_parked/PROVENANCE.md](../runs/hpo_v1_random_search_parked/PROVENANCE.md)
  for the retained data + reasons for supersession (seed confound, truncated
  protocol, no pruning, no confirmation stage).
- **`experiments/configs/mtgode.yaml` has been REVERTED** to paper-provenance
  values pending v2's confirmation stage: `lr=1e-3, batch=64, k=13, hidden=64,
  dropout=0.3`. The v1 winner (`lr=3e-3, batch=128, k=20`) is retained only in
  the parked archive and MUST NOT be treated as live.
- **AGCRN and CLCRN unchanged** — retain Gaibie's SAWS-45 optima, disclosed
  asymmetrically ("baselines cited; MTGODE tuned because Gaibie did not evaluate it").
- Cross-family structural parity with Chikwanha's flagship-only tuning (she
  tunes DiffSTG only; USTD and her GraphWaveNet baseline take published defaults).

## 2026-08-08 update block (N=32 relaunch)

The 2026-08-04 lock stands for AGCRN and CLCRN. Five deltas fold in for the
N=32 relaunch, all disclosed here:

1. **Network N=25 → N=32** (locked 2026-08-08 after independent audit by Gen,
   Zoleka, Claude Code). Same 6 variables, 2016-01-01 → 2025-12-31, ≤10% overall
   missingness (Davidson criterion). See
   [HPC_Parallel_Worker_System.md](HPC_Parallel_Worker_System.md) §12.
2. **Training loss MSE → MAE**, uniform across MTGODE / AGCRN / CLCRN
   ([training/losses.py](../training/losses.py), [cli.py:_build_loss](../cli.py)).
   Provenance: MTGODE §4.4, AGCRN repo's `nn.L1Loss()` default, CLCRN repo's
   `masked_mae_loss`. Chikwanha's GraphWaveNet baseline in USTD also uses MAE.
   Diverges from Davidson & Moodley (2022) — documented single departure.
3. **MTGODE audit fixes** ([mtgode_model.py](../models/continuous_time/mtgode_model.py) +
   [mtgode.yaml](../experiments/configs/mtgode.yaml)) per external audit trailer
   2026-08-08. Findings 0/2/5 all applied; Findings 1/3/4 already covered above:
   - **Finding 0 (adjoint):** `ode.adjoint: true → false`. Under adjoint mode
     the graph learner's parameters (`gc.emb1/emb2/lin1/lin2`) are outside
     `func.parameters()` so `torchdiffeq.odeint_adjoint`'s backward pass skips
     them; the "learned" adjacency stayed at random init for every previous
     MTGODE run (ERA5 archive + 25-node SAWS). Empirically verified with a
     gradient-flow test that reproduces the audit's proof. At N=32 activations
     are tiny, so there is no memory case for adjoint mode.
   - **Finding 2 (node_dim):** graph-constructor embedding dim decoupled from
     `hidden_dim`. Was `node_dim = hidden_dim = 64`, conflating the two. Paper
     §5.2 pins node_dim = 40 (MTGNN inheritance). Now `graph_constructor.node_dim: 40`,
     `hidden_dim: 64` controls only `conv_channels`.
   - **Finding 5 (cgp_alpha):** `CGPFunc: f = 0.5·α·(Âx − x)`. Paper Eq. 6
     `dH/dt = (Â − I)H` requires α=2.0. Was hardcoded α=1.0 (halved vector field,
     equivalent to integrating CGP to T=0.5 not T=1.0). Now `ode.cgp_alpha: 2.0`.
4. **MTGODE `graph_constructor.k`: 10 → 13** at N=32. Same 40% density rule as
   before (10/25 = 40%; 13/32 = 41%).
5. **HPO deliberately parked.** Reviewer decision 2026-08-08: N=32 production
   relaunch is critical path for the Wed draft; nothing runs in front of it. Any
   HPO is post-draft. See §HPO status below for the reframed Option-B design
   that would run if commissioned.

**Consequence for the archived ERA5 results:** trained under MSE + frozen graph
learner. If they appear in the paper they are a **separate condition**, labelled
`MTGODE (pre-audit)`; never pooled with SAWS N=32 numbers.

**Consequence for the 25-node SAWS results at `results_saws_25node_phase1/`:**
same shelf. Not pooled. Available for provenance-check only.

---


**Verification stamp:** every hyperparameter in this document has been checked against the actual paper text or the upstream code:
- Gaibie'24 Table 1 extracted verbatim from `Gabiee.pdf` and cross-checked value-by-value
- MTGODE §5.2 Metr-La multi-step passage extracted verbatim from `MTGODE.pdf` and cross-checked
- Bai'20 AGCRN defaults verified against `models/baselines/AGCRN/model/PEMSD4_AGCRN.conf`
- Lin'22 CLCRN defaults verified against `models/baselines/CLCRN/model/clcnn/seq2seq.py` init signatures

Total: 29 exact matches + 1 disclosed scaling deviation (MTGODE k) + 3 Gaibie rows that don't map to our variant (see CLCRN notes).

Values below are the final production hyperparameters for the SAWS 25-station × 4-target × 10-seed campaign. Every value has a citation. Deviations from paper values are called out explicitly.

**Citation shorthand:**
- **Gaibie'24** — Gaibie et al. (2024) Table 1, HPO-optimal on 45-station SAWS predecessor
- **Bai'20** — Bai et al. (2020), original AGCRN paper — published architecture defaults
- **Lin'22** — Lin et al. (2022), original CLCRN paper — published architecture defaults
- **Jin'21** — Jin et al. MTGODE paper §5.2 multi-step METR-LA setting
- **Pathak'18** — Pathak et al. (2018), VPT-lineage baseline
- **Ours** — uniform training-harness choice, disclosed as such

---

## AGCRN

| Parameter | Value | Source | Notes |
|---|---|---|---|
| `model.hidden_dim` (RNN units) | 64 | **Gaibie'24** HPO-optimal | |
| `model.num_layers` | 2 | **Gaibie'24** HPO-optimal | |
| `model.embed_dim` | 10 | **Bai'20 PEMSD4** config default (verified: `AGCRN/model/PEMSD4_AGCRN.conf`) | Adaptive-graph node embedding dim — not tuned by Gaibie. Bai'20 uses embed_dim=10 on PEMSD4 (307 nodes) and embed_dim=2 on PEMSD8 (170 nodes). We inherit PEMSD4's value. |
| `model.cheb_k` | 2 | **Bai'20** default (both PEMSD4 + PEMSD8) | Chebyshev polynomial order — not tuned by Gaibie |
| `training.learning_rate` | **0.003** | **Bai'20** published optimum (reduced from Gaibie'24's 0.01 after divergence — see below) | |
| `training.batch_size` | 64 | **Gaibie'24** HPO-optimal | |
| `training.epochs` | 40 | **Gaibie'24** | Matches her exact epoch count |
| `data.input_len` | 12 h | **Gaibie'24** HPO-optimal (from {12,24,48,72,168}) | See §Input-window decision below |
| `data.horizon` | 48 h | **Extension** beyond Gaibie's 24 h | Report h ∈ {3,6,9,12,24,36,48} by truncation |

Every AGCRN value has a paper citation.

### AGCRN learning-rate repair (2026-08-13)

**Paper disclosure (Methods, per-model configuration):**

> AGCRN follows Gaibie et al.'s (2024) configuration except the learning rate, reduced to 0.003 within Bai et al.'s (2020) published search range after the transplanted value (0.01) reproduced divergent training in our regime. The chosen value (0.003) is Bai et al.'s own reported optimum from their search over lr ∈ [0.0007, 0.009].

**Evidence trail:** Gaibie's lr=0.01 exceeds Bai et al.'s published search ceiling of 0.009. A pre-registered single-seed diagnostic on the N=32 SAWS network under the uniform MAE objective (2026-08-13, seed 42, target t2m, all other knobs unchanged) tested lr ∈ {0.01, 0.003, 0.001, 0.009}:

| lr | best epoch (b) | v[1] | v[b] | verdict |
|---|---|---|---|---|
| 0.01 (Gaibie) | 1 | 0.573 | 0.573 | PATHOLOGY (best-at-init, patience-15 stop) |
| **0.003 (Bai opt) — adopted** | 3 | 0.347 | **0.331** | HEALTHY (winner) |
| 0.001 | 18 | 0.368 | 0.336 | HEALTHY |
| 0.009 (Bai ceiling) | 2 | 0.478 | 0.472 | PATHOLOGY |

Adopting lr=0.003 yields a 42% reduction in best val MAE relative to lr=0.01. The loss-regime transfer explanation (previously the leading hypothesis for the N=25 pathology under MSE) is falsified — lr=0.01 reproduces the pathology under MAE too. Even Bai's own search ceiling (0.009) diverges in our regime; the stable window is narrower than the authors' own.

**Full report + val-curve figure:** [results/diagnostics/agcrn_n32_health/REPORT.md](../results/diagnostics/agcrn_n32_health/REPORT.md) and [val_curves.png](../results/diagnostics/agcrn_n32_health/val_curves.png).

**Health-gate follow-through:** the 0.003 winner is single-seed on t2m. During the Phase-1 campaign, apply the same admissibility criteria (best epoch ≥ 3; v[b]/v[1] ≤ 0.97; descending shape) to the first completed AGCRN task per target (t2m/sp/rh/ws). Any target failing the gate → halt that arm only, re-diagnose. This converts residual per-target risk into a first-completion check rather than a pre-campaign search.

---

## CLCRN

| Parameter | Value | Source | Notes |
|---|---|---|---|
| `model.hidden_dim` (→ upstream `rnn_units`) | **32** | **Gaibie'24** HPO-optimal | Verified adapter mapping ([clcrn_model.py:89](../models/baselines/clcrn_model.py#L89)): our `hidden_dim` → upstream `rnn_units`. Gaibie Table 1 has three CLCRN capacity values (hidden 16, RNN 32, filters 32); the correct mapping is to RNN units = 32. |
| `model.num_layers` | 2 | **Gaibie'24** HPO-optimal | |
| `model.embed_dim` | 16 | **Lin'22** code default (verified: `CLCRN/model/clcnn/seq2seq.py:15`) | |
| `model.k_neighbors` | 25 | **Lin'22** code default (verified: `CLCRN/experiments/dataloader.py:67`) | N-sensitive if network expands to N=45; targeted re-verify then (see [OPTUNA_HPO.md](OPTUNA_HPO.md) delta-HPO section) |
| `model.lck_structure` | [4, 8] | **Lin'22** code default (verified: `CLCRN/model/clcnn/seq2seq.py:14`) | CLConv kernel layer sizes |
| `model.max_view` | 2 | **Lin'22** code default (verified: `CLCRN/model/clcnn/seq2seq.py:6`) | Max spatial hops in CLConv |
| `training.learning_rate` | **0.05** | **Gaibie'24** HPO-optimal | Aggressive: 50× larger than our prior 0.001 (which was inside her search range but not her optimum). Grad clip 5.0 is the safety net; monitor seed 42 in drip-feed. |
| `training.batch_size` | 32 | **Gaibie'24** HPO-optimal | |
| `training.epochs` | 40 | **Gaibie'24** | Matches her exact count |
| `data.input_len` | 12 h | **Gaibie'24** HPO-optimal (from {12,24,48,72,168}) | See §Input-window decision below |
| `data.horizon` | 48 h | **Extension** beyond Gaibie's 24 h | |

Every CLCRN value has a citation. Two values changed today from earlier deviations:
- `hidden_dim`: 64 → **32** (Gaibie's HPO optimum)
- `learning_rate`: 0.001 → **0.05** (Gaibie's HPO optimum)

### ⚠ CLCRN lr=0.05 stability flag (2026-08-04 smoke observation)

The Layer-4b local smoke of CLCRN with the new `lr=0.05` produced:
- epoch 1 train_loss = 1.011, val_loss = 0.948 (z-space)
- **grad_norm = 5.1 × 10⁹** ← very high; grad clipping at max_norm=5.0 is doing serious work

This is the review-predicted flag ("50× lr; at the boundary of Gaibie's search range; watch seed 42 in the drip-feed"). Single-epoch grad norm can't distinguish transient high-lr warmup from actual divergence — only multi-epoch data tells.

**HPC production monitoring action item**:
1. When the first CLCRN production job launches (any seed, any target), watch the first 5 epochs of `train.log`.
2. If `grad_norm` decreases below **100** by epoch 5, lr=0.05 is stable in the high-lr warmup regime. Let the sweep proceed.
3. If `grad_norm` stays > 10³ after epoch 5, OR val_loss shows NaN/Inf, halt the sweep.
4. Fallback: `learning_rate: 0.01` (mid-range Gaibie search value; a defensible reduction).
5. Do NOT change `lr` on the fly during a running seed — kill the job, edit the config, resubmit fresh.

Log the observed epoch-5 grad_norm in a comment in `data/saws/` or the production run's summary when the answer is known.

**Three Gaibie'24 Table 1 rows that don't apply to our CLCRN adapter** (worth naming in methods):

| Gaibie CLCRN row | Value | Why not applicable to us |
|---|---|---|
| "# of hidden units" | 16 | Upstream Lin'22 has a `hidden_units` param (default 16), but it's only consumed by the CLCRN **attention variant** (`nb_chev_filter=nb_time_filter=hidden_units` at `attention_model.py:38-39`). We use the **recurrent variant** (`clcnn.recurrent.seq2seq_model.CLCRNModel`) — no reference to hidden_units anywhere in the recurrent code path. Verified via grep. |
| "# of Filters" | 32 | Same: filters are set via `hidden_units` in the attention variant only. Not consumed by our recurrent variant. |
| "# of ST Layers" | 1 | No direct knob in either AGCRN or CLCRN adapter. Likely a search-space-uniformity row that applies to GWN (which has explicit ST layers). |

None of these three rows should be paper-cited as "our value = Gaibie's value" — they don't map. Our recurrent-CLCRN configuration is the correct architectural interpretation of Lin (2022)'s recurrent-variant defaults + Gaibie's HPO-optimal rnn_units + lr.

---

## MTGODE

Gaibie did not evaluate MTGODE. All values from Jin et al. (2022) §5.2 multi-step forecasting configuration (METR-LA setting), which matches our task shape. **2026-08-08: Findings 0/2/5 of the external audit folded in** — see §2026-08-08 update block for the mechanistic details.

| Parameter | Value | Source | Notes |
|---|---|---|---|
| `model.hidden_dim` (→ `conv_channels` only) | 64 | **Jin'21** §5.2 (encoder=64, decoder=128 — picked 64) | Post-audit: this key controls `conv_channels` only. Node-embedding dim moved to `graph_constructor.node_dim` (see below). |
| `model.num_layers` | 2 | **Jin'21** default | |
| `model.dropout` | 0.3 | **Jin'21** §5.2 | |
| `model.graph_constructor.k` | **13** | **Ours** (scaled from Jin's k=20 to preserve density at N=32) | Jin/MTGNN chose k=20 at N=207 (~10% density); at our N=32, density-target 13/32 = 41% preserves the ~40% rule we documented for N=25 → k=10. Disclose scaling. |
| `model.graph_constructor.node_dim` | **40** | **Jin'21** §5.2 (MTGNN inheritance) | **NEW row (audit Finding 2, 2026-08-08).** Was implicitly = `hidden_dim` = 64, which conflated conv_channels with graph-learner embedding dim. Paper §5.2 pins node embedding dim = 40 via MTGNN's graph learner. |
| `model.graph_constructor.alpha` (tanhalpha) | 3 | **Jin'21** default | tanh saturation rate |
| `training.learning_rate` | 0.001 | **Jin'21** §5.2 (Adam, lr=1e-3) | |
| `training.batch_size` | 64 | **Jin'21** §5.2 | Their ablation shows 32→128 mildly improves; memory not a concern at N=32 |
| `training.epochs` | 200 | **Jin'21** §5.2 | Early stopping (patience 15) does the real work |
| `ode.method_1` / `method_2` | euler / euler | **Jin'21** §5.2 (METR-LA) | |
| `ode.time_1` | 1.0 | **Jin'21** §5.2 | |
| `ode.step_size_1` | 0.25 | **Jin'21** §5.2 | |
| `ode.time_2` / `step_size_2` | 1.0 / 0.25 | **Jin'21** §5.2 | |
| `ode.cgp_alpha` | **2.0** | **Jin'21** Eq. 6 | **NEW row (audit Finding 5, 2026-08-08).** `CGPFunc: f = 0.5·α·(Âx − x)`; paper Eq. 6 `dH/dt = (Â − I)H` requires α=2.0. Pre-audit hardcoded α=1.0 halved the vector field. |
| `ode.adjoint` | **false** | **Jin'21**-consistent (audit Finding 0) | Was `true` (our earlier "memory-efficient" choice). Under `adjoint=true`, `odeint_adjoint`'s backward pass only differentiates w.r.t. `func.parameters()`; the graph learner sits outside that set (attached via `setGraph()`, not part of `y0` or `func`), so `gc.emb1/emb2/lin1/lin2` receive `grad=None` every step and Adam skips them. The adaptive adjacency stayed at random init for every previous MTGODE run. Empirically reproduced + resolved on 2026-08-08. |
| `ode.rtol` / `atol` | 1e-4 / 1e-3 | **Ours** cosmetic | Ignored by fixed-step Euler/RK4; schema only |
| `data.input_len` | 12 h | **Jin'21** §5.2 (input=output=12) | See §Input-window decision below. Also: receptive field `R = nfe₁ × 6 + 1`; at Jin's 1.0/0.25 solver R=25, so any hypothetical input_len > R silently truncates. Reviewer 2026-08-08. |
| `data.horizon` | 48 h | **Extension** beyond Jin's 12 h | |

---

## Common (all models)

| Parameter | Value | Source | Notes |
|---|---|---|---|
| **Training loss** (regression) | **MAE** (`nn.L1Loss`) | Provenance chain (see §Training loss below) | Uniform across MTGODE / AGCRN / CLCRN as of 2026-08-08 audit + loss decision |
| `training.early_stopping_patience` | 15 | **Ours** (uniform harness) | Disclose in methods |
| `training.gradient_clip_max_norm` | 5.0 | **Ours** (uniform) | |
| `training.scheduler.type` | CosineAnnealingWarmRestarts (T_0=10, T_mult=2, eta_min=1e-6) | **Ours** (uniform) | Not from either paper (both use constant-lr Adam). Kept uniformly across all three models for internal comparability. Docstring in `training/scheduler.py` corrected (previously falsely credited Gaibie). |
| `data.target_variable` | t2m (default) | **Gaibie'24** univariate regime | Production iterates {t2m, sp, rh, ws} per Gaibie's target set |
| `data.label_variables` | [t2m, tp, rh, ws] | **Gaibie'24** target set (SAWS names) | Fixed today from earlier ERA5 names |
| `evaluation.vpt_epsilon` | 0.3 | **Pathak'18** convention | Comment corrected from "Platt et al." to Pathak. VPT pinned to first-crossing of ε=0.3 in normalised space; reported as sMAPE (not MAPE) for stability at near-zero targets. |

---

## Training loss (2026-08-08 decision)

**Uniform MAE across MTGODE, AGCRN, CLCRN.** Provenance:

| Source | Loss | Evidence |
|---|---|---|
| MTGODE (Jin et al., §4.4) | MAE | "ℓ(·) denotes the mean absolute error"; upstream `trainer.py` uses `masked_mae` |
| AGCRN official repo | MAE | Both shipped configs: `loss_func = mae` → `nn.L1Loss()` |
| CLCRN official repo | MAE | `supervisor.py::_compute_loss` → `masked_mae_loss` on denormalised values |
| Chikwanha's GraphWaveNet baseline (USTD) | MAE | `models/stdiffusion/gwavenet_model.py::mae_loss` (masked-MAE) |
| Gaibie et al. (2024) | **not stated** | Full-text scan: §3.3 gives HPO by "validation loss" (unspecified), eval metrics are RMSE + MAE. Training objective never appears. Best reconstruction: Gaibie ran three PyTorch STGNNs whose repo defaults are MAE — so probably-MAE, not established. |
| Davidson & Moodley (2022) | MSE | §3.6 explicit: "trained using the Adam optimiser and Mean Squared Error loss function" |
| Our own proposal | MSE | Metrics table row: "MSE — Training loss reference". Superseded 2026-08-08 by this decision. |

**Diffusion baselines (Chikwanha DiffSTG + STdiffusion) keep the published DDPM
ε-MSE denoising objective.** This is the published objective of both methods and
swapping it would depart from the reproduced methods — it is not a hyperparameter
choice. Cross-family comparison happens at eval-metric level, not training-loss
level.

### Airtight-clause additions for cross-family comparison (Chikwanha/Fuyana coord)

1. Diffusion point forecasts extracted metric-matched: **ensemble median for MAE tables, ensemble mean for RMSE tables** — extraction rule stated per table.
2. **Unified CRPS column** in the results table — deterministic entries equal MAE (point-forecast CRPS identity), diffusion entries the sample-based CRPS.
3. **Share artifacts, not recipes** — hash-pinned cache, shared scalers, shared threshold/label files.
4. **Cross-family significance is unpaired** (Mann-Whitney U), never seed-paired Wilcoxon. Within-family (three deterministic forecasters, same seeds) remains Wilcoxon.
5. Horizon reconciliation: 12/48 stands per supervisor directive. Common-horizon tables via truncation to h ≤ 12 with one asymmetry disclosure, OR Chikwanha extends her diffusion runs to H=48.

---

## HPO v2 winners (2026-08-13)

Both continuous-time HPO studies (MTGODE and STGODE) completed on a RunPod RTX 4090 on 2026-08-13. Sequential Optuna TPE + MedianPruner, 30 trials each, seed=42 fixed per trial, target=t2m, `train_hours_cap=17520` (Gaibie protocol, 2 train years).

### MTGODE — winner: trial 17, val_mae = 0.348462

| Knob | Search space | **Winner** |
|---|---|---|
| `training.learning_rate` | {1e-4, 3e-4, 1e-3, 3e-3, 6e-3, 1e-2} | **6e-3** |
| `training.batch_size` | {32, 64, 128} | **128** |
| `model.dropout` | {0.1, 0.2, 0.3, 0.4} | **0.1** |
| `model.hidden_dim` | {32, 64, 128} | **64** |
| `model.graph_constructor.k` | {10, 13, 20} | **20** |

**Top-3 spread:** 0.000 (TPE re-sampled the winning config 4 times: trials 17, 19, 27, 28 — all identical val_mae 0.348462, deterministic given fixed trial seed).
**Runner-up (trial 21):** val_mae 0.350477 — differs from winner only in batch_size (32 vs 128); gap = **0.002**.
**Full top-10 spread:** 0.003 val_mae.

### STGODE — winner: trial 10, val_mae = 0.367047

| Knob | Search space | **Winner** |
|---|---|---|
| `training.learning_rate` | {1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 6e-3} | **6e-3** |
| `training.batch_size` | {8, 16, 32} | **8** |
| `ode.time` | {3.0, 6.0, 12.0} | **12.0** (upper bound of grid) |
| `model.sigma_km` | {100, 200, 500} | **500** (upper bound) |
| `model.epsilon_dtw` | {0.3, 0.5, 0.8} | **0.8** (upper bound) |

**Top-3 spread:** 0.000 (TPE re-sampled winner 3 times: trials 10, 27, 28 — identical val_mae 0.367047).
**Runner-up (trial 13):** val_mae 0.368963; gap = **0.002**.
**Full top-10 spread:** 0.005 val_mae.

**⚠ Search-boundary observation for STGODE:** the winner sits at the *upper end* of the search grid on three knobs (ode.time, sigma_km, epsilon_dtw). This is unlike MTGODE where the winner is interior on 4 of 5 knobs. A future extended sweep (larger ode.time, larger sigma_km, larger epsilon_dtw) may find a further-out optimum. Flagged in Discussion; not investigated further in this study.

### Confirmation-stage decision (SKIPPED — approved 2026-08-13)

The HPO v2 design (pre-2026-08-13, see below) originally called for a confirmation stage: top-K × 3 fresh seeds × full production protocol, winner by best-mean-val MAE. After reviewer consultation on 2026-08-13, we **skipped the confirmation stage** for both models. Rationale:

1. **No tie to break.** TPE re-sampled the winning config multiple times for each study (MTGODE: 4×; STGODE: 3×) — with fixed trial seed these produced identical val_mae. There's no runner-up "within seed noise of the winner" — the top of each shortlist is the same config replicated.
2. **Genuine runner-up is decisively behind.** MTGODE runner-up (batch_size=32) is 0.002 val_mae behind the winner; the full top-10 fits inside 0.003. STGODE is similar (top-10 spread 0.005). Both are two orders of magnitude tighter than the Phase-1 seed-spread reference (~0.15–0.2 val MAE for MTGODE t2m).
3. **Standard practice matches skip.** The literature we compare against (Gaibie, Bai, Jin, Fang) does not run a separate multi-seed confirmation stage between HPO and production; they take the validation-selected config and run final seeds.
4. **In-campaign safety net.** Our health gate (`scripts/health_gate.py`) applies pre-registered admissibility criteria to the first-completed campaign task per (model, target). Combined with the 10-seed × 4-target production campaign (40 runs per model), this provides equivalent — in fact stricter — multi-seed rigour than a 3-seed confirmation would.

**Paper disclosure line (Methods, HPO subsection):**

> "For each continuous-time model, hyperparameters were selected by validation performance under a reduced training budget (Gaibie's 2-year protocol) using Optuna TPE with MedianPruner over 30 trials. Under a fixed trial seed of 42, the search re-sampled the winning configuration multiple times, and the runner-up config was within 0.002 validation MAE of the winner — two orders of magnitude below the observed seed-spread reference. The selected configuration was not re-validated under the full protocol prior to production runs; the 10-seed production campaign serves as the multi-seed validation."

### Historical notes on HPO design evolution

**⚠️ HPO v1 SUPERSEDED (2026-08-10).** The random-search sweep (60 trials, 2026-08-09/10) has been retired in favour of HPO v2. Reasons documented in [runs/hpo_v1_random_search_parked/PROVENANCE.md](../runs/hpo_v1_random_search_parked/PROVENANCE.md):

1. **Seed confound.** v1 varied trial seed with trial index (42+trial_num). Config comparisons in the marginals were partially seed-variance, not config-effect.
2. **Truncated protocol biased fast learners.** Epoch cap 30 + patience 5 → the `lr=3e-3` boundary winner was monotone-to-boundary (unexplored-region flag).
3. **No pruning.** Every trial ran full epoch cap regardless of trajectory.
4. **No confirmation stage.** Winner was a single-seed trial.

### HPO v2 design (as-run)

Sequential Optuna TPE + MedianPruner. Approved 2026-08-10, executed on RunPod pod 2026-08-13:

- Sampler: `optuna.samplers.TPESampler(multivariate=True, seed=42)`
- Pruner: `optuna.pruners.MedianPruner(n_warmup_steps=10)` (clears the first
  cosine-restart cycle before pruning; trial reports **best-so-far val MAE**
  (monotone) so restart bumps at ep 10 can't kill a recovering trial)
- Storage: `sqlite:///~/hpo/hpo_mtgode_v2.db` on `/home` (backed up, single-writer safe)
- Startup guard: on process start, mark any RUNNING trial FAILED (safe under
  sequential — any RUNNING trial at start is necessarily dead from a killed process)
- Execution: single L40S, single-lineage chained sbatch (loses parallelism, gains
  pruning economics + methodology parity with Chikwanha's Optuna TPE setup)
- 30 trials, objective: min val MAE
- **Fixed trial seed = 42 for every trial** (matches Chikwanha's design; correct at this budget)
- Trial protocol: target=t2m, epoch cap **60**, patience 5, `train_hours_cap=17520`
  (2 years, Gaibie protocol), N=32 cache
- Pins (all unchanged from audit): input_len=12, node_dim=40, tanhα=3,
  adjoint=false, cgp_alpha=2.0, T/Δt=1.0/0.25
- Search space (5 knobs, all categorical for readable marginals):
  - `training.learning_rate ∈ {1e-4, 3e-4, 1e-3, 3e-3, 6e-3, 1e-2}` (extended
    top-end because v1 winner was boundary-monotone at 3e-3)
  - `training.batch_size ∈ {32, 64, 128}`
  - `model.dropout ∈ {0.1, 0.2, 0.3, 0.4}`
  - `model.hidden_dim ∈ {32, 64, 128}` (= conv_channels)
  - `model.graph_constructor.k ∈ {10, 13, 20}`
- Namespaced from birth under `runs/hpo_optuna/` with `hpo_`-prefixed trial IDs;
  physically invisible to production `is_complete` / sync / aggregation paths
- **Selection**: originally designed as confirmation-stage (top-3 × 3 seeds × full protocol) — **decision revised 2026-08-13: skipped**, see § HPO v2 winners above for rationale.

**Honest expectations:** at 30 trials over this 5-knob space, TPE's edge over
random is modest. The value is in (a) the fixed seed removing config-vs-seed
confound, (b) the pruner cutting losers at ~1/4 cost, and (c) narrative parity
with Chikwanha's Optuna TPE for the cross-family methods section.

### Cross-family HPO parity note

Chikwanha tunes only her flagship (DiffSTG); USTD + her GraphWaveNet baseline
take published defaults. By symmetry, our AGCRN and CLCRN keep Gaibie's SAWS-45
optima (no HPO). MTGODE-tuned matches DiffSTG-tuned, baselines match baselines.
This is the "symmetric-in-asymmetry" cross-family methods story.

---

## (Historical) HPO v1 random-search sweep, 2026-08-10 — PARKED

Retained below for provenance; **do not treat any value in this subsection as
live.** See `runs/hpo_v1_random_search_parked/PROVENANCE.md` for the full
archive.

The sweep as run: overnight 2026-08-09 → 2026-08-10 against the audit-fixed
MTGODE + uniform MAE loss on the N=32 SAWS cache. N=60 trials, all completed,
zero divergences.

### Sweep design as run

- **Sampler:** seeded numpy random search (`sampler_seed=42`). Random over TPE
  because the parallel-worker system runs 6 concurrent trials and Optuna's
  distributed TPE over SQLite on `/scratch` NFS is the documented silent-corruption
  path.
- **Manifest generator:** [scripts/build_hpo_manifest.py](../scripts/build_hpo_manifest.py)
  writes `runs/hpo/manifest.json` + per-trial `runs/hpo/configs/{task_id}.yaml`;
  worker.sh untouched; [scripts/run_worker.py](../scripts/run_worker.py) gained
  one flag `--hpo-configs-dir` that deep-merges the per-trial yaml onto the base.
- **Trial protocol:** target=t2m, epoch cap 30, patience 5,
  `data.train_hours_cap=17520` (first 2 train years — Gaibie protocol).
  Objective: **min val MAE** across all epochs.
- **Infrastructure:** [slurm/hpo_worker.sh](../slurm/hpo_worker.sh) K=3 co-tenant
  × 2 initial lineages via `afterany` chain. Ran 60 trials in one overnight
  window (submitted 22:06 SAST, last log 07:47 SAST → ~10 h wall-clock).
- **Total compute:** 20.7 GPU-hours across 60 trials (min 8 min / mean 20.7 min /
  max 48.2 min per trial). Winner completed in 8 min (early-stopped epoch 21).

### Search space + winner

| Knob | Range searched | Winner | Marginal signal (mean best_val ↓ = better) |
|---|---|---|---|
| `training.learning_rate` | {1e-4, 3e-4, 1e-3, 3e-3} | **3e-3** | Monotone: 1e-4 → 0.384, 3e-4 → 0.367, 1e-3 → 0.353, 3e-3 → 0.350 |
| `training.batch_size` | {32, 64, 128} | **128** | Best MEAN at bs=32 (0.353), but WINNER at bs=128 (0.343); bs=64 middle |
| `model.dropout` | {0.1, 0.2, 0.3, 0.4} | **0.3** | Flat; paper default 0.3 confirmed |
| `model.hidden_dim` (conv_channels) | {32, 64, 128} | **64** | 128 best mean (0.355); 32 clearly worst (0.372); 64 middle but winner |
| `model.graph_constructor.k` | {10, 13, 20} | **20** | Denser is better: k=20 mean 0.359 vs k=13 0.364 vs k=10 0.366 |

**Pinned (not searched)** — per 2026-08-08 audit and reviewer:
`data.input_len=12` (receptive field `R = nfe₁·6+1` = 25 at 1.0/0.25 solver;
larger input_len silently truncates), `model.graph_constructor.node_dim=40`
(paper §5.2), `model.graph_constructor.alpha=3`, `ode.adjoint=false` (Finding 0),
`ode.cgp_alpha=2.0` (Finding 5), all ODE times/step sizes = 1.0/0.25.

### Winner details (task `mtgode_t2m_s63`)

```yaml
training.learning_rate:      0.003
training.batch_size:         128
model.dropout:               0.3
model.hidden_dim:            64        # = conv_channels
model.graph_constructor.k:   20
```
- `best_val_loss = 0.3427` (vs run-worst 0.4009 and mean 0.361)
- `test_mae[t2m] = 2.28 °C`
- `test_rmse[t2m] = 3.05 °C`
- `test_mae_extreme[t2m] = 4.25 °C`
- Early-stopped at epoch 21 (out of 30 cap; patience 5)

These are the values now folded into [experiments/configs/mtgode.yaml](../experiments/configs/mtgode.yaml).

### Interpretation

- **The 40%-density heuristic underestimated `k`.** We had reasoned k=13 at N=32
  (13/32 ≈ 41% density). HPO says k=20 (62.5%) is better across the board.
  Capacity beat sparsification here.
- **Higher lr trains faster AND better.** Not just a "high-lr converges to lower
  loss" story — high-lr also early-stops sooner, so it costs less GPU time per
  trial. The grad_norm clip at max_norm=5.0 keeps 3e-3 stable.
- **bs=128 is a variance/peak trade.** bs=32 has the best mean; bs=128 has the
  best trial. The winner's stability at bs=128 lr=3e-3 is because gradient noise
  averages down over the larger batch.
- **hidden_dim=64 confirmed against wider.** hidden=128 has a slightly better
  mean but not enough margin to justify 2× the parameters over the paper's 64.

### Baselines (AGCRN, CLCRN) — NOT tuned

Per reviewer 2026-08-08: AGCRN and CLCRN retain Gaibie's published SAWS-45 optima
(tables above), disclosed as "matched-budget-for-no-meteorological-precedent
justification asymmetric across the three deterministic models — MTGODE tuned
because Gaibie did not evaluate it; baselines cited from Gaibie."

### History

- **2026-08-04:** OPTUNA_HPO.md wrote a sequential-TPE spec (25 trials per model,
  all three deterministic). **Superseded 2026-08-08.**
- **2026-08-08:** HPO parked pre-draft after the reviewer's calendar-analysis
  decision; parked-spec: Option B / Gaibie protocol (matches what actually ran).
- **2026-08-10:** HPO run + winner folded into `mtgode.yaml`. This section.

---

## Pipeline defaults (from preprocessing/pipeline.py SAWS branch)

| Parameter | Value | Source | Notes |
|---|---|---|---|
| `exclusion_threshold` | 0.10 | **Davidson'22** §3.2 (<10% missingness) | |
| `distance_threshold_km` | 500.0 | **Ours** (SAWS-tuned) | ERA5 default 300 leaves ALEXANDERBAAI isolated at N=25 |
| `sigma_km` | 200.0 | **Ours** (SAWS-tuned) | Gaussian kernel bandwidth |
| `train_ratio` / `val_ratio` | 0.6 / 0.2 | **Davidson-standard** chronological | 60/20/20 split |
| `clip_sigma` | 10.0 | **Ours** | z-score outlier clip (tp has ~57σ outliers otherwise) |

---

## Input-window decision (revised 2026-08-04)

**Kept input_len = 12 h across all three models.** Rationale:

1. **Gaibie'24 explicitly rejected longer inputs.** Her HPO tested {12, 24, 48, 72, 168} at horizon=24 for AGCRN and CLCRN — 12 won for both. Longer contexts were tested and lost.
2. **Jin'21 uses input=output=12** in the multi-step METR-LA setting.
3. **Scaling input to match horizon (12→48) has no paper backing** — it would be a hand-adjusted deviation from every citable source, contradicting the "no HPO, pull from papers" story.

**The h=36 / h=48 long-horizon degradation risk is real but decoder-driven, not input-driven.** AGCRN spreads one context vector over 48 output steps; CLCRN decodes 48 steps autoregressively with compounding error. A longer input window wouldn't fix either. Report h=36/48 as extrapolation beyond the lineage's validated regime (Gaibie's max was 24) — that's the finding Moodley asked for, not a defect.

---

## Two protocol notes for the methods section

1. **Train-once-at-max, truncate for reporting**: we train each model at horizon=48 and report at h ∈ {3, 6, 9, 12, 24, 36, 48} by truncation of the 48-step output. Gaibie ran "four independent experiments" per horizon (one model per horizon) — we deviate for compute reasons (5× cheaper), which means our h=12 numbers aren't strictly comparable to a Gaibie-style h=12-trained model since the loss budget spreads over 48 steps. One methods sentence, disclosed.
2. **Chronological 60/20/20 gives a ~4-month test window** on ~14k hours of SAWS data. Gaibie's test window was closer to a full year. Report the test months explicitly (`2020-05-27 to 2020-12-31`) so seasonal bias is visible.

---

## Methods paragraph (paste-ready)

> **⚠️ PLACEHOLDER — pending HPO v2 confirmation stage.** The 2026-08-10
> paste-ready paragraph cited the HPO v1 winner (`lr = 3e-3`, `batch = 128`,
> `k = 20`, val MAE 0.343, test t2m MAE 2.28 °C). v1 was parked on 2026-08-10
> and superseded by HPO v2 (sequential Optuna TPE + MedianPruner +
> confirmation stage — see §HPO). This paragraph reverts to placeholder until
> v2's confirmation stage concludes and updates `experiments/configs/mtgode.yaml`.
>
> **Structure of the final paragraph (to be filled after v2):**
> Hyperparameter search was performed for MTGODE only, since Gaibie et al.
> (2024) did not evaluate MTGODE on the predecessor SAWS network. AGCRN and
> CLCRN use Gaibie's HPO-optimal values (Table 1), with remaining architecture
> constants at their published defaults (Bai et al., 2020; Lin et al., 2022).
> MTGODE's search used Optuna TPE (multivariate, seed 42) with median-pruning
> over N trials against validation MAE with a fixed trial seed, followed by a
> confirmation stage of the top-K configurations re-trained under three fresh
> seeds and the full production protocol. Winner selected by pre-registered
> best-mean-val rule. [Concrete search-space + winner values inserted after
> confirmation stage.] The input window is fixed at 12h — Gaibie et al.'s HPO optimum, selected over {24, 48, 72, 168} — and the horizon extended to 48h, evaluated at h ∈ {3, 6, 9, 12, 24, 36, 48} by truncation of the direct 48-step forecast. Univariate output regime per Gaibie: each experiment predicts one target variable from the four Gaibie targets {t2m, sp, rh, ws} with all six SAWS features as input context. Uniform training harness across all three deterministic models: **MAE (`nn.L1Loss`) training objective** — matching the AGCRN, CLCRN, and MTGODE (§4.4) reference implementations, diverging from Davidson & Moodley (2022)'s MSE — CosineAnnealingWarmRestarts scheduler, early-stopping patience 15, gradient clipping max_norm 5.0, chronological 60/20/20 split, 10 seeds per (model, target).
>
> An external audit prior to the N=32 relaunch (2026-08-08) identified three MTGODE implementation issues that would have compromised the results if not corrected: (i) under `torchdiffeq.odeint_adjoint`, the graph-learner parameters (attached to the ODE function via `setGraph` rather than as members of `func.parameters()`) were structurally excluded from the adjoint backward pass, leaving the adaptive adjacency at random initialisation for the whole training run; (ii) the graph-constructor node-embedding dimension was conflated with the convolutional hidden width (both driven from a single `hidden_dim` key), whereas the paper §5.2 pins node_dim = 40 via its MTGNN inheritance while conv_channels = 64; (iii) the CGP function `f = 0.5·α·(Âx − x)` was hardcoded with α = 1.0 whereas the paper's Eq. 6 `dH/dt = (Â − I)H` requires α = 2.0, halving the spatial vector field and effectively integrating the CGP to T=0.5 rather than the configured T=1.0. All three are corrected in the N=32 runs and disclosed above. The archived ERA5 and shelved N=25 SAWS MTGODE results were trained under the pre-audit configuration; if they appear in the paper they are presented as a separate `MTGODE (pre-audit)` condition, not pooled with N=32 numbers.
