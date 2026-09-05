# Optuna HPO — Comprehensive Single-Sweep Per Model

**Decided:** 2026-08-04 · **Executed (v2):** 2026-08-13 · **Author:** Praise Jaravani
**Purpose:** One rigorous hyperparameter optimisation pass per production continuous-time model (MTGODE, STGODE) before multi-seed production runs on the SAWS 32-station × 6-feature × horizon=48 corpus. AGCRN and CLCRN take Gaibie's published SAWS-predecessor optima (with the AGCRN lr repair from Bai's search range, see § AGCRN learning-rate repair in HYPERPARAMETERS.md) — this asymmetric-tuning stance is documented as the "symmetric-in-asymmetry" cross-family methods story.

---

## Post-HPO summary (2026-08-13)

Both continuous-time studies completed on a RunPod RTX 4090 pod on 2026-08-13. Timing, outcomes, and winners:

| Study | Trials | Wall-clock | Pruned / Failed | Best val_mae | Winner trial |
|---|---|---|---|---|---|
| MTGODE | 30 / 30 COMPLETE | ~2.5h | 0 / 0 | **0.348462** | trial 17 |
| STGODE | 30 / 30 COMPLETE | ~6.5h | 0 / 0 | **0.367047** | trial 10 |

**Total pod compute:** ~9h × $0.75/h ≈ **$6.75**. (Actual balance drawdown was ~$7.50 including startup + overhead. Well under the initial $9.89 budget.)

**Winner configs** (full details in `specs/HYPERPARAMETERS.md § HPO v2 winners`):
- MTGODE: `lr=6e-3, batch=128, dropout=0.1, hidden=64, k=20` (interior on 4 of 5 knobs)
- STGODE: `lr=6e-3, batch=8, ode.time=12.0, sigma_km=500, epsilon_dtw=0.8` (⚠ upper-bound on 3 knobs — search-boundary flagged for Discussion)

**Confirmation stage: SKIPPED** (approved 2026-08-13, reviewer consult). Both studies had TPE re-sample the winning config multiple times (MTGODE: 4×; STGODE: 3×) producing identical val_mae under the fixed trial seed; genuine runners-up were 0.002 val_mae behind — two orders of magnitude tighter than the Phase-1 seed-spread reference. In-campaign safety net: `scripts/health_gate.py` applies pre-registered admissibility criteria to the first-completed campaign task per (model, target); 10-seed × 4-target production campaign (40 runs per model) provides equivalent multi-seed rigour to a 3-seed confirmation.

**HPO output artefacts** (per model):
- `results/hpo_optuna_pod/hpo_{mtgode,stgode}_v2.db` — full Optuna study (all trials, intermediate values, timings)
- `results/hpo_optuna_pod/runs_{mtgode,stgode}/hpo_{model}_v2_top3.json` — ranked shortlist
- `results/hpo_optuna_pod/runs_{mtgode,stgode}/hpo_{model}_t2m_trial_XX/` — per-trial outputs (summary.json, trial_summary.json)

Model YAMLs updated to winner configs on 2026-08-13:
- [experiments/configs/mtgode.yaml](../experiments/configs/mtgode.yaml)
- [experiments/configs/stgode.yaml](../experiments/configs/stgode.yaml)

---

## Original scope (pre-execution — retained for provenance)

**Report-first mindset:** every trial's data is persisted. All 90+ trials become tables and figures in the report's methods / hyperparameters section — no re-runs to reconstruct HPO evidence later.

**Univariate regime (2026-08-04 amendment).** Matching Gaibie (2024) methodology — each experiment predicts one variable at a time, with all 6 features as input context. HPO tunes at `target_variable: t2m` (canonical); the winning per-model config is then applied to production runs across all four Gaibie targets: **{t2m, sp, rh, ws}**. `tp` and `wd` remain input features but are not predicted (Gaibie doesn't predict them either — rainfall needs bespoke zero-inflated methodology, wind direction has periodicity issues). See dataset.py `target_variable_index` + trainer.py loss-slicing.

---

## Why comprehensive, not just input_len

Auditing our current model configs against Gaibie et al. (2024) Table 1:

| Model | Match Gaibie | **Deviate from Gaibie** | Not-in-Gaibie (our defaults) |
|---|---|---|---|
| AGCRN | lr=0.01, batch=64, num_layers=2, hidden=64 | epochs=100 (Gaibie 40) | embed_dim=10, cheb_k=2 |
| CLCRN | batch=32, num_layers=2 | **hidden_dim=64 (Gaibie 16), lr=0.001 (Gaibie 0.05)** | embed_dim=16, k_neighbors=25, lck_structure=[4,8], max_view=2 |
| MTGODE | not in Gaibie | | hidden_dim=40, dropout=0.3, graph_k=10, ODE step sizes |

The **CLCRN deviations are load-bearing** — hidden_dim=64 vs Gaibie's HPO-optimal 16 is 4× the capacity; lr=0.001 vs her 0.05 is 50× smaller. If we ship paper results at these values without HPO verification on our SAWS network, a reviewer will ask why we deviated. That's the strongest justification for a full HPO sweep.

MTGODE has no Gaibie precedent at all — every hyperparameter is our own default.

---

## Search spaces (per-model)

Objective: **minimise validation loss at the best epoch** (early-stopping default; matches Davidson's HPO objective).
Sampler: **Optuna TPE** (tree-structured Parzen estimator) — smarter than random search for continuous + categorical mixed spaces.
Trials per model: **25 TPE trials** (Option C, 2026-08-04) — sufficient signal for 4–5 dimensional spaces per literature; leaves margin under the 24h HPC walltime.
Epochs per trial: **60, patience 15** — matches production training exactly. No retrain-of-winner needed.
Per-trial timeout: **55 min** (3300s). Trials exceeding this are killed; if any val_loss values were written to history.json before the kill, the best of those is reported to Optuna. Otherwise the trial is pruned. Prevents a runaway trial from eating the whole 24h budget.

### AGCRN — 4 knobs

| Knob | Range | Rationale |
|---|---|---|
| `data.input_len` | `{12, 24, 48, 72}` | Davidson/Gaibie range; our current 12 is Gaibie's AGCRN optimum on 45-station ERA5 — re-verify at SAWS N=25 |
| `model.embed_dim` | `{8, 10, 16, 32}` | AGCRN adaptive-graph node embedding size; not tuned by Gaibie |
| `model.cheb_k` | `{2, 3}` | Chebyshev polynomial order; not tuned by Gaibie |
| `model.dropout` | `{0.0, 0.1, 0.2, 0.3}` | Not in Gaibie's AGCRN row |

Kept at Gaibie's HPO-optimal: `learning_rate=0.01, batch_size=64, num_layers=2, hidden_dim=64`.

Search-space size: 4 × 4 × 2 × 4 = 128 combinations; TPE explores ~30.

### CLCRN — 5 knobs (three re-verify Gaibie deviations)

| Knob | Range | Rationale |
|---|---|---|
| `data.input_len` | `{12, 24, 48, 72}` | Gaibie's CLCRN HPO range |
| `model.hidden_dim` | `{16, 32, 64, 128}` | **We currently use 64; Gaibie found 16 optimal on 45 stations — need SAWS-scale verification** |
| `training.learning_rate` | `{0.0003, 0.001, 0.003, 0.01, 0.03, 0.05}` | **We use 0.001; Gaibie used 0.05 — 50× difference to explain** |
| `model.k_neighbors` | `{10, 25, 40}` | CLCRN k-NN spherical graph size; not tuned by Gaibie |
| `model.embed_dim` | `{8, 16, 32}` | Not tuned by Gaibie |

Kept at Gaibie's HPO-optimal: `batch_size=32, num_layers=2`.

Search-space size: 4 × 4 × 6 × 3 × 3 = 864; TPE explores ~30.

### MTGODE — 5 knobs (no Gaibie baseline)

| Knob | Range | Rationale |
|---|---|---|
| `data.input_len` | `{12, 24, 48, 72}` | For consistency across models |
| `model.hidden_dim` | `{20, 40, 80}` | MTGODE typical range |
| `training.learning_rate` | `{0.0003, 0.001, 0.003}` | Standard STGNN range |
| `model.graph_constructor.k` | `{5, 10, 20}` | Adaptive-graph size at 25 nodes; 20 is 80% connectivity |
| `ode.step_size_1` | `{0.2, 0.4, 0.6}` | Outer ODE solver step size; affects integration accuracy vs speed |

Kept fixed: dropout=0.3, batch_size=32, ODE method=euler, adjoint=True, other ODE params, num_layers=2.

Search-space size: 4 × 3 × 3 × 3 × 3 = 324; TPE explores ~30.

---

## What "all tuning data preserved" means

Per-model artifact tree after HPO completes:

```
outputs/hpo/{model}/
  study.db                           # Optuna SQLite — all 30 trials + hyperparameters
  trials/
    trial_00/
      config_used.yaml               # exact yaml that trained this trial
      history.json                   # per-epoch train_loss, val_loss, grad_norm, best_val_loss
      summary.json                   # test metrics (from the trial's cli.py run)
      hyperparams.json               # {input_len: 48, hidden_dim: 32, ...}
      stdout.log                     # full training log
    trial_01/ …
    trial_29/
  report/
    trial_log.csv                    # all 30 trials, one row each, ranked by val_loss
    winner_config.yaml               # the winner's config (paste-ready for production)
    learning_curves_all_trials.png   # overlaid val-loss curves, colour by best-vs-rest
    parameter_importance.png         # Optuna's fanova importance
    per_hyperparameter_scatter.png   # objective vs each knob's value
  README.md                          # sweep summary, ranked table, winner
```

Design intent: **any figure in the report's HPO section is generated from these files by a script**, so re-running the report generator produces identical numbers no matter when.

---

## Compute + wall-clock estimate

Per-trial time on l40s: ~30–60 min including test evaluation (verified estimate; local 1650 Ti scaled).
Per-model with 25 trials + 55-min timeout cap: **≤ 24h wall-clock, guaranteed** — timeout ensures budget fit.

**HPC reality (verified 2026-08-04)**:
- `MaxWall` on l40s association = **24h** (hard cap, cannot extend)
- `MaxTRESPU` on l40sugrd QOS = **2 L40S GPUs at a time** (not 5)
- `MaxSubmit` = 5 total (queued + running)

Wall-clock across the full 3-model sweep: **~2 days** with 2 concurrent GPU jobs (one model queues behind the first two). Realistic:
- t=0: submit array job (3 tasks); tasks 0+1 start running, task 2 queues
- t=24h: tasks 0+1 finish; task 2 starts
- t=48h: task 2 finishes

If HPC is congested, add up to 4h queue wait per task on top.

Storage: SQLite study + 30 trial dirs × ~10 MB per trial ≈ 300 MB per model, 1 GB total. Negligible.

---

## How to run

### Local smoke (1 trial, 3 epochs — no HPC)

```bash
python scripts/hpo_run.py --model mtgode --n-trials 1 \
    --smoke-epochs 3 --storage-uri sqlite:///outputs/hpo/mtgode/study.db
```

### HPC full sweep — one sbatch array command

```bash
sbatch scripts/jobs/hpo_run.sbatch
```

Produces 3 array tasks (agcrn, mtgode, clcrn) throttled to 2 concurrent (`--array=0-2%2`) — matches the QOS 2-GPU cap. Each task runs Optuna sequentially through 25 trials.

Resumable via SQLite: if a task dies, resubmit as `sbatch --array=<task_id> scripts/jobs/hpo_run.sbatch`. Optuna loads existing `study.db` and continues from where it stopped.

For faster local iteration or delta-HPO:
```bash
# Override trial count and timeout via env
sbatch --export=ALL,N_TRIALS=15,TRIAL_TIMEOUT=1800 scripts/jobs/hpo_run.sbatch
```

### After all sweeps complete: generate report artifacts

```bash
python scripts/hpo_report.py --model agcrn mtgode clcrn --out-dir outputs/hpo
```

Produces the four report figures + `report/trial_log.csv` + `report/winner_config.yaml` per model + a combined `outputs/hpo/summary.md` for the paper's methods section.

---

## Reporting rules (binding for the paper's HPO section)

1. Every hyperparameter reported has its search space AND chosen value shown, in a Gaibie-Table-1-style row.
2. Deviations from Gaibie's Table 1 are **called out explicitly** with the HPO justification: "For CLCRN we found hidden_dim=X optimal on our SAWS 25-station network, differing from Gaibie's ERA5-derived 45-station optimum of 16."
3. TPE / random-search / grid-search choice is disclosed with the trial count.
4. The per-hyperparameter importance plot (Optuna fanova) goes in the appendix — shows which knobs actually mattered.
5. The winner's configuration file is committed to the repo at `experiments/configs/{model}_winner.yaml`, so anyone reproducing the paper can point at the exact configuration.

---

## What this does NOT tune (deliberately)

- `horizon` — locked at 48 per specs/HORIZON_EXTENSION.md; we report 3/6/9/12/24/36/48 by truncation
- `num_layers` for AGCRN/CLCRN — Gaibie's 2 kept
- `batch_size` — Gaibie's per-model optima kept for AGCRN/CLCRN; MTGODE at 32
- Preprocessing hyperparameters (5%/10% missingness, station selection, adjacency thresholds) — orthogonal to model HPO, already justified in `data/saws/PIPELINE_AUDIT.md`
- ODE solver `method_1`/`method_2` — kept at 'euler' (MTGODE requirement)
- Scheduler type — kept at CosineAnnealingWarmRestarts across all models

If any of these become live questions during the paper writing, a second targeted HPO can re-open specific ones. This sweep is the primary evidence.

---

## N-sensitivity + planned delta-HPO when more stations arrive (2026-08-04)

**Context:** SAWS is scheduled to deliver ~20 additional stations in the same
five provinces (W/N/E Cape + Free State + KZN) in the coming days. When they
land, the network becomes denser (**N=25 → ~45**) but the geographic
footprint stays identical. This HPO sweep runs **now** on N=25 because we
cannot wait for the delivery — SAWS delays have already compressed our
schedule, and the paper deadline is end-of-week.

**Not every hyperparameter transfers cleanly from N=25 to N=45.** Split by
N-sensitivity:

**N-invariant (transfers cleanly):**
- `data.input_len` — temporal context per station
- `model.dropout` (AGCRN) — regularisation strength
- `training.learning_rate` — assuming batch_size held
- `ode.step_size_1` (MTGODE) — solver knob
- `model.hidden_dim` — per-node capacity

**N-sensitive (needs re-tuning when N=45 arrives):**
- **`model.k_neighbors`** (CLCRN) — the big one. `k=25` on 25 nodes is a
  fully-connected graph (k = N-1); on 45 nodes it's 55% connectivity — a
  totally different topological regime.
- **`model.graph_constructor.k`** (MTGODE) — same story. `k=10` on 25 nodes
  is 40% connectivity; on 45 nodes it's 22%.
- **`model.cheb_k`** (AGCRN) — Chebyshev K-hop reach covers a smaller
  fraction of the network as N grows. Less severe but relevant.
- **`model.embed_dim`** (AGCRN, CLCRN) — adaptive-graph interaction is
  O(N × embed_dim²); optimal capacity may drift.

### Delta-HPO protocol when N=45 data arrives

Do **not** rerun the full 30-trial-per-model sweep. Instead:

1. **Freeze all N-invariant winners** at whatever the N=25 HPO picked.
2. **Targeted re-sweep** of the 4 N-sensitive knobs above only:
   - AGCRN: `cheb_k`, `embed_dim` (2 knobs, ~8 trials each = 16 trials total)
   - CLCRN: `k_neighbors`, `embed_dim` (2 knobs, ~12 trials)
   - MTGODE: `graph_constructor.k` (1 knob, ~5 trials)
3. Use Optuna's `GridSampler` on the reduced spaces since they're small.
4. Reuse the same infrastructure (`scripts/hpo_run.py --n-trials N`,
   `scripts/jobs/hpo_run.sbatch`).
5. Estimated compute: **~30 delta trials × ~1h GPU = 30 GPU-hours total**
   vs. ~180 GPU-hours for a full re-sweep. Half a day wall-clock with 3
   parallel jobs.
6. Report the winner config for N=45 as `winner_config_N45.yaml` alongside
   the original `winner_config.yaml` — the paper's methods section reports
   both, so a reviewer can see the N-scaling honestly.

### Paper methods-section wording

Suggested honest phrasing:

> "Hyperparameter optimisation was conducted on the initial 25-station
> network via TPE across [search-space table]. A subsequent targeted
> re-optimisation of topology-sensitive parameters (`k_neighbors` for CLCRN,
> `graph_constructor.k` for MTGODE, `cheb_k` and `embed_dim` for AGCRN) was
> performed when the network expanded to N=45 stations, keeping all
> N-invariant hyperparameters at their initial optima. This two-stage
> protocol was necessitated by the SAWS data delivery schedule."

This is the **truth** — no hedging, no papering over. The reviewer will
respect that we understood which knobs needed re-tuning and which didn't.

## Kill criteria (in case things go sideways)

- If a model's best trial after 30 iterations has val_loss > 2× the trainer's initial baseline, something is wrong — stop the sweep, investigate the model, don't just try more trials.
- If all trials for a model early-stop at ≤ 5 epochs, likely a training instability (learning rate too high, gradient explosion). Halve the lr range, restart.
- If wall clock exceeds 30 hours per model job, kill and requeue with `--n-trials 20`.
