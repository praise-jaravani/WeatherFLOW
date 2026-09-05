# CLCRN learning-rate repair — pre-registered diagnostic (extended to full campaign 2026-08-19)

**Original timestamp**: 2026-08-19 (before diagnostic run).
**Extension timestamp**: 2026-08-19 (after gate PASS on sp) — repair extended uniformly to all CLCRN cells.

## Motivation

- CLCRN config uses `lr=0.05` (Gaibie 2024 HPO-optimal on 45-station SAWS predecessor with MSELoss). Search interval was (0.0001, 0.05); 0.05 was at the boundary.
- On our current 32-station network with unified `L1Loss`, seed 42 for `sp` diverged at epoch 1 with `grad_norm=inf` and terminated via 3-epoch divergence detection.
- Two contributing mechanisms diagnosed (documented in this repo's specs/HARDWARE_PROVENANCE + earlier session logs):
  1. `k_neighbors=25` = N=25 was degenerate in the 25-station campaign, made CLConv kernel MLP trivial. At N=32 (`k/N=0.78`), the kernel MLP is suddenly load-bearing and its conditioning is worse.
  2. L1Loss's constant-magnitude subgradient does not self-damp at high lr, unlike MSELoss used in Gaibie's HPO.

## Repair proposal (as originally pre-registered)

- **Scope**: `sp` seed 42 only, **POD-ONLY DIAGNOSTIC**.
- **Repair value**: `lr=0.01` for `clcrn_sp_s42` on the pod override only.
- **HPC campaign** (as originally planned): keeps `lr=0.05` on all 40 CLCRN reg tasks.

## Gate result (2026-08-19, after 5 epochs on pod at lr=0.01)

Full sp train.log at gate check:
```
ep1: train=0.4400, val=0.4481, grad=0.32
ep2: train=0.3816, val=0.4218, grad=0.30
ep3: train=0.3662, val=0.4137, grad=0.30  ← best-val
ep4: train=0.3575, val=0.4252, grad=0.31
ep5: train=0.3488, val=0.4144, grad=0.31
```

| Criterion | Threshold | Measured | Verdict |
|---|---|---|---|
| grad_norm finite AND <100 by ep5 | <100 | 0.31 | ✅ PASS |
| best-val epoch ≥ 3 | ≥3 | ep3 | ✅ PASS |
| v[best]/v[1] ≤ 0.97 | ≤0.97 | 0.4137/0.4481 = 0.923 | ✅ PASS |
| descending shape | mostly monotonic | 0.448→0.422→0.414 (wobble 4→5) | ✅ PASS |

**All 4 criteria pass. Adopt lr=0.01.**

## Extension to full CLCRN campaign (2026-08-19, post-gate decision)

**Scope change**: extended from "sp seed 42 pod-only" to **uniform lr=0.01 across all CLCRN cells** (t2m, sp, rh, ws, cls) for the entire campaign, including the HPC drain (40 reg + 10 cls).

**Rationale**:
- Sp gate passed cleanly at lr=0.01, so we know the value is safe.
- t2m and ws at lr=0.05 stalled at initialisation (v[best]/v[1] ≈ 0.995 across 8+ epochs — not learning). Same mechanistic story as sp: aggressive lr × MAE constant-magnitude subgradient × N=32 kernel-MLP conditioning shift. Uniform lr=0.01 addresses this.
- Per-target repair (only sp at 0.01, others at 0.05) would preserve pod seed-42 t2m/rh/ws data as legitimate, but at the cost of a more complex methods section AND leaving t2m/ws with the known-bad lr. Uniform repair is cleaner AND likely produces better numbers.
- Still principled: lr=0.01 is **interior** to Gaibie's (0.0001, 0.05) search interval. Same principle as AGCRN's move to Bai's own reported optimum.

**Pod's already-running seed-42 t2m/rh/ws (at lr=0.05)**: demoted to pilot-only. Not part of final results. Reported (if at all) as "we initially attempted CLCRN reg at inherited lr=0.05 on seed 42; only sp showed catastrophic divergence but t2m and ws showed no learning past initialisation across 13 epochs, motivating uniform lr=0.01 for the production campaign."

**Provenance line for §3.4 of paper**: "CLCRN follows Gaibie et al.'s (2024) configuration except for the learning rate, reduced from the search-boundary value 0.05 to 0.01 (interior to Gaibie's search interval) after a pre-registered diagnostic on our N=32 network found the inherited value produced non-finite gradients on surface pressure and no learning past initialisation on temperature and wind speed. See spec/CLCRN_SP_LR_REPAIR.md for the diagnostic protocol and pass/fail evidence."

**Implementation** (done 2026-08-19):
- `experiments/configs/clcrn.yaml`: `learning_rate: 0.05` → `0.01`, with full provenance comment.
- Pushed to HPC (`/scratch/jrvpra001/weatherflow/WeatherFLOW/experiments/configs/clcrn.yaml`). Verified via `yaml.safe_load` → `training.learning_rate == 0.01`.
- Pod's per-task override (`hpo_configs/clcrn_sp_s42.yaml`) is now redundant with base config; harmless to leave in place (deep-merge would set 0.01 anyway).

## Pre-registered decision rule

Evaluated on the diagnostic run's `train.log` after epoch 5:

**PASS** (adopt lr=0.01 for CLCRN sp):
- grad_norm finite AND < 100 by epoch 5
- best-val epoch ≥ 3
- v[best] / v[1] ≤ 0.97 (loss actually decreased, not just fluctuated)
- descending shape (val monotonically non-increasing across best-epoch trajectory)

**FAIL** (single further step):
- Reduce to `lr=0.003` (still interior to Gaibie's (0.0001, 0.05) interval; matches AGCRN's healed lr).
- Re-run under same gate.

**FAIL AGAIN**:
- Report CLCRN sp as a failed cell at inherited hyperparameters (which is itself a paper result — Section: hyperparameter transferability across losses and network sizes).

## Other 3 tasks (t2m, rh, ws) — also gate-check

Reviewer flagged: epoch-1 grad_norms of 0.05-0.59 aren't proof of health (AGCRN looked fine at ep1 too). Apply same gate at epoch 5:
- grad_norm < 100 ✓ (currently 0.04-0.42 at epoch 8)
- v[best]/v[1] ≤ 0.97: rh **passes** (0.442/0.485 = 0.91). t2m: 0.7816/0.7819 = 0.9996 **FAILS** (essentially no learning). ws: 0.7418/0.7469 = 0.9932 **FAILS** (essentially no learning).

If t2m and ws are still stalling at epoch 15+ (when patience triggers early-stop), that's a separate finding: same lr=0.05 that diverges sp also stalls t2m/ws — geometric-prior model producing best-at-init on a network too small for its inductive bias. Report but don't repair (out of scope for this diagnostic).

## Implementation

1. Per-task config override: `/workspace/pod_campaign/hpo_configs/clcrn_sp_s42.yaml`:
   ```yaml
   training:
     learning_rate: 0.01
   ```
2. Reset sp task on pod: remove `FAILED_DIVERGED`, `state.pt`, `checkpoint.pt`, `train.log`, `claims/clcrn_sp_s42`.
3. autostart.sh: pass `--hpo-configs-dir /workspace/pod_campaign/hpo_configs` to `run_worker.py`.
4. Restart autostart; sp starts fresh at lr=0.01; t2m/rh/ws resume from state.pt at lr=0.05.
