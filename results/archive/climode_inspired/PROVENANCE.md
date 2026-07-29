# PROVENANCE — "ClimODE-inspired" archived summaries

## What this folder contains

Two aggregated summary JSONs from runs of an internal model that was, until
2026-07-26, called "ClimODE" in code and docs:

- `climode_regression_summary.json` — n=10 seeds, Phase 1 regression
- `climode_classification_summary.json` — n=6 seeds, Phase 2 classification (partial)

These summaries have been moved out of the live Phase-1 / Phase-2 results
surfaces so no reporting pipeline accidentally treats them as measurements of
the published ClimODE architecture. They are preserved here as archival record.

## What the archived model IS

A hand-written physics-motivated **advection graph-ODE** with the following
dynamics:

```
dX/dt = -gate(X) ⊙ (A · X) + emission(X)
```

- `gate(X)`: 2-layer MLP → sigmoid (learned per-node "velocity magnitude")
- `A · X`: fixed graph-diffusion operator using the preprocessing adjacency
  (Gaussian-weighted, thresholded at 300 km)
- `emission(X)`: 2-layer MLP correction / source term
- GRU encoder compresses the T-step input window into an initial ODE state
- torchdiffeq.odeint / odeint_adjoint integrates the ODE forward H steps
- **~17K trainable parameters**

Reference: [models/continuous_time/climode_model.py](../../../models/continuous_time/climode_model.py)
(class name kept as `ClimODEModel` for CLI/config backwards-compatibility with
the archived runs; docstring header carries the same naming note as this file).

## What the archived model is NOT

It is **not** ClimODE (Verma et al., ICLR 2024,
[Aalto-QuML/ClimODE](https://github.com/Aalto-QuML/ClimODE)). The upstream repo
is present as a git submodule at `models/continuous_time/ClimODE/` but zero
code from it was imported by our model. Substantive differences:

| Component | Verma et al.'s ClimODE | Archived "ClimODE-inspired" |
|-----------|------------------------|-----------------------------|
| Continuity equation | Second-order neural continuity equation (transport + compression terms) | First-order advection-only |
| Flow velocity | Local convs (ResNet) + long-range KQV attention | 2-layer sigmoid gate |
| Emission | Gaussian emission network → μ + σ² (uncertainty quantification) | Deterministic 2-layer MLP correction term |
| Initial velocity | Penalised least-squares inference from past frames (Eq. 10 of paper) | None (GRU-encoded latent) |
| Spatial structure | Gridded H×W with `torch.gradient()` for spatial derivatives | Graph diffusion via adjacency matrix |
| Training objective | Gaussian negative log-likelihood over (μ, σ²) | MAE / WBCE (our shared harness losses) |
| Parameters | ≈ several million (Table 1 of Verma et al. shows 28M for their WeatherBench global run; the *regional* variant we care about for SA — `ClimODE_uncertain_region` — builds to ~2.8M params at K=8, H=27, W=37, i.e. two orders of magnitude larger than the archived model, not four) | ≈ 17K |

Verma et al.'s own ablation shows the attention and emission components
contribute the largest gains — so their absence is not cosmetic.

## Architecture-family framing (recommended paper positioning)

The archived model belongs to the family of **station-graph neural ODEs with
physics-inspired transport operators**. The closest published relative is
**AirPhyNet** (Hettige et al., ICLR 2024): station-graph neural ODE with
advection/diffusion transport operators applied over an adjacency-defined
graph, using an emission-style correction term. Same design family, same
"advection-inspired, graph-native" positioning.

If these numbers ever come back into the paper (see Status below), the honest
framing is: *"advection-motivated graph ODE in the AirPhyNet family, ~17K
parameters, evaluated on the same harness as [other models]."* Do NOT position
it as ClimODE — the name is earned only by a faithful reimplementation of
Verma et al.'s architecture. That work is tracked separately as the
`climode_faithful` integration (Phase-3 spike).

## Status

- **Archived** pending faithful ClimODE evaluation.
- The 10 regression seeds and 6 classification seeds in these summaries are
  **valid measurements of this specific model** — they may return as an
  efficiency / ablation appendix contribution (e.g. "how well does a cheap
  advection prior alone do on this benchmark") but they must not appear in
  headline tables under the name ClimODE.
- Reference commits:
  - **267baba** — renamed "ClimODE" → "ClimODE-inspired" throughout
    human-facing docs; added strong naming notes to code files.
  - (This commit) — moved summaries here and shrank the live matrix to
    3 core models (AGCRN, CLCRN, MTGODE) until `climode_faithful` produces
    real numbers.
- Runs on disk (`outputs/climode_era5_*/`), checkpoints, MLflow entries, and
  per-seed run directories at `results/phase{1,2}_*/runs/climode_seed_*/`
  remain intact — they are the archival record, not something to delete.
