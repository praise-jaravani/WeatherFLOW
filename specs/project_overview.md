# WeatherFLOW — Project Overview

**University of Cape Town — CSC4019Z Honours Project 2026**  
**Team:** Génevieve Chikwanha, Zoleka Fuyana, Praise Jaravani  
**Supervisor:** Dr. Deshen Moodley

---

## What

Unified benchmarking framework comparing 8 STGNN architectures on South African weather data.

**Models:**
| Family | Models |
|---|---|
| Baselines | AGCRN, CLCRN |
| Continuous-time | MTGODE, ClimODE |
| Lightweight | STLGRU, Lite-STGNN |
| Diffusion | DiffSTG, USTD |

**Workstreams:**
- Praise → continuous-time (MTGODE, ClimODE)
- Zoleka → lightweight (STLGRU, Lite-STGNN)
- Génevieve → diffusion (DiffSTG, USTD)

## Datasets

| Dataset | Status | Notes |
|---|---|---|
| ERA5 | Downloading (2010–2023) | Hourly reanalysis, 6 variables, SA bounding box |
| SAWS | Not yet received | Station data, same variables where possible |

## Tasks

- **Phase 1 (regression):** Predict future weather values. Metrics: MAE, RMSE, NRMSE, VPT.
- **Phase 2 (classification):** Predict extreme events (95th-percentile threshold). Metrics: Precision, Recall, F1, AUC-ROC.

## Execution environments

| Environment | Used for |
|---|---|
| Local conda env (`weatherflow`) | Development, preprocessing, debugging (CPU) |
| UCT HPC + Singularity container | All full training runs (A100 GPU) |

## Key deadlines

- GPU partition access email: before **2 June 2026** (email eresearch@uct.ac.za, CC Dr. Moodley)

## Repo

`git@github.com:praise-jaravani/WeatherFLOW.git`
