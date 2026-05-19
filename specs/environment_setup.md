# Environment Setup Guide

## The model

```
requirements.txt          ← single source of truth for package versions
      ↑ referenced by
      ├── environment.yml  → local conda env (development, CPU)
      └── container.def    → Singularity container (HPC, A100 GPU)
```

Whenever `requirements.txt` changes, **all three** must be updated together:
1. `conda env update -f environment.yml --prune`
2. `singularity build --fakeroot container.sif container.def`
3. Re-upload `container.sif` to HPC

## Local setup (first time)

### 1. Install Miniconda

```bash
# Linux / WSL2
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, then restart shell
```

### 2. Create the conda environment

```bash
cd /path/to/Weatherflow
conda env create -f environment.yml
```

### 3. Activate (every session)

```bash
conda activate weatherflow
```

### 4. Verify

```bash
make verify
```

## Key packages and why they're pinned

| Package | Minimum | Reason |
|---|---|---|
| `cdsapi` | 0.7.0 | New CDS endpoint (post Sept 2024); 0.6.x will not work |
| `attrs` | 22.0.0 | Required by cdsapi 0.7.x; system-packaged version too old |
| `torch` | 2.1.0 | Must include sm_80 (A100) in CUDA arch list — use NGC image on HPC |
| `netCDF4` | 1.6.0 | Reading inner NetCDF4 files from ERA5 ZIP archives |

## HPC container

Build once on Linux/WSL2:
```bash
singularity build --fakeroot container.sif container.def
```

Verify on HPC after upload:
```bash
singularity exec --nv container.sif python -c "import torch; print(torch.cuda.is_available())"
# Must print True
```

If False: select a different NGC image tag in `container.def` (must have sm_80 support).

## What NOT to do

- Never `pip install` a package outside the activated conda env
- Never commit `container.sif` (it's gitignored; it's several GB)
- Never commit `~/.cdsapirc` (contains personal API key)
- Never run `conda env create` without first activating the env for that session

## Current status (2026-05-19)

- Miniconda installed at `~/miniconda3`
- `weatherflow` conda env created and fully isolated (PYTHONNOUSERSITE=1 via activate hook)
- All packages installed inside env: torch 2.12.0+cu130, PyG 2.7.0, xarray 2025.6.1, etc.
- Apptainer: pending install (`sudo apt-get install -y apptainer` via ppa:apptainer/ppa)
- Global `~/.local` packages: still present but bypassed by the conda env activation hook

## PYTHONNOUSERSITE isolation

The conda env's activate hook (`etc/conda/activate.d/env_vars.sh`) sets `PYTHONNOUSERSITE=1`,
which prevents Python from including `~/.local/lib/python3.10/site-packages` in its path.
This ensures the conda env is fully isolated from any globally-installed packages.
When running without activating the env (e.g. raw `python3`), the global packages are visible
as before — this is intentional and does not affect the project.
