# WEATHER-FLOW — local development targets
# Every target translates to a python cli.py or python scripts/... call.
# No additional logic lives here.
#
# Usage examples:
#   make run MODEL=mtgode DATASET=era5 TASK=regression
#   make preprocess DATASET=era5
#   make baselines DATASET=era5 TASK=regression
#   make ui                           — launch MLflow UI at localhost:5000
#   make verify
#   make test
#   make compare
#
# Environment:
#   make activate-env                 — print conda activation instructions
#   make sync-env                     — update conda env if dependencies changed
#   make build-container              — rebuild Singularity .sif (Linux/WSL2 only)
#   make upload-container HPC_USER=<uct-id>
#   make sync-hpc HPC_USER=<uct-id>  — build + upload in one step

# ── Paths ─────────────────────────────────────────────────────────────────────

CONDA_ENV  := weatherflow
CONDA_BASE := $(HOME)/miniconda3

# Use the conda env's python/mlflow binaries directly so targets work whether
# or not the caller has already run `conda activate weatherflow`.
PYTHON  := $(CONDA_BASE)/envs/$(CONDA_ENV)/bin/python
MLFLOW  := $(CONDA_BASE)/envs/$(CONDA_ENV)/bin/mlflow
PYTEST  := $(CONDA_BASE)/envs/$(CONDA_ENV)/bin/pytest
CONDA   := $(CONDA_BASE)/bin/conda

CLI     := cli.py

# ── Defaults (override on the command line) ───────────────────────────────────

MODEL      ?= agcrn
DATASET    ?= era5
TASK       ?= regression
RUNS       ?= 10
PORT       ?= 5000
HPC_USER   ?= $(shell whoami)
MAX_NODES  ?=   # leave blank for full graph; set to e.g. 50 for smoke tests

# Sentinel file — touched after a successful conda env update.
# Make compares its timestamp against requirements.txt and environment.yml
# and re-runs the update only when those files are newer.
CONDA_STAMP := .conda-env.stamp

.PHONY: run smoke preprocess baselines compare ui mlflow activate-env verify test test-all clean \
        sync-env build-container upload-container sync-hpc help

# ── Environment ───────────────────────────────────────────────────────────────

## Print conda activation instructions (conda activate can't run inside make)
activate-env:
	@echo ""
	@echo "  To activate the WeatherFLOW environment, run:"
	@echo ""
	@echo "    conda activate $(CONDA_ENV)"
	@echo ""
	@echo "  To verify the active python:"
	@echo ""
	@echo "    which python  →  should show .../envs/$(CONDA_ENV)/bin/python"
	@echo ""
	@echo "  Current Makefile python: $(PYTHON)"
	@$(PYTHON) --version

## Update conda env when requirements.txt or environment.yml has changed
$(CONDA_STAMP): requirements.txt environment.yml
	$(CONDA) env update -f environment.yml --prune
	touch $(CONDA_STAMP)

sync-env: $(CONDA_STAMP)

# ── MLflow UI ─────────────────────────────────────────────────────────────────

## Launch MLflow UI at http://localhost:<PORT> (default 5000). Ctrl-C to stop.
ui:
	@echo "Launching MLflow UI → http://localhost:$(PORT)"
	@echo "Ctrl-C to stop."
	$(MLFLOW) ui \
		--backend-store-uri sqlite:///tracking/mlflow.db \
		--port $(PORT)

## Alias: make mlflow = make ui
mlflow: ui

# ── Development targets ───────────────────────────────────────────────────────

## Run a single experiment locally
##   make run MODEL=agcrn DATASET=era5 TASK=regression RUNS=1
##   make run MODEL=agcrn DATASET=era5 TASK=regression RUNS=1 MAX_NODES=50  # smoke test
run:
	$(PYTHON) $(CLI) \
		--model $(MODEL) \
		--dataset $(DATASET) \
		--task $(TASK) \
		--runs $(RUNS) \
		$(if $(MAX_NODES),--max-nodes $(MAX_NODES),)

## Quick smoke test: 50 nodes, 1 seed, 2 epochs — proves the pipeline end-to-end
smoke:
	$(PYTHON) $(CLI) \
		--model $(MODEL) \
		--dataset $(DATASET) \
		--task $(TASK) \
		--runs 1 \
		--mlflow-off \
		--max-nodes 50 \
		--config.training.epochs 2 \
		--config.training.batch_size 16 \
		--config.training.early_stopping_patience 5

## Run preprocessing pipeline only (skips training)
preprocess:
	$(PYTHON) $(CLI) \
		--model agcrn \
		--dataset $(DATASET) \
		--task $(TASK) \
		--runs 1 \
		--mlflow-off \
		--config.training.epochs 0 \
		--no-cache

## Run both baselines (AGCRN then CLCRN) back-to-back
baselines:
	$(PYTHON) scripts/run_baselines.py \
		--dataset $(DATASET) \
		--task $(TASK) \
		--runs $(RUNS)

## Print cross-model comparison table from all outputs
compare:
	$(PYTHON) scripts/compare_results.py outputs/

## Verify all setup components (run this after setup, before implementation)
verify:
	$(PYTHON) scripts/verify_setup.py

## Run the test suite (fast tests only; excludes slow ODE integration tests)
test:
	$(PYTEST) tests/ -v -m "not slow"

## Run the full test suite including slow tests (ODE integration forward passes)
test-all:
	$(PYTEST) tests/ -v

## Remove Python cache files (excludes cloned model repos)
clean:
	find . -type d -name __pycache__ \
	  -not -path "*/models/baselines/AGCRN/*" \
	  -not -path "*/models/baselines/CLCRN/*" \
	  -not -path "*/models/continuous_time/MTGODE/*" \
	  -not -path "*/models/continuous_time/ClimODE/*" \
	  -not -path "*/models/lightweight/STLGRU/*" \
	  -not -path "*/models/lightweight/Lite-STGNN/*" \
	  -not -path "*/models/diffusion/DiffSTG/*" \
	  -not -path "*/models/diffusion/USTD/*" \
	  | xargs rm -rf
	find . -name "*.pyc" \
	  -not -path "*/models/baselines/*" \
	  -not -path "*/models/continuous_time/*" \
	  -not -path "*/models/lightweight/*" \
	  -not -path "*/models/diffusion/*" | xargs rm -f

# ── Container / HPC ──────────────────────────────────────────────────────────

## Rebuild Singularity container from container.def (Linux/WSL2 only)
build-container:
	singularity build --fakeroot container.sif container.def

## Upload container.sif to UCT HPC (set HPC_USER=<your-uct-username>)
upload-container:
	scp container.sif $(HPC_USER)@hpc.uct.ac.za:/home/$(HPC_USER)/weatherflow/container.sif

## Rebuild and upload in one step
sync-hpc: build-container upload-container

# ── Help ──────────────────────────────────────────────────────────────────────

## Print all available targets
help:
	@echo ""
	@echo "WEATHER-FLOW Makefile targets:"
	@echo ""
	@grep -E '^##' Makefile | sed 's/## /  /'
	@echo ""
