# WEATHER-FLOW — local development targets
# Every target translates to a python cli.py or python scripts/... call.
# No additional logic lives here.
#
# Usage examples:
#   make run MODEL=mtgode DATASET=saws TASK=regression
#   make preprocess DATASET=saws
#   make baselines DATASET=saws TASK=regression
#   make verify
#   make test
#   make mlflow
#   make compare

PYTHON := python
CLI    := cli.py

# Defaults (override on the command line)
MODEL   ?= mtgode
DATASET ?= saws
TASK    ?= regression
RUNS    ?= 10

.PHONY: run preprocess baselines compare mlflow verify test clean help

## Run a single experiment locally
run:
	$(PYTHON) $(CLI) --model $(MODEL) --dataset $(DATASET) --task $(TASK) --runs $(RUNS)

## Run preprocessing pipeline for a dataset (no training)
preprocess:
	$(PYTHON) $(CLI) --model agcrn --dataset $(DATASET) --task regression --runs 0 --no-cache

## Run both baselines (AGCRN then CLCRN) back-to-back
baselines:
	$(PYTHON) scripts/run_baselines.py --dataset $(DATASET) --task $(TASK)

## Print cross-model comparison table from all outputs
compare:
	$(PYTHON) scripts/compare_results.py outputs/

## Launch MLflow UI in browser at localhost:5000
mlflow:
	mlflow ui --backend-store-uri sqlite:///tracking/mlflow.db --port 5000

## Verify all setup components (run this after setup, before implementation)
verify:
	$(PYTHON) scripts/verify_setup.py

## Run the test suite
test:
	pytest tests/ -v

## Remove Python cache files
clean:
	find . -type d -name __pycache__ -not -path "*/models/baselines/AGCRN/*" \
	  -not -path "*/models/baselines/CLCRN/*" \
	  -not -path "*/models/continuous_time/MTGODE/*" \
	  -not -path "*/models/continuous_time/ClimODE/*" \
	  -not -path "*/models/lightweight/STLGRU/*" \
	  -not -path "*/models/lightweight/Lite-STGNN/*" \
	  -not -path "*/models/diffusion/DiffSTG/*" \
	  -not -path "*/models/diffusion/USTD/*" \
	  | xargs rm -rf
	find . -name "*.pyc" -not -path "*/models/baselines/*" \
	  -not -path "*/models/continuous_time/*" \
	  -not -path "*/models/lightweight/*" \
	  -not -path "*/models/diffusion/*" | xargs rm -f

## Print available targets
help:
	@grep -E '^##' Makefile | sed 's/## /  /'
