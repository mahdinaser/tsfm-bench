PY ?= python3

.PHONY: data describe validate baselines trained foundation analyze paper all freeze

data:            ## clone the three competition datasets
	bash scripts/get_data.sh

describe:        ## measured summary of all 13 groups
	$(PY) -c "from tsfm_bench.data import describe_groups; print(describe_groups().to_string(index=False))"

validate:        ## acceptance test: seasonal naive vs published M4 table
	$(PY) scripts/validate_snaive.py

smoke:           ## every model on two small groups -> results/smoke/
	bash scripts/smoke.sh

foundation:
	$(PY) run.py --models ChronosBoltSmall ChronosBoltBase Chronos2 TimesFM TimesFM3 Moirai2 Sundial --groups all

baselines:
	$(PY) run.py --models SeasonalNaive Theta AutoETS AutoARIMA LightGBM --groups all

trained:
	$(PY) run.py --models LSTM NBEATS --groups all

analyze:         ## regenerate every table and figure from results/metrics
	$(PY) analyze.py

paper: analyze
	cd paper && latexmk -pdf main.tex

all: data validate foundation baselines trained analyze paper

freeze:          ## pin the versions that were actually used
	$(PY) -m pip freeze > requirements-lock.txt
