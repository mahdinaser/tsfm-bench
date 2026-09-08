#!/bin/bash
# Repeat the stochastic models under different seeds.
#
# Only three models here. Theta, AutoETS and AutoARIMA are deterministic given
# the data, and the pretrained models run at fixed weights with greedy decoding,
# so re-seeding them would produce identical files and burn hours proving it.
# scripts/check_determinism.py is what actually verifies that claim rather than
# assuming it.
#
# Kept separate from run_holdout.sh so the headline table is never waiting on
# the variance estimate.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export TSFM_BENCH_ORIGIN=${TSFM_BENCH_ORIGIN:-2026-01-01}
OUT=results/holdout_2026
RUN_GROUPS=${RUN_GROUPS:-"wiki_daily wiki_weekly wiki_monthly weather_hourly airquality_hourly energy_hourly fx_daily"}
STOCHASTIC="LightGBM LSTM NBEATS"
# The default seed already ran as part of the main table; these are the extras.
SEEDS=${SEEDS:-"20260908 20260909"}

for s in $SEEDS; do
  for m in $STOCHASTIC; do
    TSFM_BENCH_SEED=$s python run.py --models "$m" --groups $RUN_GROUPS --out $OUT 2>&1 | grep -v "^\[skip\]"
  done
done
echo "SEEDS DONE $(date)"
