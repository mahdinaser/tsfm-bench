#!/bin/bash
# The post-cutoff hold-out run: every model against every 2026 group.
#
# Ordered cheapest-first on purpose. AutoARIMA is 99% of the wall clock (4.3
# hours on one hourly group alone, against 6 seconds for Chronos-2), so running
# it last means a crash or an interrupt still leaves a complete set of results
# for everything else. run.py skips any cell whose CSV already exists, so this
# is safe to re-run.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export TSFM_BENCH_ORIGIN=${TSFM_BENCH_ORIGIN:-2026-01-01}
OUT=results/holdout_2026
# NOT named GROUPS: bash pre-populates that as a read-only array of the
# caller's unix group ids, so the run silently received "20" as a dataset name.
# The same collision already cost a cycle in scripts/smoke.sh.
RUN_GROUPS=${RUN_GROUPS:-"airquality_hourly energy_hourly fx_daily"}

FAST="SeasonalNaive Theta LightGBM ChronosBoltSmall ChronosBoltBase Chronos2 TimesFM TimesFM3 Moirai2 LSTM NBEATS AutoETS"
for m in $FAST; do
  python run.py --models "$m" --groups $RUN_GROUPS --out $OUT 2>&1 | grep -v "^\[skip\]"
done
python run.py --models AutoARIMA --groups $RUN_GROUPS --out $OUT 2>&1 | grep -v "^\[skip\]"
echo "DONE $(date)"
