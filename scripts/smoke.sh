#!/usr/bin/env bash
# Smoke test: every model on two small groups, results to results/smoke/.
# Run from the repo root:  bash scripts/smoke.sh   (log: results/smoke/smoke.log)
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p results/smoke
LOG=results/smoke/smoke.log
# GROUPS is a bash builtin (the shell's group IDs), so ${GROUPS:-...} is always "set" — the smoke run was passing gid 20 as a dataset group.
SMOKE_GROUPS="${SMOKE_GROUPS:-m3_other tourism_yearly}"
MODELS="${MODELS:-SeasonalNaive Theta AutoETS AutoARIMA LightGBM LSTM NBEATS ChronosBoltSmall ChronosBoltBase Chronos2 TimesFM TimesFM3 Moirai2 Sundial}"
{
  echo "=== smoke $(date -u +%FT%TZ) ==="
  echo "groups: $SMOKE_GROUPS"
  echo "models: $MODELS"
  python -c "import torch;print('torch',torch.__version__,'mps',torch.backends.mps.is_available())"
  for m in $MODELS; do
    echo; echo "##### $m"
    TSFM_BENCH_DEBUG=1 python run.py --models "$m" --groups $SMOKE_GROUPS --out results/smoke 2>&1 \
      | grep -vE "^\s*$|Lightning|GPU available|TPU available|HPU available|IPU available|warnings.warn|UserWarning|FutureWarning|Seed set|LOCAL_RANK|it/s\]|Sanity Checking|Predicting|Epoch" 
  done
  echo; echo "=== summary ==="
  python - <<'PY'
import json, pandas as pd
rows=[json.loads(l) for l in open("results/smoke/timing.jsonl")]
df=pd.DataFrame(rows)
cols=[c for c in ["model","group","mase","smape","wql","coverage80","wall_seconds","load_seconds","device","n_params"] if c in df.columns]
pd.set_option("display.width",200); pd.set_option("display.max_columns",20)
print(df[cols].round(3).to_string(index=False))
PY
  echo "=== done $(date -u +%FT%TZ) ==="
} 2>&1 | tee "$LOG"
