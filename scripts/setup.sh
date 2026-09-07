#!/usr/bin/env bash
# One-shot setup: environment, packages, data, acceptance test.
# Run from the repo root:   bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "=== tsfm-bench setup ==="
echo "repo: $ROOT"

# ---------------------------------------------------------------- machine
echo
echo "--- machine ---"
OS_VER="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
BRAND="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Chip|Processor Name/{print $2; exit}')"
CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
PCORES="$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo '')"
ECORES="$(sysctl -n hw.perflevel1.logicalcpu 2>/dev/null || echo '')"
RAM_B="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
RAM_GB="$(python3 -c "print(round($RAM_B/1024**3,1))" 2>/dev/null || echo unknown)"
MODEL="$(sysctl -n hw.model 2>/dev/null || echo unknown)"
ARCH="$(uname -m)"
echo "macOS        : $OS_VER"
echo "model        : $MODEL"
echo "chip         : ${BRAND:-$CHIP}"
echo "arch         : $ARCH"
echo "logical cores: $CORES  (P:${PCORES:-?} E:${ECORES:-?})"
echo "RAM          : ${RAM_GB} GB"
echo "homebrew     : $(command -v brew || echo MISSING)"
echo "conda        : $(command -v conda || echo MISSING)"

# ---------------------------------------------------------------- python
echo
echo "--- python ---"
PYBIN=""
for c in python3.12 python3.11 python3.10; do
  if command -v "$c" >/dev/null 2>&1; then PYBIN="$c"; break; fi
done
if [ -z "$PYBIN" ]; then
  V="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo none)"
  case "$V" in
    3.10|3.11|3.12) PYBIN=python3 ;;
    *) echo "ERROR: need Python 3.10-3.12; found $V."
       echo "Install one:  brew install python@3.12"
       exit 1 ;;
  esac
fi
echo "using: $PYBIN ($($PYBIN -V))"

if [ ! -d .venv ]; then "$PYBIN" -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -qU pip wheel setuptools

# ---------------------------------------------------------------- packages
echo
echo "--- installing packages (this takes a while) ---"
python -m pip install -q \
  numpy pandas scipy matplotlib rdata xarray \
  torch \
  statsforecast mlforecast lightgbm neuralforecast \
  transformers huggingface_hub \
  "chronos-forecasting>=2" \
  "timesfm[torch]" \
  uni2ts \
  || { echo; echo "Some packages failed. Rerun this script; if one package is the"; \
       echo "problem, install the rest and note the failure in results/EXCLUDED.md."; }

python -m pip freeze > requirements-lock.txt
echo "pinned $(wc -l < requirements-lock.txt) packages -> requirements-lock.txt"

echo
echo "--- torch backend ---"
python - <<'PY'
import json, platform, torch, os
info = {
    "torch": torch.__version__,
    "mps_available": bool(torch.backends.mps.is_available()),
    "mps_built": bool(torch.backends.mps.is_built()),
    "cuda_available": bool(torch.cuda.is_available()),
    "threads": torch.get_num_threads(),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "python": platform.python_version(),
}
print(json.dumps(info, indent=2))
os.makedirs("results", exist_ok=True)
json.dump(info, open("results/torch_env.json", "w"), indent=2)
PY

# record the machine facts the paper's cost section needs
python - <<PY
import json, os
os.makedirs("results", exist_ok=True)
json.dump({
  "macos": "$OS_VER", "model": "$MODEL", "chip": "${BRAND:-$CHIP}",
  "arch": "$ARCH", "logical_cores": "$CORES",
  "perf_cores": "${PCORES:-}", "eff_cores": "${ECORES:-}",
  "ram_gb": "$RAM_GB",
}, open("results/machine.json","w"), indent=2)
print("wrote results/machine.json")
PY

# ---------------------------------------------------------------- data
echo
echo "--- data ---"
mkdir -p data
if [ ! -d data/M4-methods ]; then
  git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/Mcompetitions/M4-methods.git data/M4-methods
  ( cd data/M4-methods && git sparse-checkout set Dataset )
else echo "M4 already present"; fi
[ -d data/Mcomp ] || git clone --depth 1 -q https://github.com/robjhyndman/Mcomp.git data/Mcomp
[ -d data/Tcomp ] || git clone --depth 1 -q https://github.com/ellisp/Tcomp-r-package.git data/Tcomp

echo
echo "--- groups (measured) ---"
python -c "from tsfm_bench.data import describe_groups; print(describe_groups().to_string(index=False))"

echo
echo "--- acceptance test: seasonal naive vs published M4 ---"
python scripts/validate_snaive.py

echo
echo "=== setup done ==="
echo "Next:  bash scripts/smoke.sh"
