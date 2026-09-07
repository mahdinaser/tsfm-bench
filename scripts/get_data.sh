#!/usr/bin/env bash
# Fetch the three competition datasets exactly as the paper describes.
set -euo pipefail
cd "$(dirname "$0")/../data"

git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/Mcompetitions/M4-methods.git M4-methods
( cd M4-methods && git sparse-checkout set Dataset )

git clone --depth 1 https://github.com/robjhyndman/Mcomp.git Mcomp
git clone --depth 1 https://github.com/ellisp/Tcomp-r-package.git Tcomp
