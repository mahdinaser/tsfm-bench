#!/bin/bash
# Rebuild the paper from the results. Needs TinyTeX (installed at
# ~/Library/TinyTeX; no sudo, unlike MacTeX) plus the packages below.
set -e
cd "$(dirname "$0")/.."
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
source .venv/bin/activate

python scripts/make_tables.py
python scripts/significance.py --results results/holdout_2026 >/dev/null
# features.py is slow (STL over every group) and its output changes only when
# the data does, so it is not run here by default:
#   python scripts/features.py
python scripts/corpus_familiarity.py >/dev/null
python scripts/make_figures.py >/dev/null
python scripts/make_croissant.py >/dev/null

cd paper
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null || true
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "built paper/main.pdf"
grep -c "Warning" main.log || true
