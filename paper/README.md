# Paper build

```
python scripts/make_tables.py     # regenerates tables/*.tex and figures.json
cd paper && latexmk -pdf main.tex
```

`make_tables.py` writes every table and, in `tables/claims.tex`, every number
the prose quotes as a LaTeX macro. Nothing in `main.tex` contains a
hand-typed result. If a cell is re-run, re-run the script and the sentences
follow the tables automatically.

## Before submission

- `refs.bib` has `TODO` notes on every arXiv identifier, volume and page range.
  They are deliberately not filled in from memory: an invented identifier looks
  authoritative and cannot be caught by a reader. Resolve each against the real
  record.
- The pretraining-cutoff table promised in §3.1 (`docs/dataset-plan.md`) still
  needs the supplement written out.
- The bibliography's `TODO` notes live inside `note` fields and therefore
  **print in the reference list**. That is deliberate while drafting; it also
  means the paper cannot be submitted without noticing them.

## Toolchain

TinyTeX at `~/Library/TinyTeX` (user-level, no sudo, ~200MB — MacTeX is 5GB and
needs an admin password). `paper/build.sh` runs the whole chain. Packages added
beyond the base install: `microtype booktabs natbib geometry hyperref amsmath
graphics caption`.

Note for anyone editing `refs.bib`: BibTeX has no `%` comment syntax. An inline
`%` note after a field silently swallows the field that follows it, which is how
the first build produced citations with no year.
