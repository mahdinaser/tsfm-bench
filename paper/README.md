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
- No LaTeX toolchain was available on the machine where this was drafted, so
  `main.tex` has been checked structurally (citations resolve, inputs exist,
  macros defined, braces balance) but has never been compiled.
