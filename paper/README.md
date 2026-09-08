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

- `refs.bib` is verified against arXiv abstract pages, PMLR proceedings and
  publisher DOI records. One field remains unchecked and says so in its own
  `note`: the arXiv identifier for Moirai (the PMLR record is confirmed).
- The M5 entry is deliberately titled "M5 Accuracy Competition: Results,
  Findings, and Conclusions" — no leading "The", Oxford comma. The common form
  is a miscitation.
- Submission blocker: the paper builds with `\usepackage[eandd, preprint]`.
  The `preprint` option is what shows the author block; the Evaluations &
  Datasets track is double-blind, so it MUST be removed before submitting.
- `croissant.json` still has TODO values for the repository URL and citation.
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
