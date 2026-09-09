# arXiv submission — everything the form asks for

Upload `arxiv-submission.tar.gz` (336 KB). It contains `main.tex`, `main.bbl`,
`neurips_2026.sty`, `tables/*.tex` and `figures/*.png` — and nothing else.
Verified: it compiles from a clean directory with two `pdflatex` passes, zero
errors, zero undefined references. `refs.bib` is deliberately **not** included,
because arXiv does not run BibTeX; the pre-built `main.bbl` is what it uses.

Keep the `preprint` option in `main.tex`. It is what shows the author block,
which is right for arXiv. Remove it only for the double-blind NeurIPS
submission.

---

## Title

A Later Test Set Is Not a New Domain: Pretraining Familiarity Survives a
Contamination-Free Hold-Out

## Authors

Mahdi Naser Moghadasi (BrightMind AI); Faezeh Ghaderi (University of Texas at
Arlington)

## Abstract (plain text — arXiv rejects LaTeX macros here)

Time-series foundation models are evaluated almost exclusively on public
archives that predate them, so a strong score cannot be separated from having
seen the test set during pretraining. The obvious remedy is a hold-out that
postdates the models. We build one: thirteen forecasters -- four classical,
three trained per dataset, six pretrained -- on seven groups drawn from five
domains, every observation published after the last model was released, and
every dataset rebuildable without an API key. Under this protocol pretrained
models win 5 of 7 groups, lose one to a Theta baseline, and on daily exchange
rates are indistinguishable from a seasonal naive forecast, along with every
other method tested.

We then ask what separates the wins from the losses, and report a negative
result: the two intrinsic properties one would reach for -- seasonal strength
and spectral entropy, measured on the input window -- do not account for the
pattern, and seasonal strength is if anything negatively associated with the
advantage. What does track it is corpus familiarity. Our largest gain (28%
lower MASE than the best classical method, on weekly Wikipedia pageviews) falls
on Wikipedia pageviews, the domain TimesFM's authors describe as the bulk of
its pretraining corpus, at the same granularities and differing only in time
window. Within the pretrained family, where every model forecasts identical
series so that series difficulty cancels, the TimesFM family outranks the
Chronos family by -0.53 ranks on Wikipedia against -0.09 everywhere else (1,500
vs. 754 series, Mann-Whitney p < 1e-5). We conclude that a temporal hold-out
removes memorisation of a window but not familiarity with a domain, that
benchmarks therefore need domain hold-outs stated relative to disclosed
corpora, and that the practitioner's question is less which model is better
than whether their domain is one the model was raised on.

## Categories

- Primary: **cs.LG** (Machine Learning)
- Cross-list: **stat.ML** (Machine Learning, Statistics)
- Consider also: cs.AI

## Comments field

11 pages, 2 figures, 5 tables. Code, data fetchers and per-series results:
https://github.com/mahdinaser/tsfm-bench

## Licence

"arXiv.org perpetual, non-exclusive license" is the usual choice and keeps
future journal options open. CC BY 4.0 is the more open alternative; pick
deliberately, because the choice cannot be narrowed later.

---

## Before you press submit

1. **Make the GitHub repository public.** It is private right now, so the URL
   printed in the paper and in the Comments field will 404 for every reader.
2. **Add a licence file to the repository.** The paper's Dataset Documentation
   section states that the code "carries the repository's licence", and there
   is not one yet.
3. **Endorsement.** arXiv requires an endorsement for a first submission to
   cs.LG from an author with no prior arXiv history in that category. If the
   submission is held for endorsement, that is the reason, and a colleague who
   has published in cs.LG can endorse.
4. Check the author order and affiliations are how you and your co-author want
   them.
