# tsfm-bench

Head-to-head benchmark of zero-shot time-series foundation models against tuned
LSTM / ARIMA / LightGBM baselines on small public series, with wall-clock cost.

Every number this repository reports is measured. Nothing is simulated,
estimated, or copied from another paper except the published competition
benchmarks used as acceptance tests.

## Groups

Thirteen groups, using the official competition hold-outs and horizons.

| source | groups | series | horizon |
|---|---|---|---|
| M4 | Yearly, Quarterly, Monthly, Daily | 500 random per frequency (seed 20260907) | 6 / 8 / 18 / 14 |
| M4 | Weekly, Hourly | all (359 / 414) | 13 / 48 |
| M3 | Yearly, Quarterly, Monthly, Other | all 3,003 | 6 / 8 / 18 / 8 |
| Tourism | Yearly, Quarterly, Monthly | all 1,311 | 4 / 8 / 24 |

`make describe` prints the measured counts and series lengths.

## Metrics

- **MASE** and **sMAPE** exactly as defined in the official M4 evaluation code
  (`Benchmarks and Evaluation.R` in Mcompetitions/M4-methods).
- **WQL** over the nine deciles, Chronos definition, normalised by the sum of
  absolute target values.
- **80% coverage** from the 0.1 and 0.9 quantiles.
- **Wall-clock seconds per 1,000 forecasts**, recorded per model per group in
  `results/timing.jsonl` together with the platform string.

## Acceptance test

`make validate` runs seasonal naive over the full M4 (all 100,000 series) and
compares against the published M4 benchmark table. It must reproduce the
published sNaive row before any model result is trusted:

| frequency | sMAPE (measured / published) | MASE (measured / published) |
|---|---|---|
| Yearly | 16.342 / 16.34 | 3.974 / 3.974 |
| Quarterly | 12.521 / 12.52 | 1.602 / 1.602 |
| Monthly | 15.988 / 15.99 | 1.260 / 1.260 |
| Weekly | 9.161 / 9.161 | 2.777 / 2.777 |
| Daily | 3.045 / 3.045 | 3.278 / 3.278 |
| Hourly | 13.912 / 13.91 | 1.193 / 1.193 |
| **Total** | **14.657 / 14.66** | |

## Usage

```
make data          # clone M4 / M3 / Tourism
make validate      # acceptance test
make foundation    # zero-shot foundation models
make baselines     # statistical + LightGBM
make trained       # LSTM, N-BEATS
make analyze       # regenerate all tables and figures
make paper         # compile the PDF
```

## Layout

```
tsfm_bench/         data loaders, metrics, model wrappers
run.py              run models over groups -> results/metrics/*.csv, timing.jsonl
analyze.py          -> results/tables/*.csv|tex, results/figures/*.pdf
scripts/            data fetch, acceptance test
paper/              main.tex and the compiled PDF
results/EXCLUDED.md any model that could not be run, and exactly why
```
