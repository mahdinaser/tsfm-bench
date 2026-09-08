# Dataset plan — evaluation the foundation models cannot have seen

Goal of the paper: be the reference comparison of time-series foundation
models against tuned classical and trained baselines. The one thing every
existing comparison (including ours today) gets attacked on is contamination:
M3, M4 and Tourism are inside the pretraining corpora of Chronos, TimesFM and
Moirai. This plan replaces "hope they didn't see it" with two holdouts that
are verifiable by construction.

## 1. Temporal holdout — same domains, test windows after every cutoff

Keep the public benchmarks for continuity with prior work, but score the
foundation models only on windows that start after the latest pretraining
cutoff among the models compared. For a model with cutoff `c`, any forecast
origin `t ≥ c` is uncontaminated; for the paper we use one common origin
`t* = max(c_i) + 1 month` so every model is scored on identical data.

| Model (HF repo) | Pretraining corpus | Stated data cutoff | Release (upper bound) | Source |
|---|---|---|---|---|
| Chronos-Bolt small/base (`amazon/chronos-bolt-*`) | Chronos corpus: 28 public datasets (Monash, M-competitions, Kaggle; Appendix B, Table 3 of arXiv:2403.07815) + TSMixup augmentations + KernelSynth synthetic; "nearly 100 billion time series observations" | Not stated | 26 Nov 2024 (chronos-forecasting README news) | HF model card `amazon/chronos-bolt-base`; github.com/amazon-science/chronos-forecasting |
| Chronos-2 (`amazon/chronos-2`) | "Subset of Chronos Datasets (excluding test portion of datasets that overlap with GIFT-Eval)" + "Subset of GIFT-Eval Pretrain" + synthetic univariate/multivariate (TSI, TCM generators). §5.1: "the corpus does not overlap with the test portions of any GIFT-Eval task", with "partial overlap with the training portions of some GIFT-Eval datasets" | Not stated | arXiv 17 Oct 2025; released 20 Oct 2025 | HF model card; arXiv:2510.15821 §4.1, §4.2, §5.1, Table 6 (App. A) |
| TimesFM 2.5 (`google/timesfm-2.5-200m-pytorch`) | "GiftEvalPretrain, Wikimedia Pageviews, cutoff Nov 2023 and Google Trends top queries, cutoff EoY 2022", plus synthetic/augmented | **Wikimedia pageviews: Nov 2023; Google Trends: end 2022** (only dated sources) | HF card updated 2 Oct 2025 | HF model card (Pretraining section); arXiv:2310.10688 |
| TimesFM 3.0 (`google/timesfm-3.0-pytorch`, 0.3B) | "GiftEvalPretrain excluding the datasets that overlap with fev-bench", "Wikipedia Pageviews, cutoff Nov 2023", "Google Trends top queries, cutoff EoY 2022", synthetic/augmented | **Nov 2023 / end 2022** as above | Release date not shown on card — verify | HF model card |
| Moirai 2.0 R-small (`Salesforce/moirai-2.0-R-small`) | §4: non-leaking GIFT-Eval Pretrain (3.25M series, 230B obs) + GIFT-Eval TrainTest train split (144K series) + Chronos-Mixup from non-leaking Chronos subsets (30M series) + KernelSynth (11M series) + internal Salesforce CloudOps telemetry (~2.15M series, "approximately one year, starting from January 2024") | Only the internal data is dated: **Jan 2024 → ~Jan 2025** | arXiv 12 Nov 2025 | HF model card; arXiv:2511.11698 §4 |
| Sundial base 128M (`thuml/sundial-base-128m`) | TimeBench, ~10^12 points; draws on public collections incl. LOTSA, Chronos datasets, UTSD, ERA5 meteorology; KernelSynth ≈0.05%. §5.1: "All evaluated datasets are excluded from the pre-training dataset" (TSLib, GIFT-Eval, fev) | Not stated | arXiv v1 2 Feb 2025; trillion-scale HF release May 2025 | HF model card; github.com/thuml/Sundial; arXiv:2502.00816 §4.2, App. A |

Two facts the table forces into the open: (1) no model states an
observation-level cutoff except TimesFM, whose two dated sources stop at
**November 2023**; (2) every other model's only defensible bound is its
release date, the latest being **Moirai 2.0 on 12 Nov 2025**. So the common
test origin is **t\* = 1 Dec 2025**; for a clean calendar cut the paper uses
forecast origins from **1 Jan 2026** onward, and every test window lies
entirely in 2026. Sundial's remote code pins `transformers==4.40.1` (its
README and model card); it runs in its own environment.

Candidate series with continuous public updates through 2026 (so the test
window can sit after every cutoff):

- **Electricity load and generation** — ENTSO-E Transparency (EU, hourly,
  per bidding zone) and EIA Hourly Grid Monitor (US, hourly, per balancing
  authority). Hourly and daily aggregates; strong seasonality; hundreds of
  series.
- **Road traffic** — Caltrans PeMS (5-min, aggregate to hourly/daily);
  NYC TLC taxi/FHV trip counts (daily per zone).
- **Web attention** — Wikimedia pageviews API (daily, per article; choose a
  fixed random sample of articles by ID so the choice is reproducible).
- **Public health** — CDC FluView ILI and NREVSS (weekly, per region);
  ECDC weekly respiratory surveillance.
- **Macro / finance** — FRED (weekly and monthly series updated through
  2026); daily equity and FX closes as a deliberately hard, near-random-walk
  control.
- **Retail** — hardest to source post-cutoff in public form; use it only if
  an open 2025–26 dataset is found, otherwise state its absence explicitly.

Rule for every source: raw download script in `data/sources/`, a pinned
retrieval date, a checksum, and the exact series selection rule. No
hand-picked series.

## 2. Domain holdout — sources absent from the corpora

Independently of dates, build one group from sources that do not appear in
LOTSA, the Chronos corpus or the TimesFM corpus at all. Check by grepping the
published dataset lists (LOTSA's HF repo lists every constituent). Candidates:
ENTSO-E per-zone generation by fuel type, NYC TLC per-zone counts, ECDC
surveillance, and municipal open-data feeds (bike-share trips, 311 volumes)
from 2025–26. This group is where "did it generalise or did it memorise" is
answered; it should be small but clean.

### Contamination grid (what is already inside the corpora)

Sources: Chronos corpus = arXiv:2403.07815 Appendix B Table 3 and the
`autogluon/chronos_datasets` collection (67 subsets); LOTSA =
`Salesforce/lotsa_data` (174 subsets) — GIFT-Eval Pretrain is LOTSA minus
GIFT-Eval test overlap (arXiv:2410.10393 App. E, 71 univariate + 17
multivariate datasets); TimeBench = Sundial §4.2/App. A (built largely from
LOTSA, Chronos datasets, UTSD, ERA5).

| Series family | Chronos-Bolt (Chronos corpus) | Chronos-2 / TimesFM 2.5 & 3.0 / Moirai 2.0 (GIFT-Eval Pretrain) | Sundial (TimeBench) |
|---|---|---|---|
| M4 | **Yes** — in-domain training set (Benchmark I) | Training portions only; M4 is a GIFT-Eval *test* set, test portions excluded | Excluded (M4 is a GIFT-Eval task; §5.1) |
| M3 | No — zero-shot Benchmark II only | Not in the LOTSA constituent list; unknown for GIFT-Eval Pretrain — verify App. E | Unknown (Chronos datasets contain M3) |
| Tourism | No — zero-shot Benchmark II only | Likely yes — LOTSA has tourism_monthly/quarterly/yearly | Likely yes (via LOTSA) |
| Wikipedia pageviews | **Yes** — wiki_daily_100k (pretraining-only) | **Yes** — LOTSA wiki-rolling_nips, kaggle_web_traffic; TimesFM additionally uses raw Wikimedia pageviews to Nov 2023 | Likely yes |
| Electricity: ENTSO-E / EIA | No — corpus has Electricity (15 min/hourly/weekly), ERCOT, Australian demand, not ENTSO-E or EIA | Not in the LOTSA list (has elecdemand, australian_electricity_demand) | Unknown |
| Traffic: PeMS | No — Monash "traffic" is zero-shot only | **Yes** — LOTSA PEMS03/04/07/08, PEMS_BAY, traffic_hourly/weekly | Likely yes |
| FRED | No — FRED-MD is zero-shot only | **Yes** — LOTSA fred_md | Likely yes |
| CDC ILI | No — not in the Chronos corpus | **Yes** — LOTSA cdc_fluview_ilinet, cdc_fluview_who_nrevss | Likely yes |

Two consequences for the domain holdout: ENTSO-E and EIA feeds are the only
electricity sources absent from every corpus we could list, and every other
family in the plan (PeMS, FRED, CDC, Wikipedia) is contaminated for at least
the GIFT-Eval-trained models — for those families the *temporal* holdout
(windows after t\*) is the only clean design. "Likely" and "Unknown" cells
must be resolved against GIFT-Eval Appendix E and Sundial Appendix A before
the paper claims anything.

## 3. Groups, horizons and protocol

Mirror the M4 protocol so numbers are comparable with prior work:

| Group | Frequency | Horizon | Season | Target count |
|---|---|---|---|---|
| hourly | H | 48 | 24 | 300–500 series |
| daily | D | 14 | 7 | 500–1000 |
| weekly | W | 13 | 1 (M4's official weekly setting) | 200–400 |
| monthly | M | 18 (8 for 2026 origins until more months exist) | 12 | 300–500 |

Same rolling-origin evaluation for every model, same context limit per model
(reported, not hidden), same quantile levels. Metrics stay as today: MASE,
sMAPE, WQL, 80% coverage, plus wall-clock and load time per model on the same
machine, with the device recorded.

## 4. Statistics that make it citable

- Three seeds for every stochastic model; report mean and 95% CI.
- Per-group Diebold–Mariano tests against the best classical baseline, and a
  multiple-comparison-with-the-best (MCB) plot across all models.
- A cost axis: forecasts per second and parameters, so "wins by 3% at 100×
  the cost" is a stated result, not a footnote.

## 5. Order of work

1. Fill the cutoff table (one day; this decides `t*`).
2. Write the download scripts for the three cheapest sources first — EIA,
   Wikimedia, FRED — and get a hourly/daily/weekly/monthly group out of them.
3. Run the existing grid on those groups; check the story holds before
   investing in PeMS/ENTSO-E, which are heavier to retrieve.
4. Add the domain-holdout group.
5. Only then run the full public-benchmark grid for the continuity tables.

## Known gaps to state in the paper rather than hide

- Sundial's remote code needs an isolated environment (transformers ≈4.40);
  run it separately and report the version used.
- TimesFM and Moirai 2 fall back to CPU on Apple MPS for one float64 op;
  timing for those is CPU timing and is labelled as such.
