# Fair Exams

**Do the figures speak? An open, pre-registered audit of NEET-UG 2024, India's largest exam, centre by centre.**

In July 2024 the Supreme Court made the National Testing Agency (NTA) publish the marks of all 2,333,162 NEET-UG candidates, exam centre by exam centre. It is the only time this has happened. The court compared three cities with earlier years and refused a re-test. In May 2026 the next NEET was cancelled after another leak.

This project treats NTA's allotment of candidates to centres within each city as a natural experiment, tests every centre against the rest of its own city, and measures what such a screen can and cannot catch. Every hypothesis, threshold and prediction was committed before the marks were analysed.

**Report:** [`report/report.pdf`](report/report.pdf) (15 pages). **Explorer:** [`docs/index.html`](docs/index.html) (one offline page, every centre, city and state).

## What it finds

- **The public file is the court's data.** It reproduces every figure in the Supreme Court judgment exactly (Hazaribagh 2,733 candidates, 126 at 577 marks or more, 58 at 622 or more; Patna 48,643 in 70 centres; Godhra 2,484 in 2 centres; the 577 and 622 mark cut points), plus NTA's appeared count, centre count, the 61 candidates on 720 before the July revision and the 2,321 on 700 or more.
- **Top scorers are spread almost as if by lottery; average marks are not.** The intra-class correlation of top-1% status within cities is 0.0004 (design effect 1.19). For average marks it is 0.018.
- **Nine centres have more top scorers than random allotment allows. All nine are stronger cohorts.** Every one is shifted across its whole distribution, lower quartiles included. Only one, R.K. University Unit-1 in Rajkot, also has more top scorers than its own bulk predicts (93 against 34.7), under every one of 50 seeds. Totals cannot tell a strong sub-cohort from a boosted one; only item-level data can. No finding of misconduct is made.
- **The blind test passed.** None of the centres in the 2024 incident record is flagged, as predicted before the data were opened. At most 18 of the roughly 125 Hazaribagh leak beneficiaries can have reached the national top 1%.
- **Detection limits.** A boost to about 4-5% of one centre's candidates is caught 80% of the time. A leak spread across a city is invisible to any within-city test; only a between-city funnel sees it, and only when large.
- **Allotment is a lottery in some states and not in others.** Lottery-like in 12 states (Kerala, Karnataka, Maharashtra...), strongly grouped in 9 (Madhya Pradesh, West Bengal, Uttar Pradesh...). Across the 12 states offering a regional-language paper, the share taking one predicts the grouping (Spearman 0.69). NTA prints a separate booklet for each medium.
- **Top scores are concentrated; development does not explain them.** Ten cities hold 31% of the top 1% with 14% of candidates; Sikar alone has 1,563. District development predicts mean marks (Spearman 0.48) but barely predicts toppers (0.18).
- **Ranks are fragile.** At the seat line (577 marks, rank 108,000) one question is worth 6,006 ranks; 1,428 candidates share exactly 600.

## Use the monitor on any exam

```bash
make test                                                   # 18 unit tests
PYTHONPATH=src .venv/bin/python -m fairexams.monitor --simulate --out demo     # synthetic exam, one planted fraud
PYTHONPATH=src .venv/bin/python -m fairexams.monitor --input marks.csv --out out  # your data
```

The input is one row per candidate with three columns: `centre`, `city`, `marks` (names configurable with `--centre`, `--city`, `--marks`). No identities are needed. The output folder holds `centres.csv` (every centre's statistics), `flags.csv`, `funnel.png` and `summary.md`, a plain-language summary that says what a flag does and does not mean. On the synthetic exam the planted centre is caught on both lists, a grouped strong cohort only on the first, and the exact test alone would raise 13 alarms instead of 2.

## Reproduce everything

Needs Python 3.9 or later, `pdflatex` and an internet connection.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make all        # download, verify against NTA's PDFs, panels, every analysis, figures, explorer, report
```

`make all` takes about two minutes. Every number in the report is written by the pipeline into `results/numbers.json` and `report/numbers.tex`; none is typed by hand. Two full runs give identical numbers.

## How it works

| Idea | Borrowed from | Used for |
|---|---|---|
| Exact tests under random allotment | randomisation inference (Fisher 1935; Rosenbaum 2002) | each centre against the rest of its city |
| Design effect, 1 + (n - 1) rho | survey sampling (Kish 1965) | allotment that is not a lottery |
| False discovery rate, empirical null | large-scale inference (Benjamini and Hochberg 1995; Storey 2002; Efron 2004) | 4,656 tests at once, and a check on the null itself |
| Funnel plots with overdispersion | healthcare regulation (Spiegelhalter 2005) | comparing cities of very different sizes |
| Injection-recovery | exoplanet surveys (Christiansen et al. 2013) | measuring what the screen can catch |
| Cheating from score patterns | education economics and test security (Jacob and Levitt 2003; Cizek and Wollack 2017) | the framing of the integrity screen |
| Positive deviance | public health (Marsh et al. 2004) | districts above their development-predicted rate |
| Binomial error model | psychometrics (Lord 1965) | how much a rank can be trusted |

## Layout

| Path | What |
|---|---|
| `PREREGISTRATION.md` | what is tested, how, and what counts as a null; committed before any marks were analysed |
| `DEVIATIONS.md` | four additions and two corrections, each dated with its reason |
| `DATA.md` | sources, licences, verification |
| `validation/` | blind list of incident centres, the court's figures, NTA's published tables |
| `mappings/` | hand-checked exam-city to district overrides |
| `src/fairexams/` | the pipeline (`download`, `verify`, `panels`, `districts`, `screen`, `serial`, `bulk`, `stability`, `medium`, `inject`, `validate`, `geography`, `fragility`, `monitor`, `demo`, `numbers`, `report_tables`) |
| `results/` | aggregate tables and figures (no candidate-level rows) |
| `report/` | `report.tex`, generated `numbers.tex` and tables, `report.pdf` |
| `docs/index.html` | the explorer |
| `tool/` | explorer builder and template |
| `tests/` | unit tests |

## Read a flag carefully

A flag means a centre's results are unlikely under random allotment within its city. It is a reason to review, never evidence of misconduct. Stronger cohorts allotted together (by paper medium, school or registration batch) produce the same signal. Centre names appear only because NTA published them by court order. The data are anonymised; nothing here attempts to identify a candidate.

## Data and licence

Raw data is never redistributed: `make data` fetches NTA's centre-wise results (via a public CSV parse, verified against NTA's own PDFs) and the NFHS-5 district factsheets. See `DATA.md`. Code and derived aggregates: MIT licence.
