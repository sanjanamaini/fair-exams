# Fair Exams

**An open, pre-registered statistical audit of NEET-UG 2024, India's largest exam, centre by centre.**

In July 2024 the Supreme Court made the National Testing Agency publish the marks of all 23 lakh NEET-UG candidates, exam centre by exam centre. It is the only time this has happened. The court compared a few cities with earlier years and found no systemic leak. In May 2026 the next NEET was cancelled after another leak.

This project asks what the 2024 data can and cannot reveal:

1. **Integrity.** Is any of the 4,750 centres incompatible with candidates being allotted at random within their city? And how big would a fraud have to be for a screen like this to catch it?
2. **Geography.** How concentrated are top scores across 571 cities, and how far does that follow district development?
3. **Rank fragility.** How many ranks does a single question move?

## Status

Stage 0: hypotheses, tests, thresholds and a blind validation list are committed in [`PREREGISTRATION.md`](PREREGISTRATION.md) before any marks were analysed. Results follow stage by stage.

## Layout

| Path | What |
|---|---|
| `PREREGISTRATION.md` | what is tested, how, and what counts as a null |
| `DEVIATIONS.md` | every later change, dated, with its reason |
| `DATA.md` | sources, licences, verification |
| `validation/` | blind list of incident centres; the Supreme Court's figures to reproduce |
| `src/fairexams/` | the pipeline |
| `Makefile` | `make all` rebuilds everything from public sources |

## Licence

Code and derived aggregates: MIT. Raw data is never redistributed; `make data` fetches it from the publishers.
