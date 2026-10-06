# Data sources and licences

No raw data is committed. `make data` downloads everything into `data/raw/` (git-ignored) and records a SHA-256 checksum of every file in `data/raw/MANIFEST.csv`.

| Source | What | Access | Terms |
|---|---|---|---|
| National Testing Agency (NTA), NEET-UG 2024 centre-wise results, published 20 July 2024 on the Supreme Court's order of 18 July 2024 | one PDF per centre: centre code and address, then every candidate's serial number and total marks (no names, roll numbers or subject marks) | `https://neetfs.ntaonline.in/NEET_2024_Result/<centre code>.pdf` | public and anonymised by court order; no reuse licence found, so nothing is redistributed |
| PardhavMaradani/neet-2024-center-marks (GitHub) | the same data parsed into CSV: centres (state, city, name, code), long names (addresses), and marks (code, serial, marks) | `https://raw.githubusercontent.com/PardhavMaradani/neet-2024-center-marks/main/csv/` | no licence in the repository; used as a convenience copy only after every listed and flagged centre and a random sample of 100 centres match NTA's own PDFs exactly |
| NFHS-5 district factsheets (2019-21), International Institute for Population Sciences and ICF | district indicators: literacy, schooling, electricity, sanitation, cooking fuel | compiled CSVs at `https://github.com/jvargh7/nfhs5_factsheets` | compilation MIT; cite IIPS and ICF (2021), *National Family Health Survey (NFHS-5), 2019-21* |
| Supreme Court of India, *Vanshika Yadav v. Union of India*, 2024 INSC 568, judgment of 2 August 2024 | city figures for Hazaribagh, Patna and Godhra; rank cut points; incident record | `https://api.sci.gov.in/supremecourt/2024/22797/22797_2024_1_1501_54391_Judgement_02-Aug-2024.pdf` | public judgment; figures transcribed in `validation/court_figures.csv` with paragraph numbers |

Derived results in `results/` are aggregates (per centre, per city, per district, per score) and are published under the repository licence. No file in this repository holds candidate-level rows.
