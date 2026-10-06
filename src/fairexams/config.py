"""Paths, thresholds and fixed choices shared by every stage (see PREREGISTRATION.md)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
REPORT = ROOT / "report"
VALIDATION = ROOT / "validation"

MIRROR = "https://raw.githubusercontent.com/PardhavMaradani/neet-2024-center-marks/main/csv/"
MIRROR_FILES = ("neet-2024-centers.csv", "neet-2024-center-longnames.csv", "neet-2024-center-marks.csv")
NTA_PDF = "https://neetfs.ntaonline.in/NEET_2024_Result/{code}.pdf"

# Exam arithmetic (NEET-UG 2024): +4 right, -1 wrong, 180 questions attempted at most.
MARK_RIGHT = 4
MARK_WRONG = -1
QUESTIONS = 180

# Fixed analysis choices (pre-registered)
TOP_SHARE = 0.01               # primary: national top 1%
TOP_SHARE_FINE = 0.001         # secondary: national top 0.1%
FDR_Q = 0.05                   # Benjamini-Hochberg level
TRIM_FOR_RHO = 0.01            # share of centres (largest |z|) dropped before estimating rho
N_BOOT = 2000                  # bootstrap resamples
N_PERM = 10000                 # permutations for serial-order tests
N_INJECT = 400                 # injections per detection-limit cell
N_VERIFY = 100                 # random centres re-parsed from NTA PDFs
INJECT_M_CENTRE = (5, 10, 20, 40, 80)
INJECT_M_CITY = (25, 50, 100, 200)
SIZE_BANDS = (0, 250, 500, 750, 10**9)
COURT_CUTS = {"rank_108000": 577, "rank_56000": 622}
COACHING_HUBS = ("KOTA", "SIKAR")  # residential coaching hubs, named before the data were seen

# Smoke-test mode for development only (FAIREXAMS_FAST=1); published numbers never use it.
if os.environ.get("FAIREXAMS_FAST"):
    N_BOOT, N_PERM, N_INJECT = 50, 200, 20
SEED = 20261006                # every random draw uses this seed

USER_AGENT = "fair-exams research script (open-source; contact via GitHub)"

for _d in (RAW, INTERIM, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
