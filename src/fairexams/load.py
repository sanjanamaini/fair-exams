"""Readers for the raw files (the mirror CSVs have no header row and SQL-escaped quotes)."""
from __future__ import annotations

import pandas as pd

from .config import RAW

MIRROR_DIR = RAW / "mirror"


def centres() -> pd.DataFrame:
    """One row per centre: code, state, city, name, longname."""
    c = pd.read_csv(MIRROR_DIR / "neet-2024-centers.csv", header=None,
                    names=["row", "state", "city", "name", "code"], dtype={"code": int})
    ln = pd.read_csv(MIRROR_DIR / "neet-2024-center-longnames.csv", header=None,
                     names=["code", "longname"], dtype={"code": int})
    c = c.drop(columns="row").merge(ln, on="code", how="left", validate="one_to_one")
    for col in ("state", "city", "name", "longname"):
        c[col] = c[col].astype(str).str.replace("''", "'", regex=False).str.strip()
    return c


def marks() -> pd.DataFrame:
    """One row per candidate: centre code, serial number within the centre, total marks."""
    return pd.read_csv(MIRROR_DIR / "neet-2024-center-marks.csv", header=None,
                       names=["code", "serial", "marks"],
                       dtype={"code": "int32", "serial": "int32", "marks": "int16"})
