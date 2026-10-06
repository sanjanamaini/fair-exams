"""Stage 2: map exam cities to NFHS-5 districts and build the district development table.

    python -m fairexams.districts

Exact name matches within the state are automatic; everything else comes from the hand-checked
mappings/city_district_overrides.csv (kinds: alias, town, multi, partial, none). Every override
district must exist in NFHS-5 or the stage fails. Writes data/interim/city_district.parquet,
data/interim/district_dev.parquet and results/tables/city_district_map.csv.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .config import INTERIM, RAW, ROOT, TABLES

STATE = {"ANDAMAN AND NICOBAR ISLANDS (UT)": "Andaman Nicobar Islands", "CHANDIGARH(UT)": "Punjab",
         "DADRA AND NAGAR HAVELI (UT)": "Dadra Nagar Haveli Daman Diu",
         "DAMAN AND DIU (UT)": "Dadra Nagar Haveli Daman Diu", "DELHI": "NCT Delhi",
         "JAMMU AND KASHMIR": "Jammu Kashmir", "LADAKH(UT)": "Ladakh", "PUDUCHERRY(UT)": "Puducherry",
         "LAKSHADWEEP (UT)": None}

# The five pre-registered indicators (NFHS-5 district factsheet numbering).
INDICATORS = {
    "14. Women who are literate4 (%)": "women_literate",
    "15. Women with 10 or more years of schooling (%)": "women_school10",
    "7. Population living in households with electricity (%)": "electricity",
    "9. Population living in households that use an improved sanitation facility2 (%)": "sanitation",
    "10. Households using clean fuel for cooking3 (%)": "clean_fuel",
}


def norm(s: str) -> str:
    return re.sub(r"[^A-Z]", "", str(s).upper())


def nfhs_wide() -> pd.DataFrame:
    raw = pd.read_csv(RAW / "nfhs5" / "districts.csv")
    raw = raw[raw["Indicator"].isin(INDICATORS)]
    raw["NFHS5"] = pd.to_numeric(raw["NFHS5"], errors="coerce")
    wide = raw.pivot_table(index=["state", "district"], columns="Indicator", values="NFHS5", aggfunc="first")
    return wide.rename(columns=INDICATORS).reset_index()


def city_map(cities: pd.DataFrame, dev: pd.DataFrame) -> pd.DataFrame:
    over = pd.read_csv(ROOT / "mappings" / "city_district_overrides.csv").fillna("")
    over = {(r.state, r.city): r for r in over.itertuples()}
    rows = []
    for r in cities.itertuples():
        st = STATE.get(r.state, r.state.title())
        o = over.get((r.state, r.city))
        if o is not None:
            dists = [d for d in o.districts.split(";") if d]
            kind = o.kind
        else:
            key = {norm(d): d for d in dev.loc[dev["state"] == st, "district"]}
            names = [r.city] + r.city.split("/") + [re.sub(r"\(.*?\)", "", r.city)] + re.findall(r"\((.*?)\)", r.city)
            hit = next((key[norm(x)] for x in names if norm(x) in key), None)
            dists, kind = ([hit], "exact") if hit else ([], "unmatched")
        for d in dists:
            if not ((dev["state"] == st) & (dev["district"] == d)).any():
                raise ValueError("override district not in NFHS-5: %s / %s" % (st, d))
        rows.append({"city_id": r.city_id, "state": r.state, "city": r.city, "nfhs_state": st,
                     "districts": ";".join(dists), "kind": kind})
    return pd.DataFrame(rows)


def main() -> None:
    cities = pd.read_parquet(INTERIM / "cities.parquet")
    cities = cities[~cities["overseas"]]
    dev = nfhs_wide()
    m = city_map(cities, dev)
    if (m["kind"] == "unmatched").any():
        raise ValueError("unmatched cities: %s" % m.loc[m["kind"] == "unmatched", "city"].tolist())

    # Development index: first principal component of the standardised indicators, higher = more developed.
    cols = list(INDICATORS.values())
    x = dev[cols].astype(float)
    ok = x.notna().all(axis=1)
    z = (x[ok] - x[ok].mean()) / x[ok].std(ddof=0)
    vals, vecs = np.linalg.eigh(np.cov(z.to_numpy().T))
    w = vecs[:, -1]
    w = w if w.sum() > 0 else -w
    dev.loc[ok, "dev_index"] = z.to_numpy() @ w
    loadings = dict(zip(cols, [float(v) for v in w]))
    share = float(vals[-1] / vals.sum())

    m.to_parquet(INTERIM / "city_district.parquet", index=False)
    dev.to_parquet(INTERIM / "district_dev.parquet", index=False)
    out = m.merge(cities[["city_id", "N"]], on="city_id")
    out.to_csv(TABLES / "city_district_map.csv", index=False)
    numbers = {"map_cities": int(len(m)), "map_exact": int((m["kind"] == "exact").sum()),
               "map_alias": int((m["kind"] == "alias").sum()), "map_town": int((m["kind"] == "town").sum()),
               "map_multi": int((m["kind"] == "multi").sum()), "map_partial": int((m["kind"] == "partial").sum()),
               "map_none": int((m["kind"] == "none").sum()), "nfhs_districts": int(len(dev)),
               "dev_pc1_share": share, **{"dev_loading_" + k: v for k, v in loadings.items()}}
    (TABLES / "district_numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))
    print(out.groupby("kind")["N"].agg(["size", "sum"]))


if __name__ == "__main__":
    main()
