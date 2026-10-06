"""Build docs/index.html: a self-contained explorer of every centre, city and state.

    python tool/build_explorer.py

Embeds centre and city aggregates (never candidate rows) as JSON in one HTML file with no external
scripts, so it works offline and can be hosted anywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairexams.config import INTERIM, TABLES  # noqa: E402
from fairexams.text import nice_name  # noqa: E402

EDGES = np.arange(-80, 761, 40)  # 21 bins of 40 marks


def hist(m: np.ndarray) -> list:
    return np.histogram(np.clip(m, -80, 759), bins=EDGES)[0].tolist()


def main() -> None:
    cand = pd.read_parquet(INTERIM / "candidates.parquet")
    cent = pd.read_parquet(INTERIM / "centres.parquet")
    city = pd.read_parquet(INTERIM / "cities.parquet")
    scr = pd.read_csv(TABLES / "screen_centres.csv")[["code", "E1", "z1_adj", "q_excess", "q_deficit", "zmean_adj"]]
    tail = pd.read_csv(TABLES / "bulk_tail.csv")[["code", "tail_E", "z_tail", "q_tail"]]
    stab = pd.read_csv(TABLES / "flag_stability.csv")[["centre", "flag_rate", "flag_tail_rate"]].rename(columns={"centre": "code"})
    geo = pd.read_csv(TABLES / "geo_cities.csv")[["city_id", "rate", "z_funnel", "hub"]]
    states = pd.read_csv(TABLES / "state_allotment.csv")
    df = cent.merge(scr, on="code", how="left").merge(tail, on="code", how="left").merge(stab, on="code", how="left")
    tested = df["z1_adj"].notna()
    df.loc[tested, ["flag_rate", "flag_tail_rate"]] = df.loc[tested, ["flag_rate", "flag_tail_rate"]].fillna(0.0)
    hists = {c: hist(g.to_numpy()) for c, g in cand.groupby("code")["marks"]}
    chist = {c: hist(g.to_numpy()) for c, g in cand.groupby("city_id")["marks"]}

    def r(x, d=2):
        return None if pd.isna(x) else round(float(x), d)

    centres = [[int(x.code), int(x.city_id), nice_name(x["name"]), int(x.n), int(x.top1), r(x.E1, 1), r(x.z1_adj),
                r(x.q_excess, 4), r(x.q_deficit, 4), r(x.zmean_adj), r(x.z_tail), r(x.q_tail, 4),
                r(x.flag_rate), r(x.flag_tail_rate), round(float(x["mean"]), 1), hists[x.code]]
               for _, x in df.iterrows()]
    city = city.merge(geo, on="city_id", how="left")
    cities = {int(x.city_id): [x.city.title(), x.state.title(), int(x.N), int(x.top1), int(x.centres), r(x.z_funnel),
                               bool(x.hub) if not pd.isna(x.hub) else False, bool(x.overseas), chist[x.city_id]]
              for _, x in city.iterrows()}
    st = [[x.state.title(), round(float(x.icc_marks), 4), round(float(x.sd_zmean), 2), bool(x.offers_regional),
           round(float(x.regional_share) * 100, 1)] for _, x in states.iterrows()]
    nums = {}
    for p in TABLES.glob("*_numbers.json"):
        nums.update(json.loads(p.read_text()))
    head = {k: nums[k] for k in ("candidates", "centres", "screen_centres", "flags_layer2", "tail_flags", "rho1",
                                 "rho_marks", "t_top1", "geo_hubs")}
    data = json.dumps({"c": centres, "y": cities, "s": st, "h": head, "e": EDGES.tolist()}, separators=(",", ":"))
    html = (ROOT / "tool" / "explorer_template.html").read_text().replace("/*DATA*/", data)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(html)
    print("docs/index.html: %.0f KB" % ((ROOT / "docs" / "index.html").stat().st_size / 1024))


if __name__ == "__main__":
    main()
