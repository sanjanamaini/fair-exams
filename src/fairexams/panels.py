"""Stage 2: clean panels from the verified marks, plus the arithmetic check and the court check.

    python -m fairexams.panels

Writes data/interim/{candidates,centres,cities}.parquet and, in results/tables/:
score_distribution.csv, invalid_scores.csv, court_check.csv and panel_numbers.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import load
from .config import (COURT_CUTS, INTERIM, MARK_RIGHT, MARK_WRONG, QUESTIONS, TABLES, TOP_SHARE,
                     TOP_SHARE_FINE, VALIDATION)


def valid_scores() -> set[int]:
    """Every total reachable as 4c - w with c + w <= 180 (c right, w wrong)."""
    out = set()
    for c in range(QUESTIONS + 1):
        for w in range(QUESTIONS - c + 1):
            out.add(MARK_RIGHT * c + MARK_WRONG * w)
    return out


def top_threshold(marks: np.ndarray, share: float) -> int:
    """Smallest score t such that at most `share` of all candidates score t or more."""
    n = len(marks)
    counts = pd.Series(marks).value_counts().sort_index(ascending=False)
    at_or_above = counts.cumsum()
    ok = at_or_above[at_or_above <= share * n]
    if ok.empty:  # even the top score's tie group is larger than the share: the group is empty
        return int(counts.index.max()) + 1
    return int(ok.index.min())


def rank_table(marks: np.ndarray) -> pd.DataFrame:
    """Per score: candidates on it, best rank (1 + candidates above) and worst rank (candidates at or above)."""
    counts = pd.Series(marks).value_counts().sort_index(ascending=False)
    df = pd.DataFrame({"score": counts.index.astype(int), "candidates": counts.to_numpy()})
    df["worst_rank"] = df["candidates"].cumsum()
    df["best_rank"] = df["worst_rank"] - df["candidates"] + 1
    return df


def score_at_rank(rt: pd.DataFrame, rank: int) -> int:
    return int(rt.loc[(rt["best_rank"] <= rank) & (rt["worst_rank"] >= rank), "score"].iloc[0])


def main() -> None:
    cent = load.centres()
    cand = load.marks()
    numbers = {}

    # Integrity of the panel itself.
    assert not cand.duplicated(["code", "serial"]).any(), "duplicate (centre, serial)"
    per = cand.groupby("code")["serial"].agg(["size", "max", "min"])
    assert (per["size"] == per["max"]).all() and (per["min"] == 1).all(), "serials are not 1..n"
    missing = set(cand["code"]) - set(cent["code"])
    assert not missing, "marks for unknown centres: %s" % sorted(missing)[:5]
    empty = sorted(set(cent["code"]) - set(cand["code"]))
    numbers.update({"candidates": int(len(cand)), "centres": int(len(cent)), "centres_empty": len(empty),
                    "cities": int(cent[["state", "city"]].drop_duplicates().shape[0])})

    cent["overseas"] = cent["state"].eq("OUTSIDE-INDIA")
    cent["city_id"] = (cent["state"] + " | " + cent["city"]).astype("category").cat.codes.astype("int32")
    numbers["cities_india"] = int(cent.loc[~cent["overseas"], ["state", "city"]].drop_duplicates().shape[0])
    numbers["cities_overseas"] = int(cent.loc[cent["overseas"], ["state", "city"]].drop_duplicates().shape[0])
    cand = cand.merge(cent[["code", "city_id", "overseas"]], on="code", how="left", validate="many_to_one")
    numbers["candidates_overseas"] = int(cand["overseas"].sum())

    # Arithmetic check: every total must be 4c - w with c + w <= 180.
    valid = valid_scores()
    cand["valid"] = cand["marks"].isin(valid)
    bad = cand[~cand["valid"]]
    inv = bad.groupby(["code", "marks"]).size().rename("candidates").reset_index()
    inv = inv.merge(cent[["code", "state", "city", "name"]], on="code", how="left")
    inv.to_csv(TABLES / "invalid_scores.csv", index=False)
    numbers.update({"invalid_candidates": int(len(bad)), "invalid_centres": int(bad["code"].nunique()),
                    "invalid_distinct_scores": int(bad["marks"].nunique()),
                    "valid_score_values": len(valid), "marks_min": int(cand["marks"].min()),
                    "marks_max": int(cand["marks"].max())})
    impossible_top = sorted(s for s in range(600, 721) if s not in valid)
    numbers["impossible_scores_600_720"] = ", ".join(str(s) for s in impossible_top)

    # National distribution, thresholds and ranks.
    m = cand["marks"].to_numpy()
    rt = rank_table(m)
    rt["valid"] = rt["score"].isin(valid)
    rt.to_csv(TABLES / "score_distribution.csv", index=False)
    t1, t01 = top_threshold(m, TOP_SHARE), top_threshold(m, TOP_SHARE_FINE)
    med = float(np.median(m))
    numbers.update({"t_top1": t1, "t_top01": t01, "median": med, "mean": float(m.mean()), "sd": float(m.std()),
                    "share_top1": float((m >= t1).mean()), "share_top01": float((m >= t01).mean()),
                    "n_top1": int((m >= t1).sum()), "n_top01": int((m >= t01).sum()),
                    "share_ge_median": float((m >= med).mean())})
    cand["top1"] = cand["marks"] >= t1
    cand["top01"] = cand["marks"] >= t01
    cand["ge_med"] = cand["marks"] >= med

    # Centre and city panels.
    g = cand.groupby("code")
    cpanel = pd.DataFrame({"n": g.size(), "mean": g["marks"].mean(), "sd": g["marks"].std(ddof=0),
                           "top1": g["top1"].sum(), "top01": g["top01"].sum(), "ge_med": g["ge_med"].sum(),
                           "min": g["marks"].min(), "max": g["marks"].max(),
                           "invalid": g["valid"].apply(lambda v: int((~v).sum()))}).reset_index()
    cpanel = cent.merge(cpanel, on="code", how="left")
    cpanel["n_centres_city"] = cpanel.groupby("city_id")["code"].transform("size")
    gc = cand.groupby("city_id")
    city = pd.DataFrame({"N": gc.size(), "mean": gc["marks"].mean(), "sd": gc["marks"].std(ddof=0),
                         "top1": gc["top1"].sum(), "top01": gc["top01"].sum(), "ge_med": gc["ge_med"].sum()})
    city = city.join(cent.groupby("city_id").agg(state=("state", "first"), city=("city", "first"),
                                                 centres=("code", "size"), overseas=("overseas", "first")))
    city = city.reset_index()

    # Court check: reproduce the Supreme Court's city figures (2024 INSC 568, paras 74-76 and 81).
    court = pd.read_csv(VALIDATION / "court_figures.csv")
    rows = []
    for name in ("HAZARIBAGH", "PATNA", "GODHRA"):
        cc = city[city["city"].eq(name) & ~city["overseas"]].iloc[0]
        sub = cand[cand["city_id"].eq(cc["city_id"])]
        ours = {"appeared": int(cc["N"]), "centres": int(cc["centres"]),
                "within_rank_108000": int((sub["marks"] >= COURT_CUTS["rank_108000"]).sum()),
                "within_rank_56000": int((sub["marks"] >= COURT_CUTS["rank_56000"]).sum())}
        for item, val in ours.items():
            hit = court[(court["city"] == name) & (court["item"] == item)]
            rows.append({"city": name, "item": item, "ours": val,
                         "court": int(hit["value"].iloc[0]) if len(hit) else np.nan})
    rows.append({"city": "ALL", "item": "marks_at_rank_108000", "ours": score_at_rank(rt, 108000), "court": 577})
    rows.append({"city": "ALL", "item": "marks_at_rank_56000", "ours": score_at_rank(rt, 56000), "court": 622})
    rows.append({"city": "ALL", "item": "centres", "ours": int(len(cent)), "court": 4750})
    rows.append({"city": "ALL", "item": "indian_cities", "ours": numbers["cities_india"], "court": 571})
    rows.append({"city": "ALL", "item": "overseas_cities", "ours": numbers["cities_overseas"], "court": 14})
    cchk = pd.DataFrame(rows)
    cchk["diff"] = cchk["ours"] - cchk["court"]
    cchk.to_csv(TABLES / "court_check.csv", index=False)

    cand[["code", "serial", "marks", "city_id"]].to_parquet(INTERIM / "candidates.parquet", index=False)
    cpanel.to_parquet(INTERIM / "centres.parquet", index=False)
    city.to_parquet(INTERIM / "cities.parquet", index=False)
    (TABLES / "panel_numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))
    print(cchk.to_string())
    print(inv.groupby("marks")["candidates"].sum().to_string() if len(inv) else "no invalid scores")


if __name__ == "__main__":
    main()
