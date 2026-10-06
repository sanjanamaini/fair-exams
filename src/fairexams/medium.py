"""Addition A2 (DEVIATIONS.md): where is allotment least random, and does paper medium explain it?

    python -m fairexams.medium

Per state: the intra-class correlation of marks among centres within cities (how far allotment
departs from a lottery for the bulk of candidates) and the spread of centre-mean z-scores.
NTA prints a different booklet for each medium, and regional media are offered only in their own
states (NTA public notice, 19 April 2023). Each regional medium's 2024 registrations are spread over
its states in proportion to their registrations, giving a regional-medium share per state.
Test: Spearman correlation between that share and the state ICC, across states offering a
regional medium. Hindi-belt states are reported separately (NTA gives no state split for Hindi).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import FIGURES, INTERIM, TABLES, VALIDATION
from .stats import nested_icc

MIN_CENTRES = 10


def regional_share() -> pd.Series:
    lang = pd.read_csv(VALIDATION / "nta_language_2024.csv")
    reg = pd.read_csv(VALIDATION / "nta_state_registrations_2024.csv").set_index("state")["registered_2024"]
    share = pd.Series(0.0, index=reg.index)
    for r in lang.itertuples():
        if r.states_offered == "all":
            continue
        states = [s for s in r.states_offered.split(";") if s in reg.index]
        w = reg[states] / reg[states].sum()
        share[states] += r.registered_2024 * w
    return share / reg


def main() -> None:
    cand = pd.read_parquet(INTERIM / "candidates.parquet")
    cent = pd.read_parquet(INTERIM / "centres.parquet")
    scr = pd.read_csv(TABLES / "screen_centres.csv")
    sq = cand.assign(m2=cand["marks"].astype("int64") ** 2).groupby("code")["m2"].sum()
    df = cent[cent["n_centres_city"] >= 2].copy()
    df["ss"] = df["code"].map(sq)
    share = regional_share()
    lang = pd.read_csv(VALIDATION / "nta_language_2024.csv")
    offering = set(s for v in lang["states_offered"] if v != "all" for s in v.split(";"))
    rows = []
    for st, g in df.groupby("state"):
        if len(g) < MIN_CENTRES:
            continue
        rho, _ = nested_icc(g["mean"] * g["n"], g["ss"], g["n"], g["city_id"])
        s = scr[scr["state"] == st]
        rows.append({"state": st, "centres": len(g), "cities": int(g["city_id"].nunique()), "icc_marks": rho,
                     "sd_zmean": float(s["zmean"].std()), "share_abs_zmean_gt3": float((s["zmean"].abs() > 3).mean()),
                     "registered_flags": int((s["q_excess"] <= 0.05).sum()),
                     "regional_share": float(share.get(st, np.nan)), "offers_regional": st in offering})
    out = pd.DataFrame(rows).sort_values("icc_marks", ascending=False)
    out.to_csv(TABLES / "state_allotment.csv", index=False, float_format="%.5g")
    reg = out[out["offers_regional"]]
    rho_s, p_s = stats.spearmanr(reg["regional_share"], reg["icc_marks"])
    hindi = out[~out["offers_regional"]]
    numbers = {"medium_states": int(len(out)), "medium_regional_states": int(len(reg)),
               "medium_spearman": float(rho_s), "medium_spearman_p": float(p_s),
               "medium_icc_max_state": out.iloc[0]["state"], "medium_icc_max": float(out.iloc[0]["icc_marks"]),
               "medium_lottery_like_states": int((out["sd_zmean"] < 1.5).sum()),
               "medium_sorted_states": int((out["sd_zmean"] > 3).sum())}
    for r in out.itertuples():
        k = "state_" + "".join(ch for ch in r.state.lower() if ch.isalpha())
        numbers[k + "_icc"] = float(r.icc_marks)
        numbers[k + "_sdz"] = float(r.sd_zmean)
        numbers[k + "_regshare"] = float(r.regional_share)
    (TABLES / "medium_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter(reg["regional_share"] * 100, reg["icc_marks"], color="#1F3A5F", s=22, label="offers a regional medium")
    for r in reg.itertuples():
        ax.annotate(r.state.title().replace(" (Ut)", ""), (r.regional_share * 100, r.icc_marks), fontsize=6,
                    xytext=(3, 2), textcoords="offset points")
    ax.scatter(np.zeros(len(hindi)) - 4, hindi["icc_marks"], color="#C9A227", s=22, marker="s",
               label="English and Hindi only (Hindi share unknown)")
    for r in hindi.itertuples():
        ax.annotate(r.state.title(), (-4, r.icc_marks), fontsize=5.5, xytext=(-4, 0), textcoords="offset points",
                    ha="right")
    ax.set_xlim(-14, 72)
    ax.set_xlabel("Share of the state's 2024 registrants who chose a regional-language paper (%)")
    ax.set_ylabel("Intra-class correlation of marks\namong centres within cities")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "state_allotment.png", dpi=200)
    plt.close(fig)
    print(out.to_string())
    print(json.dumps({k: v for k, v in numbers.items() if not k.startswith("state_")}, indent=1, default=float))


if __name__ == "__main__":
    main()
