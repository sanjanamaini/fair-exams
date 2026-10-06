"""Stage 3d: blind validation, court cities and centre profiles (PREREGISTRATION.md, Strand 1c).

    python -m fairexams.validate

Reports every List A and List B centre against its registered prediction, the court cities against
the between-city funnel, an upper bound on how many leak beneficiaries could have reached each
threshold, and a profile (distribution against the rest of the city) for every flagged centre.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import COURT_CUTS, FDR_Q, FIGURES, INTERIM, TABLES, VALIDATION
from .stats import random_effects_tau2


def main() -> None:
    cand = pd.read_parquet(INTERIM / "candidates.parquet")
    city = pd.read_parquet(INTERIM / "cities.parquet")
    scr = pd.read_csv(TABLES / "screen_centres.csv")
    tail = pd.read_csv(TABLES / "bulk_tail.csv")[["code", "tail_E", "z_tail", "q_tail", "flag_tail"]]
    nums = json.loads((TABLES / "panel_numbers.json").read_text())
    t1, t01 = nums["t_top1"], nums["t_top01"]
    blind = pd.read_csv(VALIDATION / "blind_list.csv")
    numbers = {}

    # 1. Every listed centre against its prediction.
    v = blind.drop(columns=["state", "city"]).merge(scr, left_on="centre_code", right_on="code", how="left")
    v = v.merge(tail, on="code", how="left")
    v["flag_excess"] = v["q_excess"] <= FDR_Q
    v["flag_deficit"] = v["q_deficit"] <= FDR_Q
    v["informative"] = v["E1"] >= 1
    cols = ["list", "role", "centre_code", "city", "centre_name", "n", "top1", "E1", "z1_adj", "q_excess",
            "q_deficit", "flag_excess", "flag_deficit", "zmean_adj", "z_tail", "q_tail", "city_rank_z1",
            "n_centres_city", "informative", "prediction"]
    v[cols].to_csv(TABLES / "validation_centres.csv", index=False, float_format="%.4g")
    a_named = v[(v["list"] == "A") & (v["role"] == "named")]
    numbers.update({"val_listA_named": int(len(a_named)),
                    "val_listA_named_excess_flags": int(a_named["flag_excess"].sum()),
                    "val_listA_named_deficit_flags": int(a_named["flag_deficit"].sum()),
                    "val_listA_all": int((v["list"] == "A").sum()),
                    "val_listA_all_excess_flags": int(v.loc[v["list"] == "A", "flag_excess"].sum()),
                    "val_listA_informative": int(v.loc[v["list"] == "A", "informative"].sum()),
                    "val_listB_excess_flags": int(v.loc[v["list"] == "B", "flag_excess"].sum())})
    for r in v.itertuples():
        k = "val_%d" % r.centre_code
        numbers.update({k + "_n": int(r.n), k + "_top1": int(r.top1), k + "_E1": float(r.E1), k + "_z": float(r.z1_adj),
                        k + "_q": float(r.q_excess), k + "_ztail": float(r.z_tail), k + "_rank": int(r.city_rank_z1),
                        k + "_centres": int(r.n_centres_city)})
    # Mean percentile of List A named centres' z among their city peers (rank-based, no distribution assumption).
    pct = []
    for r in a_named.itertuples():
        peers = scr[scr["city"] == r.city]
        pct.append(float((peers["z1_adj"] < r.z1_adj).mean()))
    numbers["val_listA_named_mean_city_percentile"] = float(np.mean(pct))

    # 2. Court cities on the between-city funnel and bounds on beneficiaries.
    ci = city[~city["overseas"]].copy()
    pi = float(city["top1"].sum() / city["N"].sum())
    y, s2 = ci["top1"] / ci["N"], pi * (1 - pi) / ci["N"]
    tau2, _, _ = random_effects_tau2(y.to_numpy(), s2.to_numpy())
    ci["z_funnel"] = (y - pi) / np.sqrt(s2 + tau2)
    rows = []
    for name in ("HAZARIBAGH", "PATNA", "GODHRA"):
        r = ci[ci["city"] == name].iloc[0]
        sub = cand[cand["city_id"] == r["city_id"]]["marks"]
        row = {"city": name, "candidates": int(r["N"]), "top1": int((sub >= t1).sum()), "top01": int((sub >= t01).sum()),
               "ge_622": int((sub >= COURT_CUTS["rank_56000"]).sum()), "ge_577": int((sub >= COURT_CUTS["rank_108000"]).sum()),
               "top1_rate": float((sub >= t1).mean()), "z_funnel": float(r["z_funnel"]),
               "funnel_rank": int((ci["z_funnel"] > r["z_funnel"]).sum() + 1)}
        rows.append(row)
        for k2, val in row.items():
            if k2 != "city":
                numbers["court_%s_%s" % (name.lower(), k2)] = val
    pd.DataFrame(rows).to_csv(TABLES / "validation_court_cities.csv", index=False, float_format="%.4g")
    numbers.update({"funnel_tau2": tau2, "funnel_pi": pi, "funnel_cities": int(len(ci))})

    # 3. Profiles of flagged centres: distribution against the rest of the city, and top-end shape.
    flagged = scr[scr["q_excess"] <= FDR_Q]
    tail_flag = tail[tail["flag_tail"]]
    show = pd.concat([flagged[["code"]], tail_flag[["code"]], scr[scr["q_deficit"] <= FDR_Q][["code"]]]).drop_duplicates()
    show = show.merge(scr, on="code").merge(tail, on="code")
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    prof = []
    nat_top = cand["marks"][cand["marks"] >= t1]
    share_top01_given_top1 = float((nat_top >= t01).mean())
    for r in show.itertuples():
        cid = cand.loc[cand["code"] == r.code, "city_id"].iloc[0]
        own = cand.loc[cand["code"] == r.code, "marks"]
        rest = cand.loc[(cand["city_id"] == cid) & (cand["code"] != r.code), "marks"]
        k1 = int((own >= t1).sum())
        k01 = int((own >= t01).sum())
        p_extreme = stats.binom.sf(k01 - 1, k1, share_top01_given_top1) if k1 else 1.0
        d = {"code": r.code, "city": r.city, "name": r.name, "n": r.n, "registered_flag": bool(r.q_excess <= FDR_Q),
             "deficit_flag": bool(r.q_deficit <= FDR_Q), "tail_flag": bool(r.flag_tail), "top1": k1, "top01": k01,
             "top01_share_of_top1": k01 / k1 if k1 else np.nan, "p_extreme_top": p_extreme}
        for q in qs:
            d["own_q%d" % int(q * 100)] = float(own.quantile(q))
            d["rest_q%d" % int(q * 100)] = float(rest.quantile(q))
        d["lower_half_shift"] = float(np.mean([own.quantile(q) - rest.quantile(q) for q in (0.1, 0.25)]))
        prof.append(d)
    prof = pd.DataFrame(prof)
    prof.to_csv(TABLES / "flag_profiles.csv", index=False, float_format="%.4g")
    numbers["nat_top01_share_of_top1"] = share_top01_given_top1
    reg = prof[prof["registered_flag"]]
    numbers["flags_lower_half_shift_positive"] = int((reg["lower_half_shift"] > 0).sum())
    numbers["flags_lower_half_shift_median"] = float(reg["lower_half_shift"].median())
    plot_profiles(cand, show)
    (TABLES / "validate_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(v[cols].to_string())
    print(pd.DataFrame(rows).to_string())
    print(prof.to_string())
    print(json.dumps({k: numbers[k] for k in numbers if not k.startswith("val_2") and not k.startswith("val_1")
                      and not k.startswith("val_3")}, indent=1, default=float))


def plot_profiles(cand, show):
    show = show.sort_values("z1_adj", ascending=False).head(12)
    n = len(show)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 1.9 * rows), sharex=True)
    bins = np.arange(-180, 725, 30)
    for ax, r in zip(np.ravel(axes), show.itertuples()):
        cid = cand.loc[cand["code"] == r.code, "city_id"].iloc[0]
        own = cand.loc[cand["code"] == r.code, "marks"]
        rest = cand.loc[(cand["city_id"] == cid) & (cand["code"] != r.code), "marks"]
        ax.hist(rest, bins=bins, density=True, color="#C8D3DB", label="rest of city")
        ax.hist(own, bins=bins, density=True, histtype="step", color="#1F3A5F", lw=1.3, label="centre")
        ax.set_title("%s, %s" % (r.city.title()[:14], r.name.title()[:22]), fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_yticks([])
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    np.ravel(axes)[0].legend(frameon=False, fontsize=6)
    fig.supxlabel("Marks (out of 720)", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "flag_profiles.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
