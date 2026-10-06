"""Worked examples for the study guide: each method applied by hand to real centres and cities.

    python -m fairexams.examples

Recomputes, step by step, numbers the pipeline produces in bulk (so the guide can show the
arithmetic) and writes results/tables/examples_numbers.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from .bulk import normal_scores
from .config import INTERIM, TABLES
from .stats import hypergeom_tails, nested_icc
from .text import nice_name


def main() -> None:
    cand = pd.read_parquet(INTERIM / "candidates.parquet")
    cent = pd.read_parquet(INTERIM / "centres.parquet")
    city = pd.read_parquet(INTERIM / "cities.parquet")
    scr = pd.read_csv(TABLES / "screen_centres.csv")
    nums = json.loads((TABLES / "panel_numbers.json").read_text())
    scrn = json.loads((TABLES / "screen_numbers.json").read_text())
    t1 = nums["t_top1"]
    ex = {}

    # 1. Exam arithmetic: why 717 is impossible and 716 is not.
    ex["arith_717_min_correct"] = int(np.ceil(717 / 4))
    ex["arith_716_c"], ex["arith_716_w"] = 179, 0

    # 2. Hypergeometric test for two centres, by hand.
    for code, key in ((260402, "oasis"), (220701, "rk")):
        c = cent[cent["code"] == code].iloc[0]
        cy = city[city["city_id"] == c["city_id"]].iloc[0]
        N, K, n, k = int(cy["N"]), int(cy["top1"]), int(c["n"]), int(c["top1"])
        p_ge, p_le, p_gt, p_eq = (float(v) for v in hypergeom_tails(k, N, K, n))
        s = scr[scr["code"] == code].iloc[0]
        ex.update({f"{key}_city_n": N, f"{key}_city_top": K, f"{key}_centre_n": n, f"{key}_centre_top": k, f"{key}_E": n * K / N,
                   f"{key}_city_share": K / N, f"{key}_p_ge": p_ge, f"{key}_p_eq": p_eq, f"{key}_p_gt": p_gt,
                   f"{key}_z_mid": float(stats.norm.isf(p_gt + 0.5 * p_eq)), f"{key}_z": float(s["z1"]),
                   f"{key}_deff": 1 + (n - 1) * scrn["rho1"], f"{key}_z_adj": float(s["z1_adj"]),
                   f"{key}_q": float(s["q_excess"])})
        # Binomial standard deviation with the finite-population correction, for intuition.
        ex[f"{key}_sd"] = float(np.sqrt(n * (K / N) * (1 - K / N) * (N - n) / (N - 1)))

    # 3. ICC by ANOVA: the two mean squares and the effective centre size.
    df = scr.merge(cent[["code", "city_id"]], on="code")
    keep = np.abs(df["z1"]).rank(ascending=False, method="first") > 0.01 * len(df)
    rho, det = nested_icc(df.loc[keep, "top1"], df.loc[keep, "top1"], df.loc[keep, "n"], df.loc[keep, "city_id"])
    ex.update({"icc_rho": rho, "icc_msb": det["msb"], "icc_msw": det["msw"], "icc_n0": det["n0"],
               "icc_df_between": det["df_between"], "icc_df_within": det["df_within"],
               "deff_n500_top": 1 + 499 * scrn["rho1"], "deff_n500_marks": 1 + 499 * scrn["rho_marks"]})

    # 4. Benjamini-Hochberg on the real p-values: the ten smallest and their bars.
    p = np.sort(stats.norm.sf(df["z1_adj"].to_numpy()))
    m = len(p)
    for i in range(10):
        ex[f"bh_p{i + 1}"] = float(p[i])
        ex[f"bh_bar{i + 1}"] = 0.05 * (i + 1) / m
    ex["bh_m"] = m
    ex["bh_expected_false_at_5pct"] = 0.05 * m

    # 5. Tail beyond the bulk for R.K. University Unit-1, step by step.
    rk = cent[cent["code"] == 220701].iloc[0]
    g = cand[cand["city_id"] == rk["city_id"]]
    own = g.loc[g["code"] == 220701, "marks"].to_numpy()
    rest = np.sort(g.loc[g["code"] != 220701, "marks"].to_numpy())
    u = normal_scores(own, rest)
    q25, mu, q75 = np.quantile(u, [0.25, 0.5, 0.75])
    sigma = (q75 - q25) / 1.349
    below = np.searchsorted(rest, t1, "left") / len(rest)
    t_u = float(stats.norm.ppf(below))
    share = float(stats.norm.sf((t_u - mu) / sigma))
    ex.update({"a1_rest_share_below_t1": below, "a1_t_u": t_u, "a1_mu": float(mu), "a1_sigma": float(sigma),
               "a1_q25": float(q25), "a1_q75": float(q75), "a1_share": share, "a1_E": share * len(own),
               "a1_rest_top_share": 1 - below})

    # 6. Funnel z for Hazaribagh, by hand.
    ci = city[~city["overseas"]]
    pi = float(city["top1"].sum() / city["N"].sum())
    hz = ci[ci["city"] == "HAZARIBAGH"].iloc[0]
    tau2 = json.loads((TABLES / "geography_numbers.json").read_text())["geo_tau2"]
    y = hz["top1"] / hz["N"]
    s2 = pi * (1 - pi) / hz["N"]
    ex.update({"funnel_pi": pi, "funnel_hz_y": float(y), "funnel_hz_s2": float(s2), "funnel_tau2": tau2,
               "funnel_hz_z": float((y - pi) / np.sqrt(s2 + tau2)), "funnel_hz_z_plain": float((y - pi) / np.sqrt(s2))})

    # 7. Regional-medium share for Gujarat, by hand.
    reg = pd.read_csv(TABLES.parent.parent / "validation" / "nta_state_registrations_2024.csv").set_index("state")["registered_2024"]
    guj_states = ["GUJARAT", "DADRA AND NAGAR HAVELI (UT)", "DAMAN AND DIU (UT)"]
    w = reg["GUJARAT"] / reg[guj_states].sum()
    ex.update({"guj_registered": int(reg["GUJARAT"]), "guj_medium": 57765, "guj_weight": float(w),
               "guj_share": float(57765 * w / reg["GUJARAT"])})

    # 8. Lord's binomial standard error at 577 marks (all 180 attempted).
    c577 = (577 + 180) / 5
    ex.update({"lord_c": c577, "lord_w": 180 - c577, "lord_sem_c": float(np.sqrt(c577 * (180 - c577) / 179)),
               "lord_sem": float(5 * np.sqrt(c577 * (180 - c577) / 179))})

    # 9. The comb around 75 marks.
    counts = cand["marks"].value_counts()
    for s in range(73, 78):
        ex[f"comb_{s}"] = int(counts.get(s, 0))
    f = np.mean([counts.get(s, 0) for s in (73, 74, 76, 77)])
    ex["comb_75_excess_share"] = float((counts.get(75, 0) - f) / (counts.get(75, 0) + 4 * f))

    # 10. Serial-order statistic for one centre.
    x = cand[cand["code"] == 260402].sort_values("serial")["marks"].to_numpy().astype(float)
    xm = x - x.mean()
    r1 = float((xm[1:] * xm[:-1]).sum() / (xm * xm).sum())
    ex.update({"serial_oasis_r1": r1, "serial_oasis_z": (r1 + 1 / len(x)) * np.sqrt(len(x))})

    # 11. One injected centre fraud, by hand: a 400-candidate-sized centre in Hyderabad, 20 boosted.
    rng = np.random.default_rng(20261006)
    top_pool = cand["marks"].to_numpy()[cand["marks"].to_numpy() >= t1]
    hyd = cent[(cent["city"] == "HYDERABAD/SECUNDERABAD")].copy()
    target = hyd.iloc[(hyd["n"] - 400).abs().argsort()].iloc[0]
    cy = city[city["city_id"] == target["city_id"]].iloc[0]
    marks = cand.loc[cand["code"] == target["code"], "marks"].to_numpy().copy()
    idx = rng.choice(len(marks), size=20, replace=False)
    before = int((marks[idx] >= t1).sum())
    marks[idx] = rng.choice(top_pool, size=20)
    k_new = int((marks >= t1).sum())
    K_new = int(cy["top1"]) - before + 20
    n_t, N_t = int(target["n"]), int(cy["N"])
    p_ge, p_le, p_gt, p_eq = (float(v) for v in hypergeom_tails(k_new, N_t, K_new, n_t))
    z_mid = float(stats.norm.isf(p_gt + 0.5 * p_eq))
    deff = 1 + (n_t - 1) * scrn["rho1"]
    ex.update({"inj_centre": nice_name(target["name"]), "inj_centre_n": n_t, "inj_city_n": N_t, "inj_centre_top_before": int(target["top1"]),
               "inj_city_top_before": int(cy["top1"]), "inj_centre_top_after": k_new, "inj_city_top_after": K_new,
               "inj_E_after": n_t * K_new / N_t, "inj_z_mid": z_mid, "inj_z_adj": z_mid / np.sqrt(deff),
               "inj_p_adj": float(stats.norm.sf(z_mid / np.sqrt(deff))), "inj_bar": scrn["bh_cutoff_layer2"]})

    # 12. A city leak at Hazaribagh: 125 beneficiaries scoring like the top 1%, spread over its centres.
    hz_codes = cent.loc[cent["city"] == "HAZARIBAGH", "code"].tolist()
    hz = cand[cand["code"].isin(hz_codes)].reset_index(drop=True)
    pick = rng.choice(len(hz), size=125, replace=False)
    hz.loc[pick, "marks"] = rng.choice(top_pool, size=125)
    K_hz = int((hz["marks"] >= t1).sum())
    N_hz = len(hz)
    zs = []
    for code, gg in hz.groupby("code"):
        n_c, k_c = len(gg), int((gg["marks"] >= t1).sum())
        a, b, c_gt, c_eq = (float(v) for v in hypergeom_tails(k_c, N_hz, K_hz, n_c))
        zs.append(float(stats.norm.isf(c_gt + 0.5 * c_eq)) / np.sqrt(1 + (n_c - 1) * scrn["rho1"]))
    y_new = K_hz / N_hz
    ex.update({"leak_hz_city_top_after": K_hz, "leak_hz_max_centre_z": max(zs), "leak_hz_rate_after": y_new,
               "leak_hz_funnel_z_after": float((y_new - pi) / np.sqrt(pi * (1 - pi) / N_hz + tau2))})

    ex = {"ex_" + k: v for k, v in ex.items()}
    (TABLES / "examples_numbers.json").write_text(json.dumps(ex, indent=1, default=float))
    print(json.dumps(ex, indent=1, default=float))


if __name__ == "__main__":
    main()
