"""Stage 3: the integrity screen (PREREGISTRATION.md, Strand 1).

    python -m fairexams.screen

Every centre in a city with two or more centres is tested against the rest of its city.
Layer 1: exact hypergeometric test of its top-1% count under random allotment.
Layer 2: the same z-score deflated by the Kish design effect 1 + (n - 1) rho, with rho the
intra-class correlation of top-1% status within cities, estimated after trimming the most extreme 1%.
Flags: Layer-2 excess with Benjamini-Hochberg q <= 0.05.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import FDR_Q, FIGURES, INTERIM, SEED, SIZE_BANDS, TABLES, TRIM_FOR_RHO
from .stats import (bh, bh_cutoff, central_matching, hypergeom_tails, mean_z, nested_icc, randomised_z,
                    storey_pi0)

BAND_LABELS = ["<250", "250-499", "500-749", ">=750"]


def load_panels():
    cand = pd.read_parquet(INTERIM / "candidates.parquet")
    cent = pd.read_parquet(INTERIM / "centres.parquet")
    city = pd.read_parquet(INTERIM / "cities.parquet")
    nums = json.loads((TABLES / "panel_numbers.json").read_text())
    return cand, cent, city, nums


def centre_frame(cand, cent, city):
    """Centres in multi-centre cities with their own and their city's aggregates."""
    sq = cand.assign(m2=cand["marks"].astype("int64") ** 2).groupby("code")["m2"].sum().rename("marks_ss")
    df = cent.merge(sq, left_on="code", right_index=True)
    df = df.merge(city[["city_id", "N", "mean", "sd", "top1", "top01", "ge_med"]]
                  .rename(columns={"mean": "city_mean", "sd": "city_sd", "top1": "K1", "top01": "K01",
                                   "ge_med": "Kmed"}), on="city_id")
    df = df[df["n_centres_city"] >= 2].copy()
    df["band"] = pd.cut(df["n"], SIZE_BANDS, right=False, labels=BAND_LABELS)
    return df.reset_index(drop=True)


def layer1(df, rng):
    out = {}
    for stat, k, K in (("1", "top1", "K1"), ("01", "top01", "K01"), ("med", "ge_med", "Kmed")):
        p_ge, p_le, p_gt, p_eq = hypergeom_tails(df[k], df["N"], df[K], df["n"])
        out["p%s_ge" % stat] = p_ge
        out["p%s_le" % stat] = p_le
        out["z%s" % stat] = randomised_z(p_gt, p_eq, rng.uniform(size=len(df)), p_lt=p_le - p_eq)
        out["E%s" % stat] = df["n"] * df[K] / df["N"]
    out["zmean"] = mean_z(df["mean"], df["n"], df["city_mean"], df["city_sd"] ** 2, df["N"])
    return pd.DataFrame(out, index=df.index)


def estimate_rho(df, z, y_sum, y_ss, trim):
    keep = np.abs(z).rank(ascending=False, method="first") > trim * len(z)
    rho, det = nested_icc(y_sum[keep], y_ss[keep], df.loc[keep, "n"], df.loc[keep, "city_id"])
    return rho, det, int((~keep).sum())


def ks_by_centre(cand, df):
    """Kolmogorov-Smirnov distance between each centre and the rest of its city."""
    marks_by_city = {c: g for c, g in cand[cand["city_id"].isin(df["city_id"])].groupby("city_id")}
    d, p = np.empty(len(df)), np.empty(len(df))
    for i, (code, cid) in enumerate(zip(df["code"], df["city_id"])):
        g = marks_by_city[cid]
        own = g["marks"].to_numpy()[g["code"].to_numpy() == code]
        rest = g["marks"].to_numpy()[g["code"].to_numpy() != code]
        r = stats.ks_2samp(own, rest, method="asymp")
        d[i], p[i] = r.statistic, r.pvalue
    return d, p


def main() -> None:
    cand, cent, city, nums = load_panels()
    rng = np.random.default_rng(SEED)
    df = centre_frame(cand, cent, city)
    df = df.join(layer1(df, rng))
    numbers = {"screen_centres": int(len(df)), "screen_cities": int(df["city_id"].nunique()),
               "screen_candidates": int(df["n"].sum()),
               "single_centre_cities": int(city.loc[city["centres"] == 1].shape[0]),
               "single_centre_candidates": int(city.loc[city["centres"] == 1, "N"].sum())}

    # Layer 2: design effects from the trimmed intra-class correlations.
    rho1, det1, trimmed = estimate_rho(df, df["z1"], df["top1"], df["top1"], TRIM_FOR_RHO)
    rho_m, detm, _ = estimate_rho(df, df["zmean"], df["mean"] * df["n"], df["marks_ss"], TRIM_FOR_RHO)
    for t in (0.0, 0.05):
        r, _, _ = estimate_rho(df, df["z1"], df["top1"], df["top1"], t)
        numbers["rho1_trim%s" % int(t * 100)] = r
    df["deff1"] = 1 + (df["n"] - 1) * rho1
    df["deffm"] = 1 + (df["n"] - 1) * rho_m
    df["z1_adj"] = df["z1"] / np.sqrt(df["deff1"])
    df["zmean_adj"] = df["zmean"] / np.sqrt(df["deffm"])
    df["p1_adj_hi"] = stats.norm.sf(df["z1_adj"])
    df["p1_adj_lo"] = stats.norm.cdf(df["z1_adj"])
    df["q_excess"] = bh(df["p1_adj_hi"])
    df["q_deficit"] = bh(df["p1_adj_lo"])
    df["q_layer1"] = bh(df["p1_ge"])
    df["q_layer1_lo"] = bh(df["p1_le"])
    df["q01_layer1"] = bh(df["p01_ge"])
    df["flag"] = df["q_excess"] <= FDR_Q
    df["ks_d"], df["ks_p"] = ks_by_centre(cand, df)
    df["city_rank_z1"] = df.groupby("city_id")["z1"].rank(ascending=False, method="min").astype(int)

    numbers.update({"rho1": rho1, "rho_marks": rho_m, "rho_trimmed_centres": trimmed,
                    "rho1_n0": det1["n0"], "deff1_median": float(df["deff1"].median()),
                    "deffm_median": float(df["deffm"].median()),
                    "flags_layer2": int(df["flag"].sum()),
                    "deficits_layer2": int((df["q_deficit"] <= FDR_Q).sum()),
                    "flags_layer1": int((df["q_layer1"] <= FDR_Q).sum()),
                    "deficits_layer1": int((df["q_layer1_lo"] <= FDR_Q).sum()),
                    "flags01_layer1": int((df["q01_layer1"] <= FDR_Q).sum()),
                    "bh_cutoff_layer2": bh_cutoff(df["p1_adj_hi"], FDR_Q),
                    "pi0_layer1": storey_pi0(df["p1_ge"]), "pi0_layer2": storey_pi0(df["p1_adj_hi"]),
                    "z1_sd": float(df["z1"].std()), "zmean_sd": float(df["zmean"].std()),
                    "zmean_adj_sd": float(df["zmean_adj"].std()), "z1_adj_sd": float(df["z1_adj"].std())})

    # Check on the null: Efron central matching by size band, against the sorting model's prediction.
    rows = []
    for band, g in df.groupby("band", observed=True):
        d0, s0 = central_matching(g["z1"])
        dm, sm = central_matching(g["zmean"])
        rows.append({"band": band, "centres": len(g), "mean_n": g["n"].mean(),
                     "delta0_top1": d0, "sigma0_top1": s0, "sigma0_top1_pred": np.sqrt(1 + (g["n"].mean() - 1) * rho1),
                     "delta0_mean": dm, "sigma0_mean": sm, "sigma0_mean_pred": np.sqrt(1 + (g["n"].mean() - 1) * rho_m)})
    nullb = pd.DataFrame(rows)
    nullb.to_csv(TABLES / "null_by_size.csv", index=False)
    d0, s0 = central_matching(df["z1"])
    numbers.update({"efron_delta0_top1": d0, "efron_sigma0_top1": s0})
    d0, s0 = central_matching(df["zmean"])
    numbers.update({"efron_delta0_mean": d0, "efron_sigma0_mean": s0})
    slope = np.polyfit(nullb["mean_n"] - 1, nullb["sigma0_mean"] ** 2 - 1, 1)[0]
    numbers["efron_slope_mean"] = float(slope)
    slope1 = np.polyfit(nullb["mean_n"] - 1, nullb["sigma0_top1"] ** 2 - 1, 1)[0]
    numbers["efron_slope_top1"] = float(slope1)

    # Asymmetry: sorting widens both tails; fraud adds only to the top.
    za = df["zmean_adj"]
    numbers.update({"asym_mean_above3": int((za > 3).sum()), "asym_mean_below3": int((za < -3).sum()),
                    "asym_mean_above2": int((za > 2).sum()), "asym_mean_below2": int((za < -2).sum()),
                    "asym_top1_above3": int((df["z1_adj"] > 3).sum()), "asym_top1_below3": int((df["z1_adj"] < -3).sum())})
    numbers["asym_mean_binom_p"] = float(stats.binomtest(numbers["asym_mean_above3"],
                                                         numbers["asym_mean_above3"] + numbers["asym_mean_below3"]).pvalue) \
        if numbers["asym_mean_above3"] + numbers["asym_mean_below3"] > 0 else 1.0

    keep_cols = ["code", "state", "city", "name", "n", "N", "n_centres_city", "band", "top1", "E1", "p1_ge", "z1",
                 "deff1", "z1_adj", "q_layer1", "q_excess", "q_deficit", "flag", "top01", "E01", "p01_ge", "z01",
                 "q01_layer1", "ge_med", "Emed", "zmed", "mean", "city_mean", "zmean", "zmean_adj", "ks_d", "ks_p",
                 "city_rank_z1", "max", "min"]
    out = df[keep_cols].sort_values("z1_adj", ascending=False)
    out.to_csv(TABLES / "screen_centres.csv", index=False, float_format="%.6g")
    out[out["flag"]].to_csv(TABLES / "flags.csv", index=False, float_format="%.6g")
    out[out["q_deficit"] <= FDR_Q].to_csv(TABLES / "deficits.csv", index=False, float_format="%.6g")
    (TABLES / "screen_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    plots(df, rho1, nullb)
    print(json.dumps(numbers, indent=1, default=float))
    print(nullb.to_string())
    print(out.head(25)[["code", "city", "name", "n", "top1", "E1", "z1", "z1_adj", "q_layer1", "q_excess", "zmean_adj"]].to_string())


def plots(df, rho1, nullb):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(-6, 10, 81)
    ax.hist(df["z1"], bins=bins, density=True, color="#9DB4C0", label="Layer 1 z (exact test)")
    ax.hist(df["z1_adj"], bins=bins, density=True, histtype="step", color="#1F3A5F", lw=1.4,
            label="Layer 2 z (design effect)")
    x = np.linspace(-6, 6, 400)
    ax.plot(x, stats.norm.pdf(x), color="#C9A227", lw=1.4, label="N(0, 1): random allotment")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("Centre z-score for top-1% scorers, relative to its own city")
    ax.set_ylabel("Density (log scale)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "screen_z_hist.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for col, pred, colr, lab in (("sigma0_top1", "sigma0_top1_pred", "#1F3A5F", "top-1% count"),
                                 ("sigma0_mean", "sigma0_mean_pred", "#C9A227", "mean marks")):
        ax.plot(nullb["mean_n"], nullb[col] ** 2, "o", color=colr, label="%s: empirical null (Efron)" % lab)
        ax.plot(nullb["mean_n"], nullb[pred] ** 2, "-", color=colr, alpha=0.7, label="%s: design-effect model" % lab)
    ax.axhline(1, color="#999", lw=1)
    ax.set_xlabel("Average candidates per centre (size band)")
    ax.set_ylabel("Null variance of centre z-scores")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "screen_null_by_size.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
