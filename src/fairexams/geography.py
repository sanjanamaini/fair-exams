"""Stage 4: geography and development (PREREGISTRATION.md, Strand 2).

    python -m fairexams.geography

1. Concentration of top-1% scorers across exam cities (top-ten shares, concentration curve, Gini).
2. Funnel plot of city top-1% rates (Spiegelhalter 2005), limits widened for overdispersion;
   cities above the upper 99.8% limit are hub cities.
3. Hub or cluster: heterogeneity of each hub city's centres (sum of squared Layer-1 z-scores).
4. Development: cities mapped to NFHS-5 districts; Spearman correlations of the development index
   with mean marks, top-1% rate and share at or above the national median, bootstrap intervals.
5. Positive deviance: units whose top-1% rate sits more than 2 SD above the development prediction.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import COACHING_HUBS, FIGURES, INTERIM, N_BOOT, SEED, TABLES
from .stats import random_effects_tau2

FUNNEL_Z = stats.norm.isf(0.001)


def concentration(city: pd.DataFrame) -> dict:
    c = city.sort_values("rate")
    x = np.concatenate([[0], np.cumsum(c["N"]) / c["N"].sum()])
    y = np.concatenate([[0], np.cumsum(c["top1"]) / c["top1"].sum()])
    gini = 1 - 2 * np.trapz(y, x)
    top = city.sort_values("top1", ascending=False).head(10)
    return {"gini": float(gini), "top10_share_top1": float(top["top1"].sum() / city["top1"].sum()),
            "top10_share_candidates": float(top["N"].sum() / city["N"].sum()), "curve": (x, y), "top10": top}


def funnel(city: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    pi = float(city["top1"].sum() / city["N"].sum())
    y, s2 = city["top1"] / city["N"], pi * (1 - pi) / city["N"]
    tau2, _, _ = random_effects_tau2(y.to_numpy(), s2.to_numpy())
    city = city.assign(z_funnel=(y - pi) / np.sqrt(s2 + tau2), z_plain=(y - pi) / np.sqrt(s2))
    city["hub"] = city["z_funnel"] > FUNNEL_Z
    city["cold"] = city["z_funnel"] < -FUNNEL_Z
    return city, pi, tau2


def hub_or_cluster(city: pd.DataFrame, scr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in city[city["hub"]].itertuples():
        s = scr[(scr["state"] == r.state) & (scr["city"] == r.city)]
        if len(s) < 2:
            rows.append({"city": r.city, "state": r.state, "centres": len(s), "Q": np.nan, "df": 0, "p_hetero": np.nan})
            continue
        Q = float((s["z1"] ** 2).sum())
        excess = (s["top1"] - s["E1"]).clip(lower=0).sort_values(ascending=False)
        rows.append({"city": r.city, "state": r.state, "centres": len(s), "Q": Q, "df": len(s) - 1,
                     "p_hetero": float(stats.chi2.sf(Q, len(s) - 1)),
                     "share_centres_above_city_rate": float((s["top1"] / s["n"] > r.top1 / r.N).mean()),
                     "flagged_centres": int((s["q_excess"] <= 0.05).sum())})
    return pd.DataFrame(rows)


def development_units(city: pd.DataFrame, cmap: pd.DataFrame, dev: pd.DataFrame) -> pd.DataFrame:
    """One unit per district (or per multi-district city), with outcomes pooled over its exam cities."""
    devk = dev.set_index(["state", "district"])
    m = city.merge(cmap[["city_id", "nfhs_state", "districts", "kind"]], on="city_id")
    m = m[m["kind"] != "none"].copy()
    m["unit"] = m["nfhs_state"] + " | " + m["districts"]
    cols = ["women_literate", "women_school10", "electricity", "sanitation", "clean_fuel", "dev_index"]
    devrows = []
    for u, g in m.groupby("unit"):
        st, ds = u.split(" | ")
        vals = devk.loc[[(st, d) for d in ds.split(";")], cols].mean()
        devrows.append({"unit": u, **vals.to_dict()})
    agg = m.groupby("unit").agg(N=("N", "sum"), top1=("top1", "sum"), ge_med=("ge_med", "sum"),
                                marks_sum=("marks_sum", "sum"), cities=("city", lambda s: "; ".join(s)),
                                kind=("kind", lambda s: "exact" if set(s) <= {"exact", "alias", "town"} else "uncertain"))
    out = agg.reset_index().merge(pd.DataFrame(devrows), on="unit")
    out["mean_marks"] = out["marks_sum"] / out["N"]
    out["top1_rate"] = out["top1"] / out["N"]
    out["ge_med_share"] = out["ge_med"] / out["N"]
    out["hub_city"] = out["cities"].apply(lambda s: any(h in s.split("; ") for h in COACHING_HUBS))
    return out


def spearman_ci(x, y, rng, n_boot):
    x, y = np.asarray(x), np.asarray(y)
    rho = stats.spearmanr(x, y)[0]
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stats.spearmanr(x[i], y[i])[0] for i in idx])
    return float(rho), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def positive_deviance(units: pd.DataFrame) -> pd.DataFrame:
    u = units.dropna(subset=["dev_index"]).copy()
    y = np.log((u["top1"] + 0.5) / (u["N"] - u["top1"] + 0.5))
    w = u["N"]
    X = np.column_stack([np.ones(len(u)), u["dev_index"]])
    beta = np.linalg.lstsq(X * np.sqrt(w.to_numpy())[:, None], y * np.sqrt(w), rcond=None)[0]
    pred = 1 / (1 + np.exp(-(X @ beta)))
    s2 = pred * (1 - pred) / u["N"]
    tau2, _, _ = random_effects_tau2((u["top1_rate"] - pred).to_numpy(), s2.to_numpy())
    u["pred_rate"] = pred
    u["z_dev"] = (u["top1_rate"] - pred) / np.sqrt(s2 + tau2)
    u.attrs["beta"], u.attrs["tau2"] = beta, tau2
    return u


def main() -> None:
    city = pd.read_parquet(INTERIM / "cities.parquet")
    cand = pd.read_parquet(INTERIM / "candidates.parquet")
    city = city.merge(cand.groupby("city_id")["marks"].sum().rename("marks_sum"), left_on="city_id", right_index=True)
    city = city[~city["overseas"]].copy()
    city["rate"] = city["top1"] / city["N"]
    scr = pd.read_csv(TABLES / "screen_centres.csv")
    cmap = pd.read_parquet(INTERIM / "city_district.parquet")
    dev = pd.read_parquet(INTERIM / "district_dev.parquet")
    rng = np.random.default_rng(SEED + 3)
    numbers = {}

    conc = concentration(city)
    numbers.update({"geo_gini_top1": conc["gini"], "geo_top10_share_top1": conc["top10_share_top1"],
                    "geo_top10_share_candidates": conc["top10_share_candidates"], "geo_cities": int(len(city))})
    top10 = conc["top10"][["state", "city", "N", "top1", "rate"]]
    top10.to_csv(TABLES / "geo_top10_cities.csv", index=False, float_format="%.5g")

    city, pi, tau2 = funnel(city)
    numbers.update({"geo_pi": pi, "geo_tau2": tau2, "geo_hubs": int(city["hub"].sum()), "geo_cold": int(city["cold"].sum()),
                    "geo_hub_candidates_share": float(city.loc[city["hub"], "N"].sum() / city["N"].sum()),
                    "geo_hub_top1_share": float(city.loc[city["hub"], "top1"].sum() / city["top1"].sum()),
                    "geo_plain_outside_998": int((city["z_plain"].abs() > FUNNEL_Z).sum())})
    city.sort_values("z_funnel", ascending=False).to_csv(TABLES / "geo_cities.csv", index=False, float_format="%.5g")
    hubs = hub_or_cluster(city, scr)
    hubs.to_csv(TABLES / "geo_hubs.csv", index=False, float_format="%.5g")
    if len(hubs):
        numbers.update({"geo_hubs_heterogeneous": int((hubs["p_hetero"] < 0.05).sum()),
                        "geo_hubs_tested": int(hubs["p_hetero"].notna().sum())})
    for h in COACHING_HUBS:
        r = city[city["city"] == h].iloc[0]
        hb = hubs[hubs["city"] == h]
        numbers.update({"geo_%s_N" % h.lower(): int(r["N"]), "geo_%s_top1" % h.lower(): int(r["top1"]),
                        "geo_%s_rate" % h.lower(): float(r["rate"]), "geo_%s_z" % h.lower(): float(r["z_funnel"]),
                        "geo_%s_centres" % h.lower(): int(r["centres"])})
        if len(hb):
            numbers.update({"geo_%s_p_hetero" % h.lower(): float(hb["p_hetero"].iloc[0]),
                            "geo_%s_share_centres_above" % h.lower(): float(hb["share_centres_above_city_rate"].iloc[0])})

    units = development_units(city, cmap, dev)
    units.to_csv(TABLES / "geo_units.csv", index=False, float_format="%.5g")
    rows = []
    variants = {"all units": units, "certain mappings only": units[units["kind"] == "exact"],
                "without Kota and Sikar": units[~units["hub_city"]],
                "units with 1,000+ candidates": units[units["N"] >= 1000]}
    for vname, u in variants.items():
        for outcome in ("mean_marks", "top1_rate", "ge_med_share"):
            rho, lo, hi = spearman_ci(u["dev_index"], u[outcome], rng, N_BOOT)
            rows.append({"variant": vname, "outcome": outcome, "units": len(u), "rho": rho, "lo": lo, "hi": hi})
        for ind in ("women_literate", "women_school10", "electricity", "sanitation", "clean_fuel"):
            rho = stats.spearmanr(u[ind], u["top1_rate"])[0]
            rows.append({"variant": vname, "outcome": "top1_rate vs " + ind, "units": len(u), "rho": float(rho),
                         "lo": np.nan, "hi": np.nan})
    corr = pd.DataFrame(rows)
    corr.to_csv(TABLES / "geo_development.csv", index=False, float_format="%.4g")
    for r in corr[corr["variant"] == "all units"].itertuples():
        key = "geo_dev_" + r.outcome.replace(" vs ", "_vs_")
        numbers[key + "_rho"] = r.rho
        if not np.isnan(r.lo):
            numbers[key + "_lo"], numbers[key + "_hi"] = r.lo, r.hi
    for vname in ("certain mappings only", "without Kota and Sikar", "units with 1,000+ candidates"):
        r = corr[(corr["variant"] == vname) & (corr["outcome"] == "top1_rate")].iloc[0]
        numbers["geo_dev_top1_rate_rho_%s" % vname.split()[0].lower()] = r.rho
    numbers["geo_units"] = int(len(units))

    pdv = positive_deviance(units)
    pdv.sort_values("z_dev", ascending=False).to_csv(TABLES / "geo_positive_deviance.csv", index=False, float_format="%.5g")
    numbers.update({"geo_pd_slope": float(pdv.attrs["beta"][1]), "geo_pd_tau2": float(pdv.attrs["tau2"]),
                    "geo_pd_positive": int((pdv["z_dev"] > 2).sum()), "geo_pd_negative": int((pdv["z_dev"] < -2).sum()),
                    "geo_pd_odds_ratio_per_sd": float(np.exp(pdv.attrs["beta"][1] * pdv["dev_index"].std()))})
    plots(city, pi, tau2, conc, pdv)
    (TABLES / "geography_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(json.dumps(numbers, indent=1, default=float))
    print(top10.to_string())
    print(city.sort_values("z_funnel", ascending=False).head(25)[["state", "city", "N", "top1", "rate", "z_funnel", "centres"]].to_string())
    print(hubs.to_string())
    print(corr.to_string())
    print(pdv.sort_values("z_dev", ascending=False).head(15)[["unit", "cities", "N", "top1_rate", "pred_rate", "z_dev", "dev_index"]].to_string())


def plots(city, pi, tau2, conc, pdv):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    n = np.logspace(np.log10(city["N"].min()), np.log10(city["N"].max()), 300)
    for z, ls in ((stats.norm.isf(0.025), ":"), (FUNNEL_Z, "-")):
        ax.plot(n, 100 * (pi + z * np.sqrt(pi * (1 - pi) / n + tau2)), color="#999", ls=ls, lw=1)
        ax.plot(n, 100 * np.clip(pi - z * np.sqrt(pi * (1 - pi) / n + tau2), 0, None), color="#999", ls=ls, lw=1)
    ax.scatter(city["N"], 100 * city["rate"], s=8, color=np.where(city["hub"], "#C9A227", "#1F3A5F"), alpha=0.8)
    offsets = {"SIKAR": (4, -2), "NAMAKKAL": (4, 0), "KOTTAYAM": (4, 0), "KOTA": (4, -6), "JAIPUR": (4, -6),
               "DELHI/NEW DELHI": (-30, 6), "PATNA": (4, -8), "HAZARIBAGH": (4, -8)}
    for name, (dx, dy) in offsets.items():
        r = city[city["city"] == name].iloc[0]
        ax.annotate(name.split("/")[0].title(), (r["N"], 100 * r["rate"]), fontsize=6.5, xytext=(dx, dy),
                    textcoords="offset points")
    ax.axhline(100 * pi, color="#1F3A5F", lw=0.8)
    ax.plot([], [], color="#999", ls=":", lw=1, label="95% limits")
    ax.plot([], [], color="#999", ls="-", lw=1, label="99.8% limits (with overdispersion)")
    ax.scatter([], [], s=8, color="#C9A227", label="hub city (above 99.8%)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_xscale("log")
    ax.set_xlabel("Candidates in the exam city (log scale)")
    ax.set_ylabel("Share in the national top 1% (%)")
    fig.tight_layout()
    fig.savefig(FIGURES / "geo_funnel.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True)
    hub = pdv["hub_city"]
    size = np.sqrt(pdv["N"]) / 4
    for ax, col, scale, lab in ((axes[0], "mean_marks", 1, "Mean marks"),
                                (axes[1], "top1_rate", 100, "Share in the national top 1% (%)")):
        ax.scatter(pdv["dev_index"], scale * pdv[col], s=size, color="#1F3A5F", alpha=0.45, lw=0)
        ax.scatter(pdv.loc[hub, "dev_index"], scale * pdv.loc[hub, col], s=26, color="#C9A227")
        rho = stats.spearmanr(pdv["dev_index"], pdv[col])[0]
        ax.set_title("%s: Spearman %.2f" % (lab.split(" (")[0], rho), fontsize=8)
        ax.set_ylabel(lab, fontsize=7)
        ax.tick_params(labelsize=6)
    for r in pdv.sort_values("z_dev", ascending=False).head(4).itertuples():
        axes[1].annotate(r.cities.split("; ")[0].title()[:12], (r.dev_index, 100 * r.top1_rate), fontsize=6,
                         xytext=(4, -2), textcoords="offset points")
    fig.supxlabel("District development index (NFHS-5; Kota and Sikar in gold)", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "geo_development.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
