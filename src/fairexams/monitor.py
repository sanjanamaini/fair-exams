"""The integrity monitor: run the screen on any exam's centre-wise results.

    python -m fairexams.monitor --input marks.csv --out monitor_out
    python -m fairexams.monitor --simulate --out monitor_demo     # synthetic exam with one planted fraud

Input: a CSV with one row per candidate and three columns (names configurable): centre, city, marks.
No identities are needed. Output in --out: centres.csv (every centre's statistics), flags.csv,
summary.md (plain-language summary with the caveats a reader needs) and funnel.png.

What it runs (see PREREGISTRATION.md and DEVIATIONS.md of this repository):
1. Exact test of each centre's count of top scorers against the rest of its city (random allotment).
2. The same test widened by the Kish design effect, with the intra-class correlation estimated
   from the data after trimming the most extreme 1% of centres (allowance for non-random allotment).
3. Benjamini-Hochberg false discovery control across centres.
4. "Tail beyond the bulk": are there more top scorers than the centre's own middle half predicts?
   A stronger cohort moves the whole distribution; a boost to some candidates moves only the top.
Flags are screening results for review, never findings of misconduct.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .bulk import bulk_prediction
from .panels import top_threshold
from .stats import bh, central_matching, hypergeom_tails, mean_z, nested_icc, randomised_z

TRIM = 0.01
SEED = 20261006


def _one_seed(c, tails, fdr, seed):
    """Randomised z-scores (exactly normal under the null), rho, design effects and flags for one seed."""
    p_gt, p_eq, p_lt, t_gt, t_eq, t_lt = tails
    rng = np.random.default_rng(seed)
    z = randomised_z(p_gt, p_eq, rng.uniform(size=len(c)), p_lt=p_lt)
    keep = np.abs(z).argsort().argsort() < len(c) - int(TRIM * len(c))
    rho, _ = nested_icc(c.loc[keep, "k"], c.loc[keep, "k"], c.loc[keep, "n"], c.loc[keep, "city"])
    z_adj = z / np.sqrt(1 + (c["n"].to_numpy() - 1) * rho)
    q = bh(stats.norm.sf(z_adj))
    zt = randomised_z(t_gt, t_eq, rng.uniform(size=len(c)), p_lt=t_lt)
    d0, s0 = central_matching(zt)
    q_tail = bh(stats.norm.sf((zt - d0) / s0))
    return {"z": z, "rho": rho, "z_adj": z_adj, "q": q, "flag": q <= fdr, "z_tail": zt, "q_tail": q_tail,
            "flag_tail": q_tail <= fdr, "tail_null": (d0, s0)}


def run_screen(cand: pd.DataFrame, top_share: float = 0.01, fdr: float = 0.05, seeds=(SEED,)) -> tuple[pd.DataFrame, dict]:
    """cand has columns centre, city, marks. Returns (centre table, summary numbers).

    Discrete tests use randomised p-values, which are exactly uniform under the null; the result for the
    first seed is reported, and with several seeds each centre also gets the share of seeds that flag it.
    """
    cand = cand[["centre", "city", "marks"]].copy()
    t1 = top_threshold(cand["marks"].to_numpy(), top_share)
    cand["top"] = cand["marks"] >= t1
    g = cand.groupby("centre")
    c = pd.DataFrame({"city": g["city"].first(), "n": g.size(), "k": g["top"].sum(), "mean": g["marks"].mean(),
                      "ss": g["marks"].apply(lambda m: float((m.astype(float) ** 2).sum()))}).reset_index()
    gc = cand.groupby("city")
    city = pd.DataFrame({"N": gc.size(), "K": gc["top"].sum(), "city_mean": gc["marks"].mean(),
                         "city_var": gc["marks"].var(ddof=0), "centres": gc["centre"].nunique()})
    c = c.join(city, on="city")
    c = c[c["centres"] >= 2].reset_index(drop=True)
    p_ge, p_le, p_gt, p_eq = hypergeom_tails(c["k"], c["N"], c["K"], c["n"])
    c["E"] = c["n"] * c["K"] / c["N"]
    c["p_exact"] = p_ge
    c["q_exact"] = bh(p_ge)
    c["z_mean"] = mean_z(c["mean"], c["n"], c["city_mean"], c["city_var"], c["N"])

    # Tail beyond the bulk: the prediction is deterministic; only the randomisation varies by seed.
    by_city = {k: v for k, v in cand.groupby("city")}
    t_gt, t_eq, t_lt = (np.empty(len(c)) for _ in range(3))
    for i, r in enumerate(c.itertuples()):
        gg = by_city[r.city]
        mask = gg["centre"].to_numpy() == r.centre
        own, rest = gg["marks"].to_numpy()[mask], np.sort(gg["marks"].to_numpy()[~mask])
        share, _, _ = bulk_prediction(own, rest, t1)
        k, n = int((own >= t1).sum()), len(own)
        t_gt[i], t_eq[i] = stats.binom.sf(k, n, share), stats.binom.pmf(k, n, share)
        t_lt[i] = stats.binom.cdf(k - 1, n, share) if k > 0 else 0.0
    tails = (p_gt, p_eq, p_le - p_eq, t_gt, t_eq, t_lt)
    runs = [_one_seed(c, tails, fdr, s) for s in seeds]
    first = runs[0]
    for col in ("z", "z_adj", "q", "flag", "z_tail", "q_tail", "flag_tail"):
        c[col] = first[col]
    c["flag_rate"] = np.mean([r["flag"] for r in runs], axis=0)
    c["flag_tail_rate"] = np.mean([r["flag_tail"] for r in runs], axis=0)
    rho_m, _ = nested_icc(c["mean"] * c["n"], c["ss"], c["n"], c["city"])
    summary = {"candidates": int(len(cand)), "centres": int(cand["centre"].nunique()), "screened": int(len(c)),
               "cities": int(cand["city"].nunique()), "top_threshold": int(t1), "rho_top": first["rho"],
               "rho_marks_untrimmed": rho_m,
               "median_design_effect": float((1 + (c["n"] - 1) * first["rho"]).median()), "flags": int(c["flag"].sum()),
               "flags_exact": int((c["q_exact"] <= fdr).sum()), "flags_tail": int(c["flag_tail"].sum()),
               "tail_null": first["tail_null"], "fdr": fdr, "top_share": top_share, "seeds": len(seeds)}
    return c.sort_values("z_adj", ascending=False), summary


def write_outputs(c: pd.DataFrame, s: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cols = ["centre", "city", "n", "k", "E", "z", "z_adj", "q_exact", "q", "flag", "z_tail", "q_tail", "flag_tail", "flag_rate", "flag_tail_rate", "z_mean"]
    c[cols].to_csv(out / "centres.csv", index=False, float_format="%.5g")
    c.loc[c["flag"] | c["flag_tail"], cols].to_csv(out / "flags.csv", index=False, float_format="%.5g")
    both = c[c["flag"] & c["flag_tail"]]
    lines = [
        "# Integrity monitor summary", "",
        "- Candidates: %d in %d centres and %d cities; %d centres in cities with two or more centres were screened." %
        (s["candidates"], s["centres"], s["cities"], s["screened"]),
        "- Top group: marks of %d or more (national top %.1f%%)." % (s["top_threshold"], 100 * s["top_share"]),
        "- How random is allotment? Intra-class correlation of top-group status within cities: %.4f (median design effect %.2f). For average marks: %.4f." %
        (s["rho_top"], s["median_design_effect"], s["rho_marks_untrimmed"]),
        "- Centres with more top scorers than random allotment allows, after the design effect (q <= %.2f): **%d** (exact test without the design effect: %d)." %
        (s["fdr"], s["flags"], s["flags_exact"]),
        "- Centres with more top scorers than their own bulk predicts (tail beyond the bulk): **%d**." % s["flags_tail"],
        "- Centres on both lists (the pattern a boost to some candidates would leave): **%d**." % len(both), "",
        "## How to read a flag", "",
        "A flag means a centre's results are unlikely under random allotment of candidates within its city. It is a reason to review, never evidence of misconduct. "
        "Innocent explanations include candidates from one school or coaching institute being allotted together, allotment by paper medium, and data errors. "
        "A flag on the first list only, with the whole distribution shifted, usually means a stronger cohort was allotted together. "
        "Leaks sold before the exam to candidates spread across a city cannot be seen by any within-city test; they need between-city comparisons or item-level answer-similarity analysis.", "",
        "## Flagged centres", "",
        "Flag rates are the share of %d random seeds that flag the centre (discrete tests use randomised p-values)." % s["seeds"], "",
        "| centre | city | candidates | top | expected | z (adjusted) | q | flag rate | z (tail beyond bulk) | q (tail) | tail flag rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in c[c["flag"] | c["flag_tail"]].itertuples():
        lines.append("| %s | %s | %d | %d | %.1f | %.2f | %.2g | %.0f%% | %.2f | %.2g | %.0f%% |" % (
            r.centre, r.city, r.n, r.k, r.E, r.z_adj, r.q, 100 * r.flag_rate, r.z_tail, r.q_tail, 100 * r.flag_tail_rate))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.scatter(c["E"], c["k"] - c["E"], s=6, color=np.where(c["flag"], "#C9A227", "#1F3A5F"), alpha=0.7)
    e = np.linspace(max(c["E"].min(), 0.1), c["E"].max(), 200)
    zc = stats.norm.isf(0.001)
    deff = float(s["median_design_effect"])
    ax.plot(e, zc * np.sqrt(e * deff), color="#999", lw=1)
    ax.plot(e, -zc * np.sqrt(e * deff), color="#999", lw=1)
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_xlabel("Expected top scorers under random allotment")
    ax.set_ylabel("Observed minus expected")
    fig.tight_layout()
    fig.savefig(out / "funnel.png", dpi=160)
    plt.close(fig)


def simulate(seed: int = 7) -> pd.DataFrame:
    """A synthetic exam: 40 cities, sorting by medium in half of them, one centre with 30 boosted candidates."""
    rng = np.random.default_rng(seed)
    rows = []
    for city in range(40):
        city_effect = rng.normal(0, 0.3)
        sorted_city = city % 2 == 0
        for centre in range(int(rng.integers(3, 15))):
            n = int(rng.integers(150, 800))
            shift = rng.normal(0, 0.35) if sorted_city else 0.0
            ability = rng.normal(city_effect + shift, 1, size=n)
            marks = np.clip(np.round(220 + 165 * ability), -40, 720)
            rows.append(pd.DataFrame({"centre": "C%02d-%02d" % (city, centre), "city": "City %02d" % city, "marks": marks}))
    df = pd.concat(rows, ignore_index=True)
    target = df.index[df["centre"] == "C07-01"][:30]
    df.loc[target, "marks"] = rng.choice(np.arange(660, 721), size=len(target))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input")
    ap.add_argument("--centre", default="centre")
    ap.add_argument("--city", default="city")
    ap.add_argument("--marks", default="marks")
    ap.add_argument("--top-share", type=float, default=0.01)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--out", default="monitor_out")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--seeds", type=int, default=50, help="seeds for the flag-stability check (first seed is reported)")
    a = ap.parse_args()
    if a.simulate:
        cand = simulate()
    else:
        cand = pd.read_csv(a.input).rename(columns={a.centre: "centre", a.city: "city", a.marks: "marks"})
    c, s = run_screen(cand, a.top_share, a.fdr, seeds=tuple(SEED + i for i in range(a.seeds)))
    write_outputs(c, s, Path(a.out))
    print((Path(a.out) / "summary.md").read_text())


if __name__ == "__main__":
    main()
