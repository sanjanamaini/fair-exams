"""Write the report's tables (report/tables/*.tex) straight from results/tables/*.csv.

    python -m fairexams.report_tables
"""
from __future__ import annotations

import pandas as pd

from .config import REPORT, TABLES
from .text import nice_name


def n(x) -> str:
    return "" if pd.isna(x) else "{:,}".format(int(round(x)))


def d(x, k=2) -> str:
    return "" if pd.isna(x) else "%.*f" % (k, x)


def q(x) -> str:
    return "" if pd.isna(x) else ("$<$0.0001" if x < 0.0001 else ("%.4f" % x if x < 0.01 else "%.3f" % x))


def pc(x, k=0) -> str:
    return "" if pd.isna(x) else "%.*f\\%%" % (k, 100 * x)


def esc(s) -> str:
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", " ").replace("#", "\\#")


def title(s: str) -> str:
    return esc(nice_name(s))


def tabular(df: pd.DataFrame, align: str, header: list, size: str = "\\small", fit: bool = False) -> str:
    lines = ["{%s" % size, "\\setlength{\\tabcolsep}{4pt}" if fit else "",
             ("\\resizebox{\\linewidth}{!}{" if fit else "") + "\\begin{tabular}{%s}" % align,
             "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(str(v) for v in r.tolist()) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}" + ("}" if fit else "") + "}"]
    return "\n".join(line for line in lines if line) + "\n"


def main() -> None:
    out = REPORT / "tables"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Published figures reproduced.
    cc = pd.read_csv(TABLES / "court_check.csv")
    labels = {("HAZARIBAGH", "appeared"): "Hazaribagh: candidates", ("HAZARIBAGH", "within_rank_108000"): "Hazaribagh: 577 marks or more",
              ("HAZARIBAGH", "within_rank_56000"): "Hazaribagh: 622 marks or more", ("PATNA", "appeared"): "Patna: candidates",
              ("PATNA", "centres"): "Patna: centres", ("PATNA", "within_rank_108000"): "Patna: 577 marks or more",
              ("PATNA", "within_rank_56000"): "Patna: 622 marks or more", ("GODHRA", "appeared"): "Godhra: candidates",
              ("GODHRA", "centres"): "Godhra: centres", ("GODHRA", "within_rank_108000"): "Godhra: 577 marks or more",
              ("GODHRA", "within_rank_56000"): "Godhra: 622 marks or more",
              ("ALL", "marks_at_rank_108000"): "Marks at rank 108,000", ("ALL", "marks_at_rank_56000"): "Marks at rank 56,000"}
    rows = []
    for r in cc.itertuples():
        if (r.city, r.item) in labels:
            rows.append({"Figure": labels[(r.city, r.item)], "Source": "Supreme Court, paras 74-76, 81", "Published": n(r.court), "This file": n(r.ours)})
    pub = pd.read_csv(TABLES / "validation_published.csv")
    plabels = {"appeared": ("Candidates who appeared", "NTA, 26 July 2024"), "centres": ("Centres", "NTA, 26 July 2024"),
               "candidates_700_or_more": ("Candidates on 700 or more", "Ministry of Education, July 2024"),
               "candidates_on_720_before_revision": ("Candidates on 720 (before the revision)", "NTA, June 2024"),
               "cities_including_abroad": ("Cities, including 14 abroad", "NTA, 26 July 2024")}
    for r in pub.itertuples():
        lab, src = plabels[r.item]
        note = "" if r.item != "cities_including_abroad" else "$^{a}$"
        rows.append({"Figure": lab, "Source": src, "Published": n(r.value), "This file": n(r.ours) + note})
    (out / "validation.tex").write_text(tabular(pd.DataFrame(rows), "llrr", ["Figure", "Source", "Published", "This file"]))

    # 2. Registered flags.
    scr = pd.read_csv(TABLES / "screen_centres.csv")
    tail = pd.read_csv(TABLES / "bulk_tail.csv")[["code", "tail_E", "z_tail", "q_tail"]]
    prof = pd.read_csv(TABLES / "flag_profiles.csv")[["code", "lower_half_shift"]]
    stab = pd.read_csv(TABLES / "flag_stability.csv")[["centre", "flag_rate"]].rename(columns={"centre": "code"})
    f = scr[scr["flag"]].merge(tail, on="code").merge(prof, on="code").merge(stab, on="code", how="left")
    f = f.sort_values("z1_adj", ascending=False)
    t = pd.DataFrame({"Centre": f["name"].map(title), "City": f["city"].map(lambda s: esc(s).title()),
                      "n": f["n"].map(n), "Top 1\\%": f["top1"].map(n), "Exp.": f["E1"].map(lambda x: d(x, 1)),
                      "z": f["z1_adj"].map(d), "q": f["q_excess"].map(q), "Seeds": f["flag_rate"].map(pc),
                      "Shift": f["lower_half_shift"].map(lambda x: "%+.0f" % x),
                      "Bulk pred.": f["tail_E"].map(lambda x: d(x, 1)), "Tail z": f["z_tail"].map(d)})
    (out / "flags.tex").write_text(tabular(t, "p{4.6cm}lrrrrrrrrr", ["Centre", "City", "$n$", "Top 1\\%", "Exp.", "$z_{adj}$", "$q$",
                                                                   "Seeds", "Shift", "Bulk", "Tail $z$"], "\\scriptsize", fit=True))

    # 3. Blind validation.
    v = pd.read_csv(TABLES / "validation_centres.csv")
    incident = {260402: "Leak origin", 220502: "Foiled insider plan", 392203: "Wrong-medium papers", 230701: "Grace marks, re-test",
                170501: "Grace marks, re-test", 170701: "Grace marks (one of two)", 170702: "Grace marks (one of two)",
                330501: "Grace marks (one of three)", 330502: "Grace marks (one of three)", 330503: "Grace marks (one of three)",
                220701: "Singled out in press", 220705: "Singled out in press"}
    t = pd.DataFrame({"List": v["list"], "Centre": v["centre_name"].map(title), "City": v["city"].map(lambda s: esc(s).title()),
                      "Incident": v["centre_code"].map(incident), "Top 1\\%": v["top1"].map(n), "Exp.": v["E1"].map(lambda x: d(x, 1)),
                      "z": v["z1_adj"].map(d), "q": v["q_excess"].map(q), "Tail z": v["z_tail"].map(d),
                      "Rank": [("%d of %d" % (a, b)) for a, b in zip(v["city_rank_z1"], v["n_centres_city"])]})
    (out / "blind.tex").write_text(tabular(t, "lp{4.2cm}lp{2.6cm}rrrrrr", ["", "Centre", "City", "Incident", "Top 1\\%", "Exp.",
                                                                          "$z_{adj}$", "$q$", "Tail $z$", "In city"], "\\scriptsize", fit=True))

    # 4. Detection limits: centre fraud.
    ic = pd.read_csv(TABLES / "inject_centre.csv")
    bands = ["<250", "250-499", "500-749", ">=750"]
    rows = []
    for m in sorted(ic["m"].unique()):
        row = {"m": str(int(m))}
        for b in bands:
            r = ic[(ic["band"] == b) & (ic["m"] == m)].iloc[0]
            row[b] = "%s / %s" % (pc(r["detect_layer2"]), pc(r["detect_tail"]))
        rows.append(row)
    hdr = ["Boosted candidates", "under 250", "250-499", "500-749", "750 or more"]
    (out / "inject_centre.tex").write_text(tabular(pd.DataFrame(rows), "lrrrr", hdr))

    # 5. Detection limits: city leaks and court scenarios.
    il = pd.read_csv(TABLES / "inject_city.csv")
    names = {"random city": "A random city"}
    t = pd.DataFrame({"Scenario": il["scenario"].map(lambda s: names.get(s, esc(s))), "Beneficiaries": il["m"].map(n),
                      "Within-city": il["detect_layer2"].map(pc), "Tail beyond bulk": il["detect_tail"].map(pc),
                      "Between-city funnel": il["detect_funnel"].map(pc)})
    (out / "inject_city.tex").write_text(tabular(t, "lrrrr", ["Scenario", "Beneficiaries", "Within-city screen", "Tail beyond bulk", "City funnel"], fit=True))

    # 6. Hub cities.
    g = pd.read_csv(TABLES / "geo_cities.csv")
    h = pd.read_csv(TABLES / "geo_hubs.csv")[["city", "state", "p_hetero"]]
    g = g[g["hub"]].merge(h, on=["city", "state"], how="left").sort_values("z_funnel", ascending=False)
    t = pd.DataFrame({"City": g["city"].map(lambda s: esc(s).title()), "State": g["state"].map(lambda s: esc(s).title()),
                      "Candidates": g["N"].map(n), "Top 1\\%": g["top1"].map(n), "Rate": g["rate"].map(lambda x: pc(x, 1)),
                      "z": g["z_funnel"].map(d), "Centres": g["centres"].map(n),
                      "Spread": g["p_hetero"].map(lambda p: "" if pd.isna(p) else ("concentrated" if p < 0.05 else "even"))})
    (out / "hubs.tex").write_text(tabular(t, "llrrrrrl", ["City", "State", "Candidates", "Top 1\\%", "Rate", "Funnel $z$", "Centres", "Across centres"], "\\scriptsize", fit=True))

    # 7. Development correlations.
    dv = pd.read_csv(TABLES / "geo_development.csv")
    dv = dv[dv["outcome"].isin(["mean_marks", "ge_med_share", "top1_rate"])]
    lab = {"mean_marks": "Mean marks", "ge_med_share": "Share at or above the median", "top1_rate": "Share in the top 1\\%"}
    rows = []
    for o in ("mean_marks", "ge_med_share", "top1_rate"):
        row = {"Outcome": lab[o]}
        for vname in ("all units", "certain mappings only", "without Kota and Sikar", "units with 1,000+ candidates"):
            r = dv[(dv["outcome"] == o) & (dv["variant"] == vname)].iloc[0]
            row[vname] = "%s [%s, %s]" % (d(r["rho"]), d(r["lo"]), d(r["hi"]))
        rows.append(row)
    (out / "development.tex").write_text(tabular(pd.DataFrame(rows), "lllll", ["Outcome", "All units", "Certain mappings", "Without Kota, Sikar", "1,000+ candidates"], "\\scriptsize"))

    # 8. Fragility at key scores.
    fk = pd.read_csv(TABLES / "fragility_key.csv")
    t = pd.DataFrame({"Marks": fk["score"].map(n), "Best rank": fk["best_rank"].map(n), "Tied": fk["tie"].map(n),
                      "1 mark": fk["ranks_1"].map(n), "4 marks": fk["ranks_4"].map(n), "5 marks": fk["ranks_5"].map(n),
                      "SEM": fk["sem"].map(lambda x: "%.0f" % x),
                      "Band": ["%s to %s" % (n(a), n(b)) for a, b in zip(fk["band_best"], fk["band_worst"])]})
    (out / "fragility.tex").write_text(tabular(t, "rrrrrrrr", ["Marks", "Best rank", "Tied on it", "1 mark", "4 marks", "5 marks", "SEM", "Ranks within one SEM"]))

    # 9. Allotment by state.
    sa = pd.read_csv(TABLES / "state_allotment.csv")
    t = pd.DataFrame({"State": sa["state"].map(lambda s: esc(s).title()), "Centres": sa["centres"].map(n), "ICC": sa["icc_marks"].map(lambda x: d(x, 4)),
                      "SD": sa["sd_zmean"].map(d), "Beyond 3": sa["share_abs_zmean_gt3"].map(pc),
                      "Regional": [pc(s, 1) if o else "Hindi/English" for s, o in zip(sa["regional_share"], sa["offers_regional"])]})
    (out / "states.tex").write_text(tabular(t, "lrrrrr", ["State", "Centres", "ICC of marks", "SD of centre $z$", "$|z|>3$", "Regional-medium share"], "\\scriptsize"))
    print("tables written:", sorted(p.name for p in out.glob("*.tex")))


if __name__ == "__main__":
    main()
