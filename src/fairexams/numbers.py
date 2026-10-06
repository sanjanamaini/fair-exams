"""Collect every number the report uses into results/numbers.json and report/numbers.tex.

    python -m fairexams.numbers

Each stage writes results/tables/<stage>_numbers.json. This merges them and writes one LaTeX macro
per number (e.g. court_hazaribagh_top1 -> \\CourtHazaribaghTopOne), formatted by kind, so no number
in the report is typed by hand.
"""
from __future__ import annotations

import json
import math
import re

from .config import REPORT, RESULTS, TABLES

DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


def macro_name(key: str) -> str:
    out = ""
    for p in re.split(r"[_\W]+", key):
        if p:
            p = re.sub(r"\d", lambda m: DIGITS[int(m.group())], p)
            out += p[0].upper() + p[1:]
    return out


def tex_escape(s: str) -> str:
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", " ")


RULES = [  # first match wins
    ("p", r"(_p$|_pvalue$|p_hetero$|binom_p$|p_extreme|_q$)"),
    ("dec1", r"(_E$|_E1$)"),
    ("int", r"(shift_median|own_median$|rest_median$|_q10$|_sem$)"),
    ("sci", r"(tau2|cutoff)"),
    ("dec4", r"(^rho1$|^rho1_trim|^rho_marks$|_icc$|^medium_icc_max$|slope|^serial_mean_r1$|max_abs_lag)"),
    ("dec2", r"(z_funnel|_z$|_z_|_sd$|_sd_|_sdz$|sigma0|delta0|_rho$|_rho_|_lo$|_hi$|corr|deff|loading|odds_ratio|"
             r"_mu$|_sigma$|_sem$|^mean$|^sd$|_ztail$|_E1$|pi0|spearman$|sd_ratio)"),
    ("pct", r"(share|_rate$|_rate_|^inject_c_|^inject_(randomcity|hazaribagh|patna)|^geo_pi$|^inject_pi$|^funnel_pi$|"
            r"^comb_|percentile|_regshare$)"),
]


def kind(k: str) -> str:
    for name, pat in RULES:
        if re.search(pat, k):
            return name
    return "auto"


def fmt(k: str, v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        return tex_escape(v)
    kd = kind(k)
    if kd == "p":
        return "$<$0.001" if v < 0.001 else "%.3f" % v
    if kd == "sci":
        m, e = ("%.2e" % v).split("e")
        return "$%s\\times10^{%d}$" % (m, int(e))
    if kd == "dec4":
        return "%.4f" % v
    if kd == "dec1":
        return "%.1f" % v
    if kd == "int":
        return "{:,}".format(int(round(v)))
    if kd == "pct":
        x = 100 * v
        d = 0 if (abs(x) >= 10 or float(x).is_integer()) else (1 if abs(x) >= 1 else 2)
        return "%.*f\\%%" % (d, x)
    if kd == "dec2":
        return "%.2f" % v
    if float(v).is_integer() or abs(v) >= 100:
        return "{:,}".format(int(round(v)))
    return "%.2f" % v


def main() -> None:
    merged = {}
    for p in sorted(TABLES.glob("*_numbers.json")):
        merged.update(json.loads(p.read_text()))
    (RESULTS / "numbers.json").write_text(json.dumps(merged, indent=1, sort_keys=True, default=float))
    lines = ["% Written by fairexams.numbers from results/numbers.json. Do not edit by hand."]
    seen = {}
    for k in sorted(merged):
        name = macro_name(k)
        if name in seen:
            raise ValueError("macro clash: %s and %s" % (k, seen[name]))
        seen[name] = k
        v = merged[k]
        if isinstance(v, list):
            continue
        lines.append("\\newcommand{\\%s}{%s}" % (name, fmt(k, v)))
    REPORT.mkdir(exist_ok=True)
    (REPORT / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("numbers: %d macros" % (len(lines) - 1))


if __name__ == "__main__":
    main()
