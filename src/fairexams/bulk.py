"""Addition A1 (DEVIATIONS.md): more top scorers than the centre's own bulk predicts?

    python -m fairexams.bulk

Sorting moves a centre's whole distribution; a boost to some candidates moves only its top.
Each centre's marks become normal scores against the rest of its city. A normal fit to the
centre's middle half (median, interquartile range) predicts its share above the national
top-1% threshold. The observed count is tested against that prediction (randomised binomial z),
calibrated with Efron's empirical null across centres, then Benjamini-Hochberg at 0.05.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from .config import FDR_Q, INTERIM, SEED, TABLES
from .stats import bh, central_matching, randomised_z


def normal_scores(x: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    """Mid-rank normal scores of x against a sorted reference sample."""
    lo = np.searchsorted(ref_sorted, x, "left")
    hi = np.searchsorted(ref_sorted, x, "right")
    n = len(ref_sorted)
    f = np.clip((lo + 0.5 * (hi - lo)) / n, 0.5 / n, 1 - 0.5 / n)
    return stats.norm.ppf(f)


def bulk_prediction(own: np.ndarray, rest_sorted: np.ndarray, t1: int) -> tuple[float, float, float]:
    """Predicted share at or above t1 from the centre's middle half; returns (share, mu, sigma)."""
    u = normal_scores(own, rest_sorted)
    q25, mu, q75 = np.quantile(u, [0.25, 0.5, 0.75])
    sigma = max((q75 - q25) / 1.349, 1e-3)
    n = len(rest_sorted)
    below = np.searchsorted(rest_sorted, t1, "left") / n
    t_u = stats.norm.ppf(np.clip(below, 0.5 / n, 1 - 0.5 / n))
    return float(stats.norm.sf((t_u - mu) / sigma)), float(mu), float(sigma)


def tail_test(df: pd.DataFrame, cand: pd.DataFrame, t1: int, rng) -> pd.DataFrame:
    by_city = {c: g for c, g in cand.groupby("city_id")}
    rows = []
    for r in df.itertuples():
        g = by_city[r.city_id]
        mask = g["code"].to_numpy() == r.code
        own = g["marks"].to_numpy()[mask]
        rest = np.sort(g["marks"].to_numpy()[~mask])
        share, mu, sigma = bulk_prediction(own, rest, t1)
        k, n = int((own >= t1).sum()), len(own)
        rows.append({"code": r.code, "bulk_mu": mu, "bulk_sigma": sigma, "tail_share_pred": share,
                     "tail_E": n * share, "tail_k": k,
                     "p_gt": stats.binom.sf(k, n, share), "p_eq": stats.binom.pmf(k, n, share),
                     "p_lt": stats.binom.cdf(k - 1, n, share) if k > 0 else 0.0})
    out = pd.DataFrame(rows)
    out["z_tail"] = randomised_z(out["p_gt"].to_numpy(), out["p_eq"].to_numpy(), rng.uniform(size=len(out)),
                                 p_lt=out["p_lt"].to_numpy())
    return out.drop(columns=["p_gt", "p_eq", "p_lt"])


def main() -> None:
    cand = pd.read_parquet(INTERIM / "candidates.parquet")
    screen = pd.read_csv(TABLES / "screen_centres.csv")
    t1 = json.loads((TABLES / "panel_numbers.json").read_text())["t_top1"]
    cent = pd.read_parquet(INTERIM / "centres.parquet")[["code", "city_id"]]
    df = screen.merge(cent, on="code")
    rng = np.random.default_rng(SEED + 1)
    tail = tail_test(df, cand, t1, rng)
    d0, s0 = central_matching(tail["z_tail"])
    tail["p_tail"] = stats.norm.sf((tail["z_tail"] - d0) / s0)
    tail["q_tail"] = bh(tail["p_tail"])
    tail["flag_tail"] = tail["q_tail"] <= FDR_Q
    out = df[["code", "state", "city", "name", "n", "top1", "E1", "z1_adj", "q_excess", "flag"]].merge(tail, on="code")
    out = out.sort_values("z_tail", ascending=False)
    out.to_csv(TABLES / "bulk_tail.csv", index=False, float_format="%.6g")
    informative = out[out["E1"] >= 5]
    numbers = {"tail_delta0": d0, "tail_sigma0": s0, "tail_flags": int(out["flag_tail"].sum()),
               "tail_flags_of_registered": int((out["flag_tail"] & out["flag"]).sum()),
               "tail_registered_flags": int(out["flag"].sum()),
               "tail_sd_z": float(out["z_tail"].std()), "tail_corr_with_bulk_mu": float(np.corrcoef(out["z_tail"], out["bulk_mu"])[0, 1]),
               "tail_informative_centres": int(len(informative)),
               "tail_informative_sd_z": float(informative["z_tail"].std()),
               "tail_bh_cutoff": float(out.loc[out["flag_tail"], "p_tail"].max()) if out["flag_tail"].any() else FDR_Q / len(out)}
    for code, name in ((220701, "rk"), (152101, "jehanabad"), (171202, "kondagaon")):
        r = out[out["code"] == code].iloc[0]
        numbers.update({"tail_%s_E" % name: float(r["tail_E"]), "tail_%s_k" % name: int(r["tail_k"]),
                        "tail_%s_z" % name: float(r["z_tail"]), "tail_%s_E1" % name: float(r["E1"])})
    (TABLES / "bulk_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(json.dumps(numbers, indent=1, default=float))
    cols = ["code", "city", "name", "n", "top1", "E1", "z1_adj", "flag", "bulk_mu", "bulk_sigma", "tail_E", "z_tail", "q_tail"]
    print(out[cols].head(15).to_string())
    print(out[out["flag"]][cols].to_string())


if __name__ == "__main__":
    main()
