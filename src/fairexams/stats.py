"""Statistical building blocks used by the screen, the injections and the geography strand."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def hypergeom_tails(k, N, K, n):
    """Exact tails for a centre's count k of top scorers under random allotment within its city.

    N candidates in the city, K of them top scorers, n at the centre. Returns P(X >= k), P(X <= k),
    P(X > k) and P(X = k), all as arrays.
    """
    k, N, K, n = (np.asarray(a) for a in (k, N, K, n))
    p_ge = stats.hypergeom.sf(k - 1, N, K, n)
    p_le = stats.hypergeom.cdf(k, N, K, n)
    p_gt = stats.hypergeom.sf(k, N, K, n)
    p_eq = stats.hypergeom.pmf(k, N, K, n)
    return p_ge, p_le, p_gt, p_eq


def randomised_z(p_gt, p_eq, u, p_lt=None):
    """Randomised probability integral transform: exactly N(0, 1) under the null; positive = excess.

    Each tail is computed directly (not as 1 minus the other) so deep deficits stay finite.
    """
    p_hi = np.clip(p_gt + u * p_eq, 1e-300, 1.0)
    if p_lt is None:
        return stats.norm.isf(p_hi)
    p_lo = np.clip(p_lt + (1 - u) * p_eq, 1e-300, 1.0)
    return np.where(p_hi < 0.5, stats.norm.isf(p_hi), stats.norm.ppf(p_lo))


def mean_z(centre_mean, n, city_mean, city_var, N):
    """Finite-population z for a centre mean drawn without replacement from its city."""
    se = np.sqrt(city_var / n * (N - n) / np.maximum(N - 1, 1))
    return (centre_mean - city_mean) / se


def nested_icc(y_centre_sum, y_centre_ss, n, city):
    """ANOVA estimator of the intra-class correlation of centres nested in cities.

    Inputs per centre: sum of y, sum of y squared, size, and city id. Returns (rho, details).
    """
    df = pd.DataFrame({"s": y_centre_sum, "ss": y_centre_ss, "n": n, "city": city})
    ybar_j = df["s"] / df["n"]
    g = df.groupby("city")
    city_n = g["n"].transform("sum")
    city_mean = g["s"].transform("sum") / city_n
    ss_between = float((df["n"] * (ybar_j - city_mean) ** 2).sum())
    ss_within = float((df["ss"] - df["s"] ** 2 / df["n"]).sum())
    J, C, N = len(df), df["city"].nunique(), float(df["n"].sum())
    df_b, df_w = J - C, N - J
    msb, msw = ss_between / df_b, ss_within / df_w
    n0 = (N - float((df["n"] ** 2 / city_n).sum())) / df_b
    sigma_a2 = max((msb - msw) / n0, 0.0)
    rho = sigma_a2 / (sigma_a2 + msw) if (sigma_a2 + msw) > 0 else 0.0
    return rho, {"msb": msb, "msw": msw, "n0": n0, "df_between": df_b, "df_within": df_w}


def bh(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def bh_cutoff(p, q=0.05):
    """Largest p-value declared significant by BH at level q (or the first-discovery bar if none)."""
    p = np.sort(np.asarray(p, dtype=float))
    m = len(p)
    ok = np.nonzero(p <= q * np.arange(1, m + 1) / m)[0]
    return float(p[ok[-1]]) if len(ok) else q / m


def storey_pi0(p, lam=0.5):
    p = np.asarray(p, dtype=float)
    return float(min(1.0, (p > lam).mean() / (1 - lam)))


def central_matching(z, half_width=1.5, bins=40):
    """Efron's central matching: fit a quadratic to the log histogram of z near its centre.

    Returns (delta0, sigma0). The fit window is median +/- half_width times a resistant spread.
    """
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    med = np.median(z)
    spread = (np.quantile(z, 0.75) - np.quantile(z, 0.25)) / 1.349
    lo, hi = med - half_width * spread, med + half_width * spread
    counts, edges = np.histogram(z[(z >= lo) & (z <= hi)], bins=bins, range=(lo, hi))
    mids = (edges[:-1] + edges[1:]) / 2
    keep = counts > 0
    b2, b1, _ = np.polyfit(mids[keep], np.log(counts[keep]), 2, w=np.sqrt(counts[keep]))
    if b2 >= 0:
        return float(med), float(spread)
    sigma0 = np.sqrt(-1 / (2 * b2))
    return float(b1 * sigma0 ** 2), float(sigma0)


def random_effects_tau2(y, s2, trim=0.1):
    """DerSimonian-Laird between-unit variance, on winsorised standardised residuals (Spiegelhalter 2005)."""
    y, s2 = np.asarray(y, float), np.asarray(s2, float)
    w = 1 / s2
    target = np.sum(w * y) / np.sum(w)
    z = (y - target) / np.sqrt(s2)
    lo, hi = np.quantile(z, [trim, 1 - trim])
    zw = np.clip(z, lo, hi)
    Q = float(np.sum(zw ** 2))
    k = len(y)
    tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    return tau2, target, Q
