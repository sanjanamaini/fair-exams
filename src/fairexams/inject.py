"""Stage 3c: detection limits by injection-recovery (PREREGISTRATION.md, Strand 1b).

    python -m fairexams.inject

Synthetic frauds are planted in the real data, one at a time, and the screen is re-run for the
affected city with rho, the empirical null and the Benjamini-Hochberg bars held at their real values.
- Centre fraud: m candidates at one centre get marks drawn from the national top 1%.
- City leak: m candidates spread over one city's centres get the same.
Detection: registered Layer 2 (any centre in the city below the real BH bar), Addition A1 (same,
for the tail-beyond-bulk test), and for city leaks the between-city funnel (city z above 3.09 when
it was not already).
Also two case studies from the court record: about 125 beneficiaries at Hazaribagh centres and about
30 at Patna centres, with marks drawn from the national top 1% or the national top 10%.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from .bulk import bulk_prediction
from .config import (INJECT_M_CENTRE, INJECT_M_CITY, INTERIM, N_INJECT, SEED, SIZE_BANDS, TABLES, TOP_SHARE)
from .screen import BAND_LABELS
from .stats import hypergeom_tails, random_effects_tau2, randomised_z

FUNNEL_Z = stats.norm.isf(0.001)  # upper 99.8% control limit


class World:
    """Real-data state needed to re-run the screen for one city."""

    def __init__(self):
        self.cand = pd.read_parquet(INTERIM / "candidates.parquet")
        cent = pd.read_parquet(INTERIM / "centres.parquet")
        self.city = pd.read_parquet(INTERIM / "cities.parquet")
        nums = json.loads((TABLES / "panel_numbers.json").read_text())
        scr = json.loads((TABLES / "screen_numbers.json").read_text())
        bul = json.loads((TABLES / "bulk_numbers.json").read_text())
        self.t1 = nums["t_top1"]
        self.rho1 = scr["rho1"]
        self.cut2 = scr["bh_cutoff_layer2"]
        self.d0, self.s0, self.cut_tail = bul["tail_delta0"], bul["tail_sigma0"], bul["tail_bh_cutoff"]
        m = self.cand["marks"].to_numpy()
        self.top_pool = m[m >= self.t1]
        self.top10_pool = m[m >= np.quantile(m, 0.9)]
        self.multi = cent[cent["n_centres_city"] >= 2][["code", "city_id", "n"]].copy()
        self.multi["band"] = pd.cut(self.multi["n"], SIZE_BANDS, right=False, labels=BAND_LABELS)
        self.marks_by_code = {c: g["marks"].to_numpy() for c, g in self.cand.groupby("code")}
        self.codes_by_city = self.multi.groupby("city_id")["code"].apply(list).to_dict()
        # Between-city funnel from the real data (Indian cities, top-1% rate).
        ci = self.city[~self.city["overseas"]]
        self.pi = float(self.city["top1"].sum() / self.city["N"].sum())
        y, s2 = ci["top1"] / ci["N"], self.pi * (1 - self.pi) / ci["N"]
        self.tau2, _, _ = random_effects_tau2(y.to_numpy(), s2.to_numpy())
        self.city_z = ((y - self.pi) / np.sqrt(s2 + self.tau2)).set_axis(ci["city_id"]).to_dict()

    def detect_city(self, city_id, new_marks: dict, rng) -> tuple[bool, bool]:
        """Re-run Layer 2 and A1 for every centre of a city given modified marks for some centres."""
        codes = self.codes_by_city[city_id]
        marks = {c: new_marks.get(c, self.marks_by_code[c]) for c in codes}
        N = sum(len(v) for v in marks.values())
        K = sum(int((v >= self.t1).sum()) for v in marks.values())
        hit2 = hit_tail = False
        for c in codes:
            own = marks[c]
            n, k = len(own), int((own >= self.t1).sum())
            p_ge, p_le, p_gt, p_eq = hypergeom_tails(k, N, K, n)
            z = float(randomised_z(p_gt, p_eq, rng.uniform(), p_lt=p_le - p_eq))
            z2 = z / np.sqrt(1 + (n - 1) * self.rho1)
            hit2 |= stats.norm.sf(z2) <= self.cut2
            if c in new_marks:
                rest = np.sort(np.concatenate([marks[o] for o in codes if o != c]))
                share, _, _ = bulk_prediction(own, rest, self.t1)
                pt = (stats.binom.sf(k, n, share), stats.binom.pmf(k, n, share),
                      stats.binom.cdf(k - 1, n, share) if k > 0 else 0.0)
                zt = float(randomised_z(pt[0], pt[1], rng.uniform(), p_lt=pt[2]))
                hit_tail |= stats.norm.sf((zt - self.d0) / self.s0) <= self.cut_tail
        return bool(hit2), bool(hit_tail)

    def detect_funnel(self, city_id, added_top: int) -> bool:
        row = self.city[self.city["city_id"] == city_id].iloc[0]
        if self.city_z.get(city_id, 0) > FUNNEL_Z:
            return False  # already above the limit: an injection cannot be told apart
        y = (row["top1"] + added_top) / row["N"]
        z = (y - self.pi) / np.sqrt(self.pi * (1 - self.pi) / row["N"] + self.tau2)
        return bool(z > FUNNEL_Z)


def plant(marks: np.ndarray, idx: np.ndarray, pool: np.ndarray, rng) -> np.ndarray:
    out = marks.copy()
    out[idx] = rng.choice(pool, size=len(idx))
    return out


def centre_fraud(w: World, rng) -> pd.DataFrame:
    rows = []
    for band in BAND_LABELS:
        sub = w.multi[w.multi["band"] == band]
        for m in INJECT_M_CENTRE:
            elig = sub[sub["n"] > 2 * m]
            hits2 = hitst = 0
            for _ in range(N_INJECT):
                r = elig.iloc[rng.integers(len(elig))]
                own = w.marks_by_code[r["code"]]
                new = plant(own, rng.choice(len(own), size=m, replace=False), w.top_pool, rng)
                h2, ht = w.detect_city(r["city_id"], {r["code"]: new}, rng)
                hits2 += h2
                hitst += ht
            rows.append({"band": band, "m": m, "injections": N_INJECT, "detect_layer2": hits2 / N_INJECT,
                         "detect_tail": hitst / N_INJECT, "mean_n": float(elig["n"].mean())})
    return pd.DataFrame(rows)


def city_leak(w: World, rng, city_ids=None, ms=INJECT_M_CITY, pool=None, label="random city") -> pd.DataFrame:
    pool = w.top_pool if pool is None else pool
    rows = []
    for m in ms:
        if city_ids is None:
            sizes = w.city.set_index("city_id")["N"]
            elig = [c for c in w.codes_by_city if sizes[c] >= 5 * m]
        else:
            elig = city_ids
        h2 = ht = hf = 0
        for _ in range(N_INJECT):
            cid = elig[rng.integers(len(elig))]
            codes = w.codes_by_city[cid]
            sizes = np.array([len(w.marks_by_code[c]) for c in codes])
            flat = rng.choice(sizes.sum(), size=m, replace=False)
            owner = np.searchsorted(np.cumsum(sizes), flat, side="right")
            new, added = {}, 0
            for j in np.unique(owner):
                c = codes[j]
                local = flat[owner == j] - (np.cumsum(sizes)[j] - sizes[j])
                before = int((w.marks_by_code[c][local] >= w.t1).sum())
                new[c] = plant(w.marks_by_code[c], local, pool, rng)
                added += int((new[c][local] >= w.t1).sum()) - before
            a, b = w.detect_city(cid, new, rng)
            h2 += a
            ht += b
            hf += w.detect_funnel(cid, added)
        rows.append({"scenario": label, "m": m, "injections": N_INJECT, "detect_layer2": h2 / N_INJECT,
                     "detect_tail": ht / N_INJECT, "detect_funnel": hf / N_INJECT})
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED + 2)
    w = World()
    cf = centre_fraud(w, rng)
    cl = city_leak(w, rng)
    by_name = w.city[~w.city["overseas"]].set_index("city")["city_id"]
    cases = []
    for name, m in (("HAZARIBAGH", 125), ("PATNA", 30)):
        for pool, lab in ((w.top_pool, "top 1%"), (w.top10_pool, "top 10%")):
            d = city_leak(w, rng, city_ids=[int(by_name[name])], ms=(m,), pool=pool, label="%s, %s" % (name.title(), lab))
            cases.append(d)
    cases = pd.concat(cases, ignore_index=True)
    cf.to_csv(TABLES / "inject_centre.csv", index=False, float_format="%.4g")
    pd.concat([cl, cases], ignore_index=True).to_csv(TABLES / "inject_city.csv", index=False, float_format="%.4g")

    numbers = {"inject_tau2": w.tau2, "inject_pi": w.pi, "inject_top10_floor": int(w.top10_pool.min())}
    for band in BAND_LABELS:
        sub = cf[cf["band"] == band]
        for col, key in (("detect_layer2", "l2"), ("detect_tail", "tail")):
            ok = sub[sub[col] >= 0.8]
            numbers["inject_min_m80_%s_%s" % (key, band)] = int(ok["m"].min()) if len(ok) else -1
    for r in cf.itertuples():
        numbers["inject_c_%s_%d_l2" % (r.band, r.m)] = r.detect_layer2
        numbers["inject_c_%s_%d_tail" % (r.band, r.m)] = r.detect_tail
    for r in pd.concat([cl, cases]).itertuples():
        key = r.scenario.split(",")[0].replace(" ", "").lower() + ("_top10" if "10%" in r.scenario else "")
        for col in ("detect_layer2", "detect_tail", "detect_funnel"):
            numbers["inject_%s_%d_%s" % (key, r.m, col.split("_")[1])] = getattr(r, col)
    (TABLES / "inject_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(cf.to_string())
    print(pd.concat([cl, cases]).to_string())
    print("tau2", w.tau2, "pi", w.pi, "top10 floor", int(w.top10_pool.min()), "share", TOP_SHARE)


if __name__ == "__main__":
    main()
