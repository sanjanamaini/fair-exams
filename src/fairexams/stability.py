"""Addition A4 (DEVIATIONS.md): how stable are the flags across random seeds?

    python -m fairexams.stability

Discrete tests use randomised p-values (exactly uniform under the null), so a borderline centre can be
flagged under one seed and not another. The integrity monitor is run on the NEET data with 50 seeds;
every centre flagged under any seed is listed with the share of seeds that flag it.
"""
from __future__ import annotations

import json

import pandas as pd

from .config import INTERIM, SEED, TABLES
from .monitor import run_screen

N_SEEDS = 50


def main() -> None:
    cand = pd.read_parquet(INTERIM / "candidates.parquet").rename(columns={"code": "centre", "city_id": "city"})
    c, s = run_screen(cand, seeds=tuple(SEED + i for i in range(N_SEEDS)))
    reg = pd.read_csv(TABLES / "screen_centres.csv")[["code", "city", "name", "flag"]].rename(
        columns={"code": "centre", "city": "city_name", "flag": "registered_flag"})
    tail = pd.read_csv(TABLES / "bulk_tail.csv")[["code", "flag_tail"]].rename(columns={"code": "centre", "flag_tail": "registered_tail_flag"})
    out = c[(c["flag_rate"] > 0) | (c["flag_tail_rate"] > 0)].merge(reg, on="centre").merge(tail, on="centre")
    out = out[["centre", "city_name", "name", "n", "k", "E", "flag_rate", "registered_flag", "flag_tail_rate",
               "registered_tail_flag"]].sort_values(["flag_rate", "flag_tail_rate"], ascending=False)
    out.to_csv(TABLES / "flag_stability.csv", index=False, float_format="%.4g")
    r = out[out["registered_flag"]]
    numbers = {"stab_seeds": N_SEEDS, "stab_any_flag": int((out["flag_rate"] > 0).sum()),
               "stab_flag_majority": int((out["flag_rate"] >= 0.5).sum()),
               "stab_flag_95": int((out["flag_rate"] >= 0.95).sum()),
               "stab_registered_min_rate": float(r["flag_rate"].min()),
               "stab_registered_all_seeds": int((r["flag_rate"] == 1).sum()),
               "stab_tail_any": int((out["flag_tail_rate"] > 0).sum()),
               "stab_tail_majority": int((out["flag_tail_rate"] >= 0.5).sum()),
               "stab_tail_95": int((out["flag_tail_rate"] >= 0.95).sum()),
               "stab_rk_unit1_rate": float(out.loc[out["centre"] == 220701, "flag_rate"].iloc[0]),
               "stab_rk_unit1_tail_rate": float(out.loc[out["centre"] == 220701, "flag_tail_rate"].iloc[0])}
    (TABLES / "stability_numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))
    print(out.to_string())


if __name__ == "__main__":
    main()
