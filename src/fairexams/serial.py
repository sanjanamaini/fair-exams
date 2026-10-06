"""Stage 3b: does serial order inside a centre carry information? (PREREGISTRATION.md, two pathways)

    python -m fairexams.serial

Per centre: lag-1 autocorrelation of marks along NTA's serial numbers, standardised against a
random order, pooled across centres; the same at lags 1-30. A random subsample of 100 centres is
also checked against 10,000 genuine permutations each, to confirm the standardisation.
If pooled autocorrelation is indistinguishable from zero (pathway A), no room-level scan is run.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import INTERIM, N_PERM, SEED, TABLES

LAGS = range(1, 31)


def autocorr(x: np.ndarray, lag: int = 1) -> float:
    x = x - x.mean()
    d = float((x * x).sum())
    return float((x[lag:] * x[:-lag]).sum() / d) if d > 0 else 0.0


def main() -> None:
    cand = pd.read_parquet(INTERIM / "candidates.parquet").sort_values(["code", "serial"])
    rng = np.random.default_rng(SEED)
    rows, lag_sums = [], {k: [] for k in LAGS}
    groups = {c: g["marks"].to_numpy().astype(float) for c, g in cand.groupby("code")}
    for code, x in groups.items():
        n = len(x)
        if n < 30:
            continue
        r1 = autocorr(x)
        rows.append({"code": code, "n": n, "r1": r1, "z": (r1 + 1 / n) * np.sqrt(n)})
        if n >= 100:
            for k in LAGS:
                lag_sums[k].append(autocorr(x, k))
    df = pd.DataFrame(rows)
    pooled = float(df["z"].sum() / np.sqrt(len(df)))

    # Permutation check on 100 random centres: exact permutation p-values against the z approximation.
    sample = rng.choice(df["code"].to_numpy(), size=min(100, len(df)), replace=False)
    perm_rows = []
    for code in sample:
        x = groups[code] - groups[code].mean()
        obs = autocorr(groups[code])
        sims = np.empty(N_PERM)
        for b in range(N_PERM):
            y = rng.permutation(x)
            sims[b] = (y[1:] * y[:-1]).sum() / (y * y).sum()
        perm_rows.append({"code": code, "r1": obs, "perm_mean": sims.mean(), "perm_sd": sims.std(),
                          "perm_p_two_sided": float((np.abs(sims - sims.mean()) >= abs(obs - sims.mean())).mean())})
    perm = pd.DataFrame(perm_rows)
    lag_profile = pd.DataFrame({"lag": list(LAGS), "mean_autocorr": [float(np.mean(lag_sums[k])) for k in LAGS],
                                "centres": [len(lag_sums[k]) for k in LAGS]})

    numbers = {"serial_centres": int(len(df)), "serial_mean_r1": float(df["r1"].mean()),
               "serial_mean_z": float(df["z"].mean()), "serial_sd_z": float(df["z"].std()),
               "serial_pooled_z": pooled, "serial_max_abs_lag_mean": float(lag_profile["mean_autocorr"].abs().max()),
               "serial_perm_centres": int(len(perm)), "serial_perm_share_p05": float((perm["perm_p_two_sided"] < 0.05).mean()),
               "serial_perm_sd_ratio": float((perm["perm_sd"] * np.sqrt(df.set_index("code").loc[perm["code"], "n"].to_numpy())).mean()),
               "serial_pathway": "A" if abs(pooled) < 3 else "B"}
    df.to_csv(TABLES / "serial_centres.csv", index=False, float_format="%.6g")
    lag_profile.to_csv(TABLES / "serial_lags.csv", index=False, float_format="%.6g")
    perm.to_csv(TABLES / "serial_permutation_check.csv", index=False, float_format="%.6g")
    (TABLES / "serial_numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
