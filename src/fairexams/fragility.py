"""Stage 5: rank fragility (PREREGISTRATION.md, Strand 3).

    python -m fairexams.fragility

From the national distribution of the 20 July 2024 file:
- per score: candidates tied on it, and ranks crossed by 1, 4 and 5 marks (a question left
  blank instead of wrong, blank instead of right, wrong instead of right);
- at the court's cut points (577 and 622 marks, ranks 1,08,000 and 56,000) and other scores;
- the 26 July revision: about 4 lakh candidates (press figure) lost 5 marks on one physics
  question; bounds on rank movement if they were spread evenly over the distribution;
- an illustrative measurement band from Lord's binomial error model, assuming all 180 questions
  were attempted (an upper estimate: candidates who leave questions blank have narrower bands).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import COURT_CUTS, FIGURES, MARK_RIGHT, MARK_WRONG, QUESTIONS, TABLES

KEY_SCORES = (450, 550, COURT_CUTS["rank_108000"], 600, COURT_CUTS["rank_56000"], 650, 680, 700)
REVISED_CANDIDATES = 400000  # "over four lakh" lost 5 marks (Business Today, 24 July 2024)


def lord_sem(score: np.ndarray, attempted: int = QUESTIONS) -> np.ndarray:
    """Lord's conditional SEM for number right among `attempted` items, in marks (one swap = 5 marks)."""
    step = MARK_RIGHT - MARK_WRONG
    c = (score - MARK_WRONG * attempted) / step  # s = 4c - (A - c)  =>  c = (s + A) / 5
    c = np.clip(c, 0, attempted)
    return step * np.sqrt(c * (attempted - c) / (attempted - 1))


def main() -> None:
    dist = pd.read_csv(TABLES / "score_distribution.csv").sort_values("score")
    counts = pd.Series(0, index=np.arange(-180, 721))
    counts[dist["score"].to_numpy()] = dist["candidates"].to_numpy()
    above = counts[::-1].cumsum()[::-1] - counts  # candidates strictly above each score
    total = int(counts.sum())

    def crossed(s, d):
        """Candidates with marks in (s, s + d]: places gained by d more marks."""
        return int(counts.loc[s + 1:min(s + d, 720)].sum())

    rows = []
    for s in range(-180, 721):
        if counts[s] == 0:
            continue
        rows.append({"score": s, "candidates": int(counts[s]), "best_rank": int(above[s]) + 1,
                     "worst_rank": int(above[s] + counts[s]), "ranks_1": crossed(s, 1), "ranks_4": crossed(s, 4),
                     "ranks_5": crossed(s, 5), "lord_sem_180": float(lord_sem(np.array([s]))[0])})
    fr = pd.DataFrame(rows)

    def rank_at(s):
        s = int(np.clip(round(s), -180, 720))
        return int(above[s]) + 1

    fr["band_best_rank"] = [rank_at(s + e) for s, e in zip(fr["score"], fr["lord_sem_180"])]
    fr["band_worst_rank"] = [rank_at(s - e) for s, e in zip(fr["score"], fr["lord_sem_180"])]
    fr.to_csv(TABLES / "fragility_scores.csv", index=False, float_format="%.4g")

    f = REVISED_CANDIDATES / total
    numbers = {"fr_total": total, "fr_revised_share": f}
    key = []
    for s in KEY_SCORES:
        r = fr[fr["score"] == s].iloc[0] if (fr["score"] == s).any() else None
        if r is None:
            continue
        k = {"score": s, "best_rank": int(r["best_rank"]), "tie": int(r["candidates"]), "ranks_5": int(r["ranks_5"]),
             "ranks_4": int(r["ranks_4"]), "ranks_1": int(r["ranks_1"]), "sem": float(r["lord_sem_180"]),
             "band_best": int(r["band_best_rank"]), "band_worst": int(r["band_worst_rank"]),
             "revision_gain_if_spared": f * crossed(s, 5),
             "revision_loss_if_hit": (1 - f) * int(counts.loc[s - 4:s].sum())}
        key.append(k)
        for name, val in k.items():
            if name != "score":
                numbers["fr_%d_%s" % (s, name)] = val
    key = pd.DataFrame(key)
    key.to_csv(TABLES / "fragility_key.csv", index=False, float_format="%.4g")

    top = fr[fr["best_rank"] <= 108000]
    numbers.update({"fr_max_tie": int(fr["candidates"].max()), "fr_max_tie_score": int(fr.loc[fr["candidates"].idxmax(), "score"]),
                    "fr_max_tie_top108k": int(top["candidates"].max()),
                    "fr_max_tie_top108k_score": int(top.loc[top["candidates"].idxmax(), "score"]),
                    "fr_720_tie": int(counts[720]), "fr_715_tie": int(counts[715]),
                    "fr_within5_of_577": int(counts.loc[573:582].sum()),
                    "fr_within5_of_622": int(counts.loc[618:627].sum()),
                    "fr_median_ranks5_top108k": float(top["ranks_5"].median())})
    numbers.update({"fr_ge700": int(counts.loc[700:].sum()), "fr_eq720": int(counts[720])})

    # Addition A3 (DEVIATIONS.md): the comb. A candidate who attempts all 180 questions scores 5c - 180,
    # a multiple of 5, so their mass shows up as excess at multiples of 5 over the four neighbouring totals.
    comb = []
    for lo in range(-20, 630, 50):
        blocks = [(counts[s], np.mean([counts[s + k] for k in (-2, -1, 1, 2)]))
                  for s in range(lo - lo % 5, lo + 50, 5) if -180 <= s - 2 and s + 2 <= 720]
        m = np.array(blocks)
        comb.append({"band_lo": lo, "band_hi": lo + 49, "candidates": int(m[:, 0].sum() + 4 * m[:, 1].sum()),
                     "full_attempt_share": float((m[:, 0] - m[:, 1]).sum() / (m[:, 0] + 4 * m[:, 1]).sum())})
    comb = pd.DataFrame(comb)
    comb.to_csv(TABLES / "fragility_comb.csv", index=False, float_format="%.4g")
    numbers.update({"comb_low": float(comb.loc[comb["band_lo"] <= 30, "full_attempt_share"].mean()),
                    "comb_mid": float(comb.loc[(comb["band_lo"] >= 230) & (comb["band_lo"] <= 480), "full_attempt_share"].mean())})
    (TABLES / "fragility_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    plot(fr)
    print(key.to_string())
    print(json.dumps({k: v for k, v in numbers.items() if not k[3:4].isdigit()}, indent=1, default=float))


def plot(fr):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    a1.bar(fr["score"], fr["candidates"], width=1.0, color="#1F3A5F")
    for s, lab in ((COURT_CUTS["rank_108000"], "rank 1,08,000"), (COURT_CUTS["rank_56000"], "rank 56,000")):
        a1.axvline(s, color="#C9A227", lw=1)
        a1.annotate(lab, (s, a1.get_ylim()[1] * 0.92), fontsize=6, rotation=90, ha="right", va="top")
    a1.set_xlabel("Marks", fontsize=7)
    a1.set_ylabel("Candidates on each total", fontsize=7)
    a1.tick_params(labelsize=6)
    a1.set_title("Every total, 2.33 million candidates", fontsize=8)
    sub = fr[fr["score"] >= 300]
    a2.plot(sub["score"], sub["ranks_5"], color="#1F3A5F", lw=1.2, label="one question (5 marks)")
    a2.plot(sub["score"], sub["candidates"], color="#C9A227", lw=1.0, label="candidates tied on the total")
    a2.set_yscale("log")
    a2.set_xlabel("Marks", fontsize=7)
    a2.set_ylabel("Ranks", fontsize=7)
    a2.tick_params(labelsize=6)
    a2.legend(frameon=False, fontsize=6)
    a2.set_title("Ranks crossed by one question", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fragility.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
