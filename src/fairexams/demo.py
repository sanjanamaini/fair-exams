"""The monitor on a synthetic exam, for the report's deployment section.

    python -m fairexams.demo

40 cities (half of them sorting candidates into stronger and weaker centres), one centre where 30
candidates get top marks. Records what each layer of the monitor finds.
"""
from __future__ import annotations

import json

from .config import SEED, TABLES
from .monitor import run_screen, simulate


def main() -> None:
    cand = simulate()
    c, s = run_screen(cand, seeds=tuple(SEED + i for i in range(20)))
    planted = c[c["centre"] == "C07-01"].iloc[0]
    numbers = {"demo_candidates": s["candidates"], "demo_centres": s["centres"], "demo_cities": s["cities"],
               "demo_flags": s["flags"], "demo_flags_exact": s["flags_exact"], "demo_flags_tail": s["flags_tail"],
               "demo_both": int((c["flag"] & c["flag_tail"]).sum()),
               "demo_planted_z": float(planted["z_adj"]), "demo_planted_ztail": float(planted["z_tail"]),
               "demo_planted_flag_rate": float(planted["flag_rate"]), "demo_planted_tail_rate": float(planted["flag_tail_rate"]),
               "demo_planted_both": bool(planted["flag"] and planted["flag_tail"]),
               "demo_rho_top": s["rho_top"], "demo_design_effect": s["median_design_effect"]}
    (TABLES / "demo_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(json.dumps(numbers, indent=1, default=float))


if __name__ == "__main__":
    main()
