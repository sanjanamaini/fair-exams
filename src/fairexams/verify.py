"""Stage 1: check the mirror against NTA's own PDFs.

    python -m fairexams.verify              # 100 random centres plus every centre in the validation lists
    python -m fairexams.verify --flagged    # also every centre in results/tables/flags.csv

A centre passes only if the (serial, marks) pairs parsed from NTA's PDF equal the mirror's exactly.
Any failure exits non-zero, which stops `make`.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time

import numpy as np
import pandas as pd
import requests
from pypdf import PdfReader

from . import load
from .config import N_VERIFY, NTA_PDF, RAW, SEED, TABLES, USER_AGENT, VALIDATION

PDF_DIR = RAW / "nta_pdf"
NUMERIC_LINE = re.compile(r"^[\s\-\d]+$")
INTEGER = re.compile(r"-?\d+")
HEADER = re.compile(r"Centre:\s*(\d+)")


def parse_pdf(data: bytes) -> tuple[int, list[tuple[int, int]]]:
    """Return the centre code in the header and the (serial, marks) pairs, in serial order."""
    reader = PdfReader(io.BytesIO(data))
    code, nums = None, []
    for page in reader.pages:
        text = page.extract_text()
        m = HEADER.search(text)
        if m and code is None:
            code = int(m.group(1))
        body = text.split("Srlno", 1)[1] if "Srlno" in text else ""
        for line in body.splitlines():
            if line.strip() and NUMERIC_LINE.match(line):
                nums.extend(int(x) for x in INTEGER.findall(line))
    if len(nums) % 2:
        raise ValueError("odd number of integers in PDF body")
    pairs = sorted(zip(nums[0::2], nums[1::2]))
    serials = [s for s, _ in pairs]
    if serials != list(range(1, len(serials) + 1)):
        raise ValueError("serial numbers are not 1..n")
    return code, pairs


def get_pdf(code: int) -> bytes:
    path = PDF_DIR / ("%d.pdf" % code)
    if not path.exists():
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                r = requests.get(NTA_PDF.format(code=code), headers={"User-Agent": USER_AGENT}, timeout=60)
                r.raise_for_status()
                path.write_bytes(r.content)
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        time.sleep(0.3)  # be polite to NTA's server
    return path.read_bytes()


def check(codes, mirror: pd.DataFrame, reason: dict) -> pd.DataFrame:
    by_code = {c: g for c, g in mirror.groupby("code")}
    rows = []
    for code in codes:
        g = by_code.get(code)
        mine = sorted(zip(g["serial"].tolist(), g["marks"].tolist())) if g is not None else []
        try:
            header_code, pairs = parse_pdf(get_pdf(code))
            ok = header_code == code and pairs == mine
            note = "" if ok else ("header %s" % header_code if header_code != code else "pairs differ")
        except Exception as e:  # recorded as a failure, never skipped
            pairs, ok, note = [], False, "%s: %s" % (type(e).__name__, e)
        rows.append({"code": code, "reason": reason[code], "n_pdf": len(pairs), "n_mirror": len(mine),
                     "match": ok, "note": note})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagged", action="store_true")
    args = ap.parse_args()

    cent = load.centres()
    mirror = load.marks()
    rng = np.random.default_rng(SEED)
    reason = {}
    for c in sorted(rng.choice(cent["code"].to_numpy(), size=N_VERIFY, replace=False).tolist()):
        reason[c] = "random"
    for c in pd.read_csv(VALIDATION / "blind_list.csv")["centre_code"]:
        reason.setdefault(int(c), "validation list")
    out_name = "verification.csv"
    if args.flagged:
        flags = pd.read_csv(TABLES / "flags.csv")
        reason = {}
        for c in flags["code"]:
            reason[int(c)] = "flagged"
        out_name = "verification_flagged.csv"

    res = check(list(reason), mirror, reason)
    res.to_csv(TABLES / out_name, index=False)
    summary = {"verified_centres": int(len(res)), "verified_match": int(res["match"].sum()),
               "verified_candidates": int(res["n_pdf"].sum())}
    key = "verify_flagged" if args.flagged else "verify"
    (TABLES / ("%s_numbers.json" % key)).write_text(json.dumps(summary, indent=1))
    print(res.groupby("reason")["match"].agg(["size", "sum"]))
    print(json.dumps(summary))
    if not res["match"].all():
        print(res[~res["match"]].to_string())
        sys.exit(1)


if __name__ == "__main__":
    main()
