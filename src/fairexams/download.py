"""Stage 1: download the centre-wise marks (mirror CSVs) and NFHS-5 district factsheets.

    python -m fairexams.download

Every file lands in data/raw/ (git-ignored) with a SHA-256 checksum in data/raw/MANIFEST.csv.
NTA's own PDFs are fetched by fairexams.verify, which checks the mirror against them.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import time

import requests

from .config import MIRROR, MIRROR_FILES, RAW, USER_AGENT

NFHS_URL = "https://raw.githubusercontent.com/jvargh7/nfhs5_factsheets/main/data%20for%20analysis/districts.csv"


def fetch(url: str, dest, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    jobs = [(MIRROR + name, RAW / "mirror" / name) for name in MIRROR_FILES]
    jobs.append((NFHS_URL, RAW / "nfhs5" / "districts.csv"))
    rows = []
    for url, dest in jobs:
        fetch(url, dest)
        rows.append({"url": url, "path": str(dest.relative_to(RAW)), "bytes": dest.stat().st_size,
                     "sha256": sha256(dest), "fetched": dt.date.today().isoformat()})
        print("%-45s %12d bytes" % (dest.relative_to(RAW), dest.stat().st_size))
    with open(RAW / "MANIFEST.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
