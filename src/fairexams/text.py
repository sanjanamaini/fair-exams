"""Display names for centres and cities (NTA's lists are in capitals)."""
from __future__ import annotations

import re

FIXES = (("R.K.University", "R.K. University"), ("Dav ", "DAV "), ("Sr Sec", "Sr. Sec."), ("(Unit-2)    ", "(Unit-2) "),
         ("Kv ", "KV "), ("Dps ", "DPS "))


def nice_name(s: str) -> str:
    """Title case without capitalising after apostrophes (William's, not William'S)."""
    t = str(s).title()
    t = re.sub(r"(['’`])S\b", r"\1s", t)
    for a, b in FIXES:
        t = t.replace(a, b)
    return t
