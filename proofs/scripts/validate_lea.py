#!/usr/bin/env python3
"""Check Lea labels in supplement.org against Lean theorem names.

Same mechanical contract as d-SEAMS 2.0 validate_lea_si.py: every
`% lea: formalize label=...` has a `theorem <label>` in the Lean tree,
and the Lean tree has no `sorry`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORG = ROOT / "org" / "supplement.org"
LEAN = ROOT / "lean"


def main() -> int:
    org = ORG.read_text()
    labels = re.findall(r"% lea: formalize label=([A-Za-z0-9_]+)", org)
    if not labels:
        print("no lea formalize markers in", ORG, file=sys.stderr)
        return 1
    lean_src = "\n".join(p.read_text() for p in LEAN.rglob("*.lean"))
    missing = []
    for lab in labels:
        if not re.search(rf"^theorem {re.escape(lab)}\b", lean_src, re.M):
            missing.append(lab)
    if "sorry" in lean_src:
        missing.append("sorry in lean")
    extra = []
    for m in re.finditer(r"^theorem ([A-Za-z0-9_]+)\b", lean_src, re.M):
        name = m.group(1)
        if name not in labels and name not in {"lerp_const"}:
            extra.append(name)
    if missing:
        print("lea labels missing Lean theorems:", *missing, sep="\n  ", file=sys.stderr)
        return 1
    print(f"ok: {len(labels)} Lea labels; supporting {extra or 'none extra'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
