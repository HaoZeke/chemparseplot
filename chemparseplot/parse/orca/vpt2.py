# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""ORCA VPT2 fundamental transition tables.

Ports the wailord ``orcaVis.vpt2_transitions`` text extractor. OPI does not
currently expose a dedicated VPT2 fundamentals API on ``Output``; when that
lands, prefer it behind ``backend=auto`` the same way as IR.

.. versionadded:: 1.9.11
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Vpt2Fundamental", "parse_orca_vpt2_fundamentals"]

_HDR = re.compile(r"Fundamental transition", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Vpt2Fundamental:
    """One VPT2 fundamental transition row."""

    mode: int
    harmonic_cm1: float
    vpt2_cm1: float
    diff_cm1: float


def _as_text(path_or_text: str | Path) -> str:
    p = Path(path_or_text) if not isinstance(path_or_text, Path) else path_or_text
    try:
        if p.is_file() and "\n" not in str(path_or_text):
            return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return str(path_or_text)


def parse_orca_vpt2_fundamentals(path_or_text: str | Path) -> list[Vpt2Fundamental]:
    """Parse the ``Fundamental transition`` VPT2 table from ORCA output text."""
    text = _as_text(path_or_text)
    lines = text.splitlines()
    rows: list[Vpt2Fundamental] = []
    for i, line in enumerate(lines):
        if not _HDR.search(line):
            continue
        j = i + 1
        # skip underlines / column headers
        while j < len(lines) and (
            set(lines[j].strip()) <= {"-", "="}
            or "Mode" in lines[j]
            or "w[i]" in lines[j]
            or lines[j].strip() == ""
        ):
            j += 1
        while j < len(lines):
            if "---" in lines[j] or set(lines[j].strip()) <= {"-"}:
                break
            parts = lines[j].split()
            if len(parts) < 4:
                break
            try:
                rows.append(
                    Vpt2Fundamental(
                        mode=int(parts[0]),
                        harmonic_cm1=float(parts[1]),
                        vpt2_cm1=float(parts[2]),
                        diff_cm1=float(parts[3]),
                    )
                )
            except ValueError:
                break
            j += 1
        if rows:
            break
    if not rows:
        msg = "VPT2 fundamental transition table not found (did you run VPT2?)"
        raise ValueError(msg)
    return rows
