# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""ORCA Mulliken / Loewdin atomic population charges.

Text path ports wailord ``single_population_analysis``. When OPI is installed
and property JSON is available, prefer
:meth:`opi.output.core.Output.get_mulliken` /
:meth:`opi.output.core.Output.get_loewdin`.

.. versionadded:: 1.9.11
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chemparseplot.parse.orca._opi import opi_available

__all__ = ["AtomicCharge", "parse_orca_populations"]

PopKind = Literal["Mulliken", "Loewdin"]
_Backend = Literal["auto", "opi", "text"]

_HDR = {
    "Mulliken": re.compile(r"MULLIKEN ATOMIC CHARGES"),
    "Loewdin": re.compile(r"LOEWDIN ATOMIC CHARGES"),
}


@dataclass(frozen=True, slots=True)
class AtomicCharge:
    """One atomic charge (optional spin) from a population analysis."""

    atom_index: int
    symbol: str
    charge: float
    spin: float | None = None
    method: str = "Mulliken"


def _as_text(path_or_text: str | Path) -> str:
    p = Path(path_or_text) if not isinstance(path_or_text, Path) else path_or_text
    try:
        if p.is_file() and "\n" not in str(path_or_text):
            return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return str(path_or_text)


def _parse_text(text: str, kind: PopKind) -> list[AtomicCharge]:
    hdr = _HDR[kind]
    lines = text.splitlines()
    rows: list[AtomicCharge] = []
    for i, line in enumerate(lines):
        if not hdr.search(line):
            continue
        has_spin = "SPIN" in line.upper()
        j = i + 1
        while j < len(lines) and (
            set(lines[j].strip()) <= {"-"} or lines[j].strip() == ""
        ):
            j += 1
        while j < len(lines):
            if "Sum" in lines[j] or "--" in lines[j]:
                break
            parts = lines[j].split()
            if len(parts) < 3:
                break
            # formats: "0 H :  -0.1" or "0 H : -0.1  0.0"
            try:
                anum = int(parts[0])
            except ValueError:
                break
            # symbol may be "H" or "H:"
            sym = parts[1].rstrip(":")
            # Charge/spin floats only after the atom index + symbol (skip index).
            rest = parts[2:]
            nums = [float(x) for x in rest if _is_float(x)]
            if not nums:
                break
            spin = nums[1] if has_spin and len(nums) >= 2 else None
            charge = nums[0]
            rows.append(
                AtomicCharge(
                    atom_index=anum,
                    symbol=sym,
                    charge=charge,
                    spin=spin,
                    method=kind,
                )
            )
            j += 1
        if rows:
            break
    return rows


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_opi(basename: str, working_dir: Path, kind: PopKind) -> list[AtomicCharge]:
    from chemparseplot.parse.orca._opi import get_opi_output_class

    Output = get_opi_output_class()
    out = Output(basename, working_dir=working_dir)
    # property JSON path
    try:
        if kind == "Mulliken":
            blocks = out.get_mulliken()
        else:
            blocks = out.get_loewdin()
    except Exception:
        return []
    if not blocks:
        return []
    rows: list[AtomicCharge] = []
    for block in blocks:
        charges = getattr(block, "atomiccharges", None) or []
        # atomiccharges is list[list[float]]
        for flat in charges:
            for idx, ch in enumerate(flat):
                rows.append(
                    AtomicCharge(
                        atom_index=idx,
                        symbol="?",
                        charge=float(ch),
                        spin=None,
                        method=kind,
                    )
                )
    return rows


def parse_orca_populations(
    path_or_text: str | Path,
    kind: PopKind = "Mulliken",
    *,
    backend: _Backend = "auto",
    basename: str | None = None,
    working_dir: str | Path | None = None,
) -> list[AtomicCharge]:
    """Parse Mulliken or Loewdin atomic charges.

    Prefer OPI property JSON when ``basename`` is set and ``opi`` is available;
    otherwise parse the text ``.out`` tables (wailord-compatible).
    """
    if kind not in _HDR:
        msg = f"Unsupported population kind {kind!r}; use Mulliken or Loewdin"
        raise ValueError(msg)

    if backend in ("auto", "opi") and basename is not None and opi_available():
        wd = Path(working_dir) if working_dir is not None else Path.cwd()
        rows = _parse_opi(basename, wd, kind)
        if rows or backend == "opi":
            return rows
    if backend == "opi":
        return []

    rows = _parse_text(_as_text(path_or_text), kind)
    if not rows:
        msg = f"{kind} atomic charges not found in ORCA output"
        raise ValueError(msg)
    return rows
