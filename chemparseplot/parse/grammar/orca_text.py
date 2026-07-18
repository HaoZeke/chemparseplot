# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Grammar-oriented extractors for text-heavy ORCA output sections.

Prefer structured backends (OPI) when available. This module covers sections
that remain text dumps: final single-point energies and Cartesian geometry
blocks. Foundation for VPT2/IR/populations ports from wailord.

```{versionadded} 1.9.9
```
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chemparseplot.parse.grammar._deps import (
    get_grammar_class,
    get_node_visitor_class,
)
from chemparseplot.units import Q_

# Named rules are real expressions (not aliases) so visitors fire reliably.
_ENERGY_LINE_GRAMMAR_SRC = r"""
energy_line = label ws value ws?
label       = "FINAL SINGLE POINT ENERGY"
value       = ~r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
ws          = ~r"[ \t]+"
"""

_CART_BLOCK_GRAMMAR_SRC = r"""
cart_block  = header newline dashes newline (atom_line newline)+
header      = "CARTESIAN COORDINATES (ANGSTROEM)"
dashes      = ~r"-{3,}"
atom_line   = ws? atype ws float_ ws float_ ws float_ ws?
atype       = ~r"[A-Za-z][a-zA-Z0-9]{0,2}"
float_      = ~r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
newline     = ~r"\n"
ws          = ~r"[ \t]+"
"""

_energy_grammar: Any | None = None
_cart_grammar: Any | None = None

_ENERGY_LINE_RE = re.compile(
    r"^.*FINAL SINGLE POINT ENERGY[ \t]+"
    r"([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?).*$",
    re.MULTILINE,
)

_CART_BLOCK_RE = re.compile(
    r"CARTESIAN COORDINATES \(ANGSTROEM\)\n"
    r"-{3,}\n"
    r"((?:[ \t]*[A-Za-z][a-zA-Z0-9]{0,2}[ \t]+"
    r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[ \t]+"
    r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[ \t]+"
    r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[ \t]*\n)+)",
    re.MULTILINE,
)

_N_ATOMS_RE = re.compile(
    r"Number of atoms\s*(?:\.{3,}|\.\.\.)\s*(\d+)",
)


def _energy_g() -> Any:
    global _energy_grammar
    if _energy_grammar is None:
        Grammar = get_grammar_class()
        _energy_grammar = Grammar(_ENERGY_LINE_GRAMMAR_SRC)
    return _energy_grammar


def _cart_g() -> Any:
    global _cart_grammar
    if _cart_grammar is None:
        Grammar = get_grammar_class()
        _cart_grammar = Grammar(_CART_BLOCK_GRAMMAR_SRC)
    return _cart_grammar


def _flatten(xs: Any) -> list[Any]:
    out: list[Any] = []
    if isinstance(xs, list):
        for x in xs:
            out.extend(_flatten(x))
    elif xs is not None and xs != "":
        out.append(xs)
    return out


@dataclass(frozen=True, slots=True)
class OrcaAtomCoord:
    """One atom in an ORCA Cartesian (Å) block."""

    symbol: str
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class OrcaTextSummary:
    """Structured slice of text ORCA output."""

    energies_hartree: tuple[float, ...]
    last_geometry: tuple[OrcaAtomCoord, ...]
    n_atoms: int | None

    @property
    def final_energy_hartree(self) -> float | None:
        if not self.energies_hartree:
            return None
        return self.energies_hartree[-1]

    @property
    def final_energy(self) -> Any:
        """Last final single-point energy as a pint Quantity (hartree)."""
        e = self.final_energy_hartree
        if e is None:
            return Q_([], "hartree")
        return Q_(e, "hartree")


def extract_final_energies_hartree(text: str) -> list[float]:
    """All ``FINAL SINGLE POINT ENERGY`` values in document order.

    Candidate lines are selected with a line scan; each is validated and
    extracted with a parsimonious line grammar.
    """
    g = _energy_g()
    NodeVisitor = get_node_visitor_class()

    class EnergyVisitor(NodeVisitor):
        def __init__(self) -> None:
            self.value: float | None = None

        def visit_value(self, node: Any, visited_children: Any) -> float:
            self.value = float(node.text)
            return self.value

        def generic_visit(self, node: Any, visited_children: Any) -> Any:
            return visited_children or node.text

    out: list[float] = []
    for m in _ENERGY_LINE_RE.finditer(text):
        line = m.group(0).strip()
        try:
            tree = g.parse(line)
        except Exception:
            continue
        v = EnergyVisitor()
        v.visit(tree)
        if v.value is not None:
            out.append(v.value)
    return out


def extract_cartesian_angstrom_blocks(text: str) -> list[list[OrcaAtomCoord]]:
    """Parse every ``CARTESIAN COORDINATES (ANGSTROEM)`` block via grammar."""
    g = _cart_g()
    NodeVisitor = get_node_visitor_class()

    class CartVisitor(NodeVisitor):
        def __init__(self) -> None:
            self.atoms: list[OrcaAtomCoord] = []

        def visit_atom_line(self, node: Any, visited_children: Any) -> OrcaAtomCoord:
            flat = _flatten(visited_children)
            symbol = next(
                x for x in flat if isinstance(x, str) and x.strip() and x[0].isalpha()
            )
            nums = [x for x in flat if isinstance(x, float)]
            atom = OrcaAtomCoord(symbol=symbol, x=nums[0], y=nums[1], z=nums[2])
            self.atoms.append(atom)
            return atom

        def visit_atype(self, node: Any, visited_children: Any) -> str:
            return node.text

        def visit_float_(self, node: Any, visited_children: Any) -> float:
            return float(node.text)

        def generic_visit(self, node: Any, visited_children: Any) -> Any:
            return visited_children or node.text

    blocks: list[list[OrcaAtomCoord]] = []
    for m in _CART_BLOCK_RE.finditer(text):
        chunk = (
            "CARTESIAN COORDINATES (ANGSTROEM)\n"
            "---------------------------------\n" + m.group(1)
        )
        if not chunk.endswith("\n"):
            chunk += "\n"
        try:
            tree = g.parse(chunk)
        except Exception:
            continue
        v = CartVisitor()
        v.visit(tree)
        if v.atoms:
            blocks.append(v.atoms)
    return blocks


def extract_n_atoms(text: str) -> int | None:
    """Best-effort ``Number of atoms`` from ORCA text (last occurrence)."""
    matches = _N_ATOMS_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def parse_orca_text_summary(text: str) -> OrcaTextSummary:
    """Summarize final energies and last Cartesian geometry from ORCA text."""
    energies = extract_final_energies_hartree(text)
    blocks = extract_cartesian_angstrom_blocks(text)
    last_geom: tuple[OrcaAtomCoord, ...] = ()
    if blocks:
        last_geom = tuple(blocks[-1])
    return OrcaTextSummary(
        energies_hartree=tuple(energies),
        last_geometry=last_geom,
        n_atoms=extract_n_atoms(text),
    )


def parse_orca_text_file(path: str | Path) -> OrcaTextSummary:
    """Read an ORCA ``.out`` and return :class:`OrcaTextSummary`."""
    return parse_orca_text_summary(Path(path).read_text())
