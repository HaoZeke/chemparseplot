# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""XYZ grammar (parsimonious) for multi-atom frames.

Port of the wailord paper thesis: structured grammar parsing for XYZ rather
than ad-hoc line splits. Multi-character element symbols and scientific
floats are supported.

```{versionadded} 1.9.9
```
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chemparseplot.parse.grammar._deps import (
    get_grammar_class,
    get_node_visitor_class,
)

# Each rule that needs a visitor must be a real expression (not an alias);
# parsimonious collapses pure aliases so visit_natoms would never fire.
_XYZ_GRAMMAR_SRC = r"""
meta          = natoms newline comment_line newline coord_block ws?
natoms        = ~r"\d+"
comment_line  = ~r"[^\n]*"
coord_block   = (atom_line newline?)+
atom_line     = ws? atype ws float_ ws float_ ws float_ ws?
atype         = ~r"[A-Za-z][a-zA-Z0-9]{0,2}"
float_        = ~r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
newline       = ~r"\n"
ws            = ~r"[ \t]+"
"""

_grammar_cache: Any | None = None


def xyz_grammar() -> Any:
    """Compiled XYZ grammar (cached)."""
    global _grammar_cache
    if _grammar_cache is None:
        Grammar = get_grammar_class()
        _grammar_cache = Grammar(_XYZ_GRAMMAR_SRC)
    return _grammar_cache


@dataclass(frozen=True, slots=True)
class XyzAtom:
    """One atom line: element symbol and Cartesian coordinates (Å)."""

    symbol: str
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class XyzFrame:
    """Parsed single-frame XYZ."""

    natoms: int
    comment: str
    atoms: tuple[XyzAtom, ...]

    @property
    def symbols(self) -> list[str]:
        return [a.symbol for a in self.atoms]

    @property
    def counts(self) -> Counter:
        return Counter(self.symbols)

    @property
    def system(self) -> str:
        """Composition slug as ``ElementCount`` joined (e.g. ``H2``, ``C1O2``)."""
        flat: list[str] = []
        for elem, n in self.counts.items():
            flat.append(elem)
            flat.append(str(n))
        return "".join(flat)

    def coordinates(self) -> list[tuple[float, float, float]]:
        return [(a.x, a.y, a.z) for a in self.atoms]


def _flatten(xs: Any) -> list[Any]:
    out: list[Any] = []
    if isinstance(xs, list):
        for x in xs:
            out.extend(_flatten(x))
    elif xs is not None and xs != "":
        out.append(xs)
    return out


def _make_visitor() -> Any:
    NodeVisitor = get_node_visitor_class()

    class XyzVisitor(NodeVisitor):
        def __init__(self) -> None:
            self.natoms: int | None = None
            self.comment: str = ""
            self.atoms: list[XyzAtom] = []

        def visit_natoms(self, node: Any, _visited_children: Any) -> int:
            self.natoms = int(node.text)
            return self.natoms

        def visit_comment_line(self, node: Any, _visited_children: Any) -> str:
            self.comment = node.text
            return self.comment

        def visit_atom_line(self, node: Any, visited_children: Any) -> XyzAtom:
            del node  # structure from visited_children
            flat = _flatten(visited_children)
            symbol = next(
                x for x in flat if isinstance(x, str) and x.strip() and x[0].isalpha()
            )
            nums = [x for x in flat if isinstance(x, float)]
            atom = XyzAtom(symbol=symbol, x=nums[0], y=nums[1], z=nums[2])
            self.atoms.append(atom)
            return atom

        def visit_atype(self, node: Any, _visited_children: Any) -> str:
            return node.text

        def visit_float_(self, node: Any, _visited_children: Any) -> float:
            return float(node.text)

        def generic_visit(self, node: Any, visited_children: Any) -> Any:
            return visited_children or node.text

    return XyzVisitor()


def parse_xyz_text(text: str) -> XyzFrame:
    """Parse a single-frame XYZ string with the grammar visitor.

    Parameters
    ----------
    text:
        Full XYZ content including atom count and comment line.

    Returns
    -------
    XyzFrame
        Structured frame (natoms, comment, atoms).
    """
    body = text if text.endswith("\n") else text + "\n"
    tree = xyz_grammar().parse(body)
    visitor = _make_visitor()
    visitor.visit(tree)
    natoms = visitor.natoms if visitor.natoms is not None else len(visitor.atoms)
    if natoms != len(visitor.atoms):
        msg = f"XYZ natoms={natoms} but parsed {len(visitor.atoms)} atom lines"
        raise ValueError(msg)
    return XyzFrame(
        natoms=natoms,
        comment=visitor.comment,
        atoms=tuple(visitor.atoms),
    )


def parse_xyz_file(path: str | Path) -> XyzFrame:
    """Read and parse an XYZ file."""
    return parse_xyz_text(Path(path).read_text())


def write_xyz_text(
    frame: XyzFrame,
    *,
    comment: str | None = None,
) -> str:
    """Serialize a frame back to XYZ text (Å)."""
    c = frame.comment if comment is None else comment
    lines = [str(frame.natoms), c.rstrip("\n")]
    for a in frame.atoms:
        lines.append(f"{a.symbol:3s} {a.x:12.5f} {a.y:12.5f} {a.z:12.5f}")
    return "\n".join(lines) + "\n"
