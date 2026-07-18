# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Grammar / AST track for text-heavy formats (parsimonious).

Optional extra: ``pip install 'chemparseplot[grammar]'``.

Prefer structured APIs (OPI) for ORCA products that expose them; use this
package for XYZ and text-only ORCA sections (energy lines, Cartesian blocks).

```{versionadded} 1.9.9
```
"""

from __future__ import annotations

from chemparseplot.parse.grammar._deps import (
    grammar_available,
    reset_grammar_cache,
)

__all__ = [
    "grammar_available",
    "parse_orca_text_file",
    "parse_orca_text_summary",
    "parse_xyz_file",
    "parse_xyz_text",
    "reset_grammar_cache",
]


def __getattr__(name: str):
    if name in (
        "parse_xyz_text",
        "parse_xyz_file",
        "XyzFrame",
        "XyzAtom",
        "write_xyz_text",
    ):
        from chemparseplot.parse.grammar import xyz as _xyz

        return getattr(_xyz, name)
    if name in (
        "parse_orca_text_summary",
        "parse_orca_text_file",
        "OrcaTextSummary",
        "OrcaAtomCoord",
        "extract_final_energies_hartree",
        "extract_cartesian_angstrom_blocks",
    ):
        from chemparseplot.parse.grammar import orca_text as _ot

        return getattr(_ot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
