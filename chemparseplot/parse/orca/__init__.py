# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""ORCA output parsers (public surface).

NEB and other structured products: use :func:`parse_orca_neb` and friends.
OPI is an optional internal backend — do not ``import opi`` for suite flows.

Text-heavy sections (final energy lines, Cartesian blocks): grammar track via
:func:`parse_orca_text_summary` (requires ``chemparseplot[grammar]``).
"""

from chemparseplot.parse.orca import geomscan
from chemparseplot.parse.orca.neb import (
    parse_orca_neb,
    parse_orca_neb_fallback,
)

__all__ = [
    "geomscan",
    "parse_orca_neb",
    "parse_orca_neb_fallback",
    "parse_orca_text_summary",
]


def __getattr__(name: str):
    if name == "parse_orca_text_summary":
        from chemparseplot.parse.grammar.orca_text import parse_orca_text_summary

        return parse_orca_text_summary
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
