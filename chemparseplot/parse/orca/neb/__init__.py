# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""ORCA NEB parsing utilities (public).

Backends (selected via :func:`parse_orca_neb` ``backend=``):

- **OPI** (optional): ORCA 6.1+ structured output via the ``opi`` package,
  loaded only inside chemparseplot — not a public consumer dependency.
- **legacy**: regex / ``.interp`` parsing for older ORCA outputs.
"""

from chemparseplot.parse.orca._opi import opi_available
from chemparseplot.parse.orca.neb.interp import extract_interp_points
from chemparseplot.parse.orca.neb.opi_parser import (
    HAS_OPI,
    parse_orca_neb,
    parse_orca_neb_fallback,
)

__all__ = [
    "HAS_OPI",
    "extract_interp_points",
    "opi_available",
    "parse_orca_neb",
    "parse_orca_neb_fallback",
]
