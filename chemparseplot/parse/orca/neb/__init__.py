# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""ORCA NEB parsing utilities.

Supports both:
- OPI (ORCA Python Interface) for ORCA 6.1+ JSON output
- Legacy regex parsing for older ORCA versions
"""

from chemparseplot.parse.orca.neb.interp import extract_interp_points
from chemparseplot.parse.orca.neb.opi_parser import (
    HAS_OPI,
    parse_orca_neb,
    parse_orca_neb_fallback,
)

__all__ = [
    "HAS_OPI",
    "extract_interp_points",
    "parse_orca_neb",
    "parse_orca_neb_fallback",
]
