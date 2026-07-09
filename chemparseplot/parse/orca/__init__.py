# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""ORCA output parsers (public surface).

NEB and other structured products: use :func:`parse_orca_neb` and friends.
OPI is an optional internal backend — do not ``import opi`` for suite flows.
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
]
