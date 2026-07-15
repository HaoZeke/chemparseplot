# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Back-compat re-export of :mod:`chemparseplot.parse.con.io`.

Prefer ``from chemparseplot.parse.con import ...`` for new code.
"""

from __future__ import annotations

from chemparseplot.parse.con.io import (  # noqa: F401
    CON_SUFFIXES,
    frame_with_energy,
    is_con_path,
    read_con_as_ase,
    read_con_frames,
    read_first_atoms,
    write_atoms_as_con,
    write_con_frames,
)

__all__ = [
    "CON_SUFFIXES",
    "frame_with_energy",
    "is_con_path",
    "read_con_as_ase",
    "read_con_frames",
    "read_first_atoms",
    "write_atoms_as_con",
    "write_con_frames",
]
