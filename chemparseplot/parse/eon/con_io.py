# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Back-compat re-export of :mod:`chemparseplot.parse.con.io`.

Prefer ``from chemparseplot.parse.con import ...`` for new code.
Existing eOn callers keep importing this module.
"""

from __future__ import annotations

from chemparseplot.parse.con.io import (
    CON_SUFFIXES,
    frame_energy_ev,
    frame_positions_angstrom,
    frame_with_energy,
    frame_with_positions,
    is_con_path,
    iter_con_frames,
    iter_corpus_frames,
    read_con_as_ase,
    read_con_frames,
    read_first_atoms,
    read_first_frame,
    write_atoms_as_con,
    write_con_frames,
)

__all__ = [
    "CON_SUFFIXES",
    "frame_energy_ev",
    "frame_positions_angstrom",
    "frame_with_energy",
    "frame_with_positions",
    "is_con_path",
    "iter_con_frames",
    "iter_corpus_frames",
    "read_con_as_ase",
    "read_con_frames",
    "read_first_atoms",
    "read_first_frame",
    "write_atoms_as_con",
    "write_con_frames",
]
