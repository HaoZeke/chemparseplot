# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""General readcon-core CON/convel integration.

This package is the suite home for **any** CON trajectory (not only eOn
job directories). eOn-specific movie/dat loaders remain under
:mod:`chemparseplot.parse.eon`; they call these helpers for I/O.

Public surface:

- I/O: :func:`read_con_frames`, :func:`read_con_as_ase`, :func:`write_con_frames`
- Tables: :func:`frames_to_table`, :func:`load_con_trajectory`
- Types: :class:`ConTrajectory`

.. versionadded:: 1.9.12
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
from chemparseplot.parse.con.trajectory import (
    ConTrajectory,
    frames_to_table,
    load_con_trajectory,
)

__all__ = [
    "CON_SUFFIXES",
    "ConTrajectory",
    "frame_energy_ev",
    "frame_positions_angstrom",
    "frame_with_energy",
    "frame_with_positions",
    "frames_to_table",
    "is_con_path",
    "iter_con_frames",
    "iter_corpus_frames",
    "load_con_trajectory",
    "read_con_as_ase",
    "read_con_frames",
    "read_first_atoms",
    "read_first_frame",
    "write_atoms_as_con",
    "write_con_frames",
]
