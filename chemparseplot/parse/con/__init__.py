# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""General readcon-core CON/convel (+ chemfiles) integration.

This package is the suite home for **any** CON trajectory (not only eOn
job directories). eOn-specific movie/dat loaders remain under
:mod:`chemparseplot.parse.eon`; they call these helpers for I/O.

Public surface:

- I/O: :func:`read_con_frames`, :func:`read_con_as_ase`, :func:`write_con_frames`
- Tables: :func:`frames_to_table`, :func:`load_con_trajectory`, :func:`load_trajectory`
- Chemfiles: :func:`chemfiles_available`, :func:`load_chemfiles_trajectory`
- Types: :class:`ConTrajectory`

.. versionadded:: 1.9.12
"""

from __future__ import annotations

from chemparseplot.parse.con.chemfiles import (
    chemfiles_available,
    load_chemfiles_trajectory,
    read_chemfiles_frames,
)
from chemparseplot.parse.con.io import (
    CON_SUFFIXES,
    frame_with_energy,
    is_con_path,
    read_con_as_ase,
    read_con_frames,
    read_first_atoms,
    write_atoms_as_con,
    write_con_frames,
)
from chemparseplot.parse.con.trajectory import (
    ConTrajectory,
    frames_to_table,
    load_con_trajectory,
    load_trajectory,
)

__all__ = [
    "CON_SUFFIXES",
    "ConTrajectory",
    "chemfiles_available",
    "frame_with_energy",
    "frames_to_table",
    "is_con_path",
    "load_chemfiles_trajectory",
    "load_con_trajectory",
    "load_trajectory",
    "read_chemfiles_frames",
    "read_con_as_ase",
    "read_con_frames",
    "read_first_atoms",
    "write_atoms_as_con",
    "write_con_frames",
]
