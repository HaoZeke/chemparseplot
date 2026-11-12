# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Public CON/convel helpers.

The implementation lives in :mod:`chemparseplot.parse.eon.con_io`, which is
the single readcon entry point. New code can import from
:mod:`chemparseplot.parse.con`.
"""

from __future__ import annotations

from chemparseplot.parse.eon.con_io import (
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


def _readcon():
    """Forward to the readcon loader so patches of ``eon.con_io`` apply."""
    from chemparseplot.parse.eon.con_io import _readcon as impl

    return impl()


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
