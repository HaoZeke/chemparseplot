# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Chemfiles bridge via readcon-core (XYZ/PDB/GRO/… → ConFrame).

Requires a readcon wheel built with chemfiles (``readcon-chemfiles`` or a
conda/pixi build that enables ``has_chemfiles_support()``).

.. versionadded:: 1.9.12
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "chemfiles_available",
    "load_chemfiles_trajectory",
    "read_chemfiles_frames",
]


def chemfiles_available() -> bool:
    """True when the installed readcon build links chemfiles."""
    try:
        import readcon
    except ImportError:
        return False
    probe = getattr(readcon, "has_chemfiles_support", None)
    if callable(probe):
        try:
            return bool(probe())
        except Exception:
            return False
    return hasattr(readcon, "read_chemfiles")


def _require_chemfiles():
    from chemparseplot.parse.con.io import _readcon

    readcon = _readcon()
    if not chemfiles_available():
        msg = (
            "This readcon build has no chemfiles support. "
            "Install a chemfiles-enabled package (e.g. readcon-chemfiles / "
            "conda-forge readcon with chemfiles) to load XYZ/PDB/GRO/… "
            "via readcon.read_chemfiles."
        )
        raise ImportError(msg)
    return readcon


def read_chemfiles_frames(path: str | Path) -> list[Any]:
    """Read all frames from a chemfiles-supported path as ``ConFrame`` list."""
    readcon = _require_chemfiles()
    return list(readcon.read_chemfiles(str(path)))


def load_chemfiles_trajectory(path: str | Path):
    """Load XYZ/PDB/GRO/… into a :class:`ConTrajectory` via chemfiles."""
    from chemparseplot.parse.con.trajectory import ConTrajectory, frames_to_table

    path = Path(path)
    frames = read_chemfiles_frames(path)
    atoms_list = [frame.to_ase() for frame in frames]
    return ConTrajectory(
        path=path.resolve(),
        frames=frames,
        atoms_list=atoms_list,
        table=frames_to_table(frames),
        source="chemfiles",
    )
