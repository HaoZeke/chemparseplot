# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Public readcon-backed CON/convel helpers for eOn parsers.

All ``.con`` / ``.convel`` I/O in chemparseplot goes through this module.
``readcon`` stays an optional extra (``chemparseplot[neb]``). Import errors
surface only when these helpers are called. Writers omit precision and leave
canonical off, which is RoundTrip on ``readcon>=0.17.1``. Nothing here opens
a corpus.

```{versionadded} 1.8.1
```
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CON_SUFFIXES",
    "frame_with_energy",
    "frame_with_positions",
    "is_con_path",
    "read_con_as_ase",
    "read_con_frames",
    "read_first_atoms",
    "read_first_frame",
    "write_atoms_as_con",
    "write_con_frames",
]

CON_SUFFIXES = (".con", ".convel")

_READCON_HINT = (
    "readcon is required for CON I/O. Install with: "
    "pip install 'chemparseplot[neb]' (readcon>=0.17.1)"
)


def is_con_path(path: str | Path) -> bool:
    """Return True when *path* looks like an eOn CON/convel file."""
    return Path(path).suffix.lower() in CON_SUFFIXES


def _readcon():
    try:
        import readcon
    except ImportError as exc:  # pragma: no cover - exercised when extra missing
        raise ImportError(_READCON_HINT) from exc
    return readcon


def _metadata_dict(frame: Any) -> dict[str, Any]:
    metadata = frame.metadata
    if hasattr(metadata, "items"):
        return {str(key): value for key, value in metadata.items()}
    return dict(metadata or {})


def _clone_frame(frame: Any) -> Any:
    """Copy a frame through the ``ConFrame`` constructor so Atom sections travel."""
    readcon = _readcon()
    return readcon.ConFrame(
        frame.cell,
        frame.angles,
        list(frame.atoms),
        frame.prebox_header,
        frame.postbox_header,
        _metadata_dict(frame),
    )


def read_con_frames(path: str | Path) -> list[Any]:
    """Read all frames from a CON/convel file as ``readcon.ConFrame`` objects."""
    readcon = _readcon()
    return list(readcon.read_con(str(path)))


def read_con_as_ase(path: str | Path) -> list:
    """Read all frames from a CON/convel file as ASE ``Atoms`` objects."""
    readcon = _readcon()
    return list(readcon.read_con_as_ase(str(path)))


def read_first_frame(path: str | Path) -> Any:
    """Read the first CON frame and stop. Later frames are not parsed."""
    readcon = _readcon()
    return readcon.read_first_frame(str(path))


def read_first_atoms(path: str | Path):
    """Read the first frame of a CON file as an ASE ``Atoms`` object."""
    return read_first_frame(path).to_ase()


def frame_with_energy(frame: Any, energy: float) -> Any:
    """Return a ``ConFrame`` copy whose total energy is *energy* (eV).

    The copy keeps Atom sections. Energy is applied only with
    ``ConFrame.set_energy``, which rejects a non-finite value. A wheel
    without ``set_energy`` raises the ``readcon>=0.17.1`` install hint.
    """
    clone = _clone_frame(frame)
    setter = getattr(clone, "set_energy", None)
    if not callable(setter):
        raise ImportError(_READCON_HINT)
    setter(float(energy))
    return clone


def frame_with_positions(frame: Any, positions: Any) -> Any:
    """Return a ``ConFrame`` copy with Cartesian positions replaced.

    Each ``Atom`` row is copied and only ``x`` / ``y`` / ``z`` are written.
    ``fixed``, ``atom_id``, ``mass``, velocities, forces, displacements, and
    spreads stay. That is correct for an unmoved slice or a pure translation.
    It is the wrong helper for ``displacement.con``: a rotation or a reorder
    must drop those sections and let eOn recompute them.
    """
    coords = np.asarray(positions, dtype=np.float64)
    n_atoms = len(frame.atoms)
    if coords.shape != (n_atoms, 3):
        msg = f"positions must have shape ({n_atoms}, 3), got {tuple(coords.shape)}"
        raise ValueError(msg)
    clone = _clone_frame(frame)
    for atom, xyz in zip(clone.atoms, coords, strict=True):
        atom.x = xyz[0].item()
        atom.y = xyz[1].item()
        atom.z = xyz[2].item()
    return clone


def write_con_frames(path: str | Path, frames: Sequence[Any]) -> Path:
    """Write ``ConFrame`` objects to *path* via ``readcon.write_con``.

    Precision is omitted and canonical stays off. There is no corpus write.
    """
    readcon = _readcon()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    readcon.write_con(str(out), list(frames))
    return out


def write_atoms_as_con(
    path: str | Path,
    atoms_list: Sequence[Any],
    *,
    energies: Sequence[float | None] | None = None,
) -> Path:
    """Convert ASE ``Atoms`` that were never a ``ConFrame`` and write CON.

    Optional *energies* (eV) are attached per frame with
    :func:`frame_with_energy`. ``from_ase`` drops displacements and spreads.
    ASE tags become atom ids only when they are distinct and non-negative;
    otherwise ids are sequential. Two atoms of one symbol whose masses
    disagree past the reader tolerance fail at ``build``. Precision is
    omitted and canonical stays off.
    """
    readcon = _readcon()
    frames = []
    for idx, atoms in enumerate(atoms_list):
        frame = readcon.ConFrame.from_ase(atoms)
        if energies is not None and idx < len(energies) and energies[idx] is not None:
            frame = frame_with_energy(frame, float(energies[idx]))
        frames.append(frame)
    return write_con_frames(path, frames)
