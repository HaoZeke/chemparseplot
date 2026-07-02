# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Public readcon-backed CON/convel helpers for eOn parsers.

All ``.con`` / ``.convel`` I/O in chemparseplot should go through this module so
metadata-native energies (``readcon>=0.7``) stay on one code path. ``readcon``
remains an optional extra (``chemparseplot[neb]``); import errors surface only
when these helpers are called.

```{versionadded} 1.8.1
```
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

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

CON_SUFFIXES = (".con", ".convel")


def is_con_path(path: str | Path) -> bool:
    """Return True when *path* looks like an eOn CON/convel file."""
    return Path(path).suffix.lower() in CON_SUFFIXES


def _readcon():
    try:
        import readcon
    except ImportError as exc:  # pragma: no cover - exercised when extra missing
        msg = (
            "readcon is required for CON I/O. Install with: "
            "pip install 'chemparseplot[neb]' (readcon>=0.7.0)"
        )
        raise ImportError(msg) from exc
    return readcon


def read_con_frames(path: str | Path) -> list[Any]:
    """Read all frames from a CON/convel file as ``readcon.ConFrame`` objects."""
    readcon = _readcon()
    return list(readcon.read_con(str(path)))


def read_con_as_ase(path: str | Path) -> list:
    """Read all frames from a CON/convel file as ASE ``Atoms`` objects."""
    readcon = _readcon()
    return list(readcon.read_con_as_ase(str(path)))


def read_first_atoms(path: str | Path):
    """Read the first frame of a CON file as an ASE ``Atoms`` object."""
    return read_con_as_ase(path)[0]


def frame_with_energy(frame: Any, energy: float) -> Any:
    """Return a ``ConFrame`` copy with per-frame total ``energy`` set (eV).

    Uses ``ConFrame.set_energy`` when available (readcon>=0.7) so metadata stays
    schema-valid; falls back to reconstructing the frame with string metadata.
    """
    readcon = _readcon()
    ConFrame = readcon.ConFrame
    metadata = frame.metadata
    if hasattr(metadata, "items"):
        metadata_dict = {str(k): v for k, v in metadata.items()}
    else:
        metadata_dict = dict(metadata or {})
    clone = ConFrame(
        frame.cell,
        frame.angles,
        frame.atoms,
        frame.prebox_header,
        frame.postbox_header,
        metadata_dict,
    )
    value = float(energy)
    setter = getattr(clone, "set_energy", None)
    if callable(setter):
        setter(value)
        return clone
    scalar = getattr(clone, "set_scalar_metadata", None)
    if callable(scalar):
        scalar("energy", value)
        return clone
    metadata_dict["energy"] = str(value)
    return ConFrame(
        frame.cell,
        frame.angles,
        frame.atoms,
        frame.prebox_header,
        frame.postbox_header,
        metadata_dict,
    )


def write_con_frames(path: str | Path, frames: Sequence[Any]) -> Path:
    """Write ``ConFrame`` objects to *path* via ``readcon.write_con``."""
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
    """Convert ASE ``Atoms`` to ``ConFrame`` and write a CON file.

    Optional *energies* (eV) are attached per frame with :func:`frame_with_energy`.
    """
    readcon = _readcon()
    frames = []
    for idx, atoms in enumerate(atoms_list):
        frame = readcon.ConFrame.from_ase(atoms)
        if energies is not None and idx < len(energies) and energies[idx] is not None:
            frame = frame_with_energy(frame, float(energies[idx]))
        frames.append(frame)
    return write_con_frames(path, frames)
