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

from collections.abc import Iterator, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

__all__ = [
    "CON_SUFFIXES",
    "frame_with_energy",
    "frame_energy_eV",
    "iter_con_frames",
    "iter_corpus_frames",
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


def iter_con_frames(path: str | Path) -> Iterator[Any]:
    """Yield CON frames without retaining the complete trajectory."""
    yield from _readcon().iter_con(str(path))


def iter_corpus_frames(
    path: str | Path, *, trajectory: int | None = None,
    require_energy: bool = False, require_forces: bool = False,
) -> Iterator[Any]:
    """Yield selected stored frames in trajectory and frame-index order.

    The corpus is opened read-only. Frame decoding is lazy; selection retains
    only the trajectory and frame keys, including for compressed stores.
    """
    from readcon_db import ConCorpus

    corpus = ConCorpus(str(path), readonly=True)
    keys = corpus.select(traj_id=trajectory, require_energy=require_energy,
                         require_forces=require_forces)
    for trajectory_id, frame_index in sorted(keys):
        yield corpus.get_frame(trajectory_id, frame_index)


def _energy_unit(frame: Any) -> str:
    return frame.metadata.get("units", {}).get("energy", "eV")


def _energy_factor(source: str, target: str) -> float:
    if source == target:
        return 1.0
    from readcon_db import unit_conversion_factor

    return unit_conversion_factor(source, target)


def frame_energy_eV(frame: Any) -> float | None:
    """Return the stored total energy in eV without evaluating a potential."""
    value = frame.metadata.get("energy")
    if value is None:
        value = getattr(frame, "energy", None)
    if value is None:
        return None
    return float(value) * _energy_factor(_energy_unit(frame), "eV")


def read_con_as_ase(path: str | Path) -> list:
    """Read all frames from a CON/convel file as ASE ``Atoms`` objects."""
    readcon = _readcon()
    return list(readcon.read_con_as_ase(str(path)))


def read_first_atoms(path: str | Path):
    """Read the first frame of a CON file as an ASE ``Atoms`` object."""
    return _readcon().read_first_frame(str(path)).to_ase()


def frame_with_energy(frame: Any, energy: float) -> Any:
    """Return a ``ConFrame`` copy with the supplied total energy in eV.

    The stored number uses the frame's declared energy unit. Geometry, forces,
    and their unit declarations are retained.

    Uses ``ConFrame.set_energy`` when available (readcon>=0.7) so metadata stays
    schema-valid; falls back to reconstructing the frame with string metadata.
    """
    readcon = _readcon()
    ConFrame = readcon.ConFrame
    metadata = deepcopy(frame.metadata)
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
    value = float(energy) * _energy_factor("eV", _energy_unit(frame))
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
