# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Load a CON or ASE trajectory for the public plot helpers.

``.con`` and ``.convel`` go through :mod:`chemparseplot.parse.eon.con_io`.
Other suffixes stay on ``ase.io.read``. This module does not open a corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chemparseplot.parse.eon.con_io import CON_SUFFIXES, is_con_path, read_con_frames

__all__ = ["ConTrajectory", "load_trajectory"]


@dataclass
class ConTrajectory:
    """One loaded trajectory.

    ``energies`` are per-frame totals in eV (NaN when the frame has none).
    ``table`` is a polars frame whose ``fmax`` column comes from ``frame.frc``
    when a forces section is present. ``frames`` are ``ConFrame`` objects for
    ``source == "con"`` and ASE ``Atoms`` otherwise.
    """

    path: Path
    source: str
    n_frames: int
    energies: list[float]
    table: Any
    frames: list[Any]


def _fmax_from_forces(forces: Any) -> float | None:
    if forces is None:
        return None
    array = np.asarray(forces, dtype=float)
    if array.size == 0:
        return None
    return float(np.max(np.linalg.norm(array, axis=1)))


def _energy_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def _ase_energy(atoms: Any) -> float:
    info = getattr(atoms, "info", None) or {}
    if "energy" in info:
        return _energy_or_nan(info["energy"])
    calc = getattr(atoms, "calc", None)
    if calc is None:
        return float("nan")
    try:
        return _energy_or_nan(atoms.get_potential_energy())
    except Exception:
        return float("nan")


def _ase_fmax(atoms: Any) -> float | None:
    if getattr(atoms, "calc", None) is None:
        return None
    try:
        forces = atoms.get_forces()
    except Exception:
        return None
    return _fmax_from_forces(forces)


def _table(energies: list[float], fmax: list[float | None]):
    import polars as pl

    return pl.DataFrame(
        {
            "frame": list(range(len(energies))),
            "energy": energies,
            "fmax": pl.Series("fmax", fmax, dtype=pl.Float64),
        }
    )


def _load_con(path: Path) -> ConTrajectory:
    frames = read_con_frames(path)
    energies = [_energy_or_nan(frame.energy) for frame in frames]
    fmax = [_fmax_from_forces(getattr(frame, "frc", None)) for frame in frames]
    return ConTrajectory(
        path=path,
        source="con",
        n_frames=len(frames),
        energies=energies,
        table=_table(energies, fmax),
        frames=frames,
    )


def _load_ase(path: Path) -> ConTrajectory:
    from ase.io import read as ase_read

    loaded = ase_read(str(path), index=":")
    frames = list(loaded) if isinstance(loaded, list) else [loaded]
    energies = [_ase_energy(atoms) for atoms in frames]
    fmax = [_ase_fmax(atoms) for atoms in frames]
    suffix = path.suffix.lower().lstrip(".")
    return ConTrajectory(
        path=path,
        source=suffix or "ase",
        n_frames=len(frames),
        energies=energies,
        table=_table(energies, fmax),
        frames=frames,
    )


def load_trajectory(path: str | Path) -> ConTrajectory:
    """Load *path* for plotting.

    ``.con`` and ``.convel`` use :func:`chemparseplot.parse.eon.con_io.read_con_frames`.
    Every other suffix uses ``ase.io.read``. ``fmax`` is the maximum per-atom
    force norm from ``frame.frc`` when that column exists.
    """
    file_path = Path(path)
    if is_con_path(file_path) or file_path.suffix.lower() in CON_SUFFIXES:
        return _load_con(file_path)
    return _load_ase(file_path)
