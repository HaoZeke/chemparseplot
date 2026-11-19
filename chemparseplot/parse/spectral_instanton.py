"""Read the indexed CON frames and CPMD sidecars of a spectral oracle batch."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pint import Quantity

from chemparseplot.parse.con.io import frame_positions_angstrom, read_first_frame
from chemparseplot.units import Q_

_CARTESIAN_DIMENSIONS = 3


@dataclass(frozen=True)
class OracleBatch:
    """Trajectory and energies/forces in numeric node order."""

    positions: Quantity
    energy: Quantity
    forces: Quantity
    masses: Quantity
    mobile: np.ndarray
    cell: Quantity
    angles: Quantity
    atom_ids: tuple[int, ...]


def _indexed(directory: Path, pattern: str) -> dict[int, Path]:
    paths = {}
    for path in directory.iterdir():
        if match := re.fullmatch(pattern, path.name):
            index = int(match[1])
            if index in paths:
                msg = f"duplicate node {index} in {directory}"
                raise ValueError(msg)
            paths[index] = path
    return paths


def read_oracle_batch(directory: str | Path) -> OracleBatch:
    """Read one complete spectral-instanton Slurm oracle batch.

    The spectral host writes one CON geometry and one scalar eV energy followed
    by Cartesian eV/Å force rows for each node. This reader checks node and
    atom order, masses, and fixed flags across frames. CON input uses readcon.

    The caller supplies the sampling grid, temperature, and energy reference:
    the sidecars describe surface evaluations and may represent any trial.
    """
    directory = Path(directory)
    geometries = _indexed(directory, r"geom-(\d+)\.con")
    results = _indexed(directory, r"result-(\d+)\.txt")
    if not geometries:
        msg = f"no oracle geometry files in {directory}"
        raise ValueError(msg)
    expected = set(range(len(geometries)))
    for name, files in (("geometry", geometries), ("result", results)):
        if files.keys() != expected:
            msg = (
                f"{name} nodes differ from the geometry sequence: "
                f"missing {sorted(expected - files.keys())}, "
                f"unexpected {sorted(files.keys() - expected)}"
            )
            raise ValueError(msg)
    positions, energy, forces = [], [], []
    identity = None
    cell = angles = None
    for index in range(len(geometries)):
        frame = read_first_frame(geometries[index])
        atoms = tuple((a.atom_id, a.symbol, a.mass, tuple(a.fixed)) for a in frame.atoms)
        frame_cell = Q_(
            frame.cell, frame.metadata.get("units", {}).get("length", "angstrom")
        )
        frame_angles = np.asarray(frame.angles)
        if identity is None:
            identity = atoms
            cell = frame_cell.to("angstrom")
            angles = frame_angles
        elif identity != atoms:
            msg = f"atom identities, masses, or fixed masks differ at node {index}"
            raise ValueError(msg)
        if (
            cell is None
            or not np.array_equal(frame_cell.to("angstrom").magnitude, cell.magnitude)
            or not np.array_equal(frame_angles, angles)
        ):
            msg = f"cell differs at node {index}"
            raise ValueError(msg)
        rows = results[index].read_text(encoding="utf-8").splitlines()
        if len(rows) != len(atoms) + 1 or any(
            len(row.split()) != _CARTESIAN_DIMENSIONS for row in rows[1:]
        ):
            msg = f"invalid energy/force record at node {index}"
            raise ValueError(msg)
        e = float(rows[0])
        f = np.asarray([row.split() for row in rows[1:]], dtype=float)
        xyz = frame_positions_angstrom(frame)
        if not np.isfinite(e) or not np.isfinite(f).all() or not np.isfinite(xyz).all():
            msg = f"nonfinite energy, forces or coordinates at node {index}"
            raise ValueError(msg)
        positions.append(xyz)
        energy.append(e)
        forces.append(f)
    if identity is None or cell is None or angles is None:
        msg = "empty spectral oracle batch"
        raise ValueError(msg)
    if any(any(atom[3]) and not all(atom[3]) for atom in identity):
        msg = "spectral oracle batches require whole-atom fixed masks"
        raise ValueError(msg)
    return OracleBatch(
        Q_(np.asarray(positions), "angstrom"),
        Q_(np.asarray(energy), "eV"),
        Q_(np.asarray(forces), "eV / angstrom"),
        Q_(np.asarray([atom[2] for atom in identity]), "amu"),
        np.asarray([not any(atom[3]) for atom in identity]),
        cell,
        Q_(angles, "degree"),
        tuple(atom[0] for atom in identity),
    )
