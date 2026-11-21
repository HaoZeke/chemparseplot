"""Read ORCA's formatted Hessian and Cartesian normal modes."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OrcaHessian:
    symbols: tuple[str, ...]
    masses_amu: np.ndarray
    positions_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray
    frequencies_cm1: np.ndarray
    cartesian_modes: np.ndarray


def _matrix(lines, size):
    dimensions = tuple(map(int, lines[0].split()))
    if dimensions not in {(size,), (size, size)}:
        raise ValueError("ORCA matrix dimensions disagree with atom count")
    matrix = np.full((size, size), np.nan)
    columns = []
    for line in lines[1:]:
        values = line.split()
        if all(value.isdigit() for value in values):
            columns = [int(value) for value in values]
            if len(set(columns)) != len(columns) or any(c >= size for c in columns):
                raise ValueError("Invalid ORCA column indices")
            continue
        row = int(values[0])
        if not 0 <= row < size or len(values) != len(columns) + 1 or not columns:
            raise ValueError("Invalid ORCA matrix row")
        if not np.isnan(matrix[row, columns]).all():
            raise ValueError("Duplicate ORCA matrix entries")
        matrix[row, columns] = [float(v.replace("D", "E")) for v in values[1:]]
    if not np.isfinite(matrix).all():
        raise ValueError("Incomplete or nonfinite ORCA matrix")
    return matrix


def read_orca_hessian(path: Path) -> OrcaHessian:
    """Return ordered vibrational frequencies and modes (one Cartesian mode per row)."""
    sections = {}
    lines = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if line.startswith("$"):
            lines = []
            sections[line] = lines
        elif lines is not None and line and not line.startswith("#"):
            lines.append(line)
    atom_lines = sections["$atoms"]
    n = int(atom_lines[0])
    rows = [line.split() for line in atom_lines[1:]]
    if n <= 0 or len(rows) != n or any(len(row) != 5 for row in rows):
        raise ValueError("Invalid ORCA atom table")
    atoms = np.array([row[1:] for row in rows], dtype=float)
    if not np.isfinite(atoms).all() or np.any(atoms[:, 0] <= 0):
        raise ValueError("Invalid ORCA masses or coordinates")
    size = 3 * n
    freq = sections["$vibrational_frequencies"]
    frequencies = [line.split() for line in freq[1:]]
    if int(freq[0]) != size or [int(row[0]) for row in frequencies] != list(range(size)):
        raise ValueError("Incomplete or unordered ORCA frequencies")
    values = np.array([float(row[1]) for row in frequencies])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite ORCA frequency")
    return OrcaHessian(tuple(row[0] for row in rows), atoms[:, 0], atoms[:, 1:],
                       _matrix(sections["$hessian"], size), values,
                       _matrix(sections["$normal_modes"], size).T.reshape(size, n, 3))
