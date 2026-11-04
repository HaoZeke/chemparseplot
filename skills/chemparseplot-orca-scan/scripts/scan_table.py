"""Energies of an ORCA relaxed surface scan, with the length unit checked.

    python scripts/scan_table.py [ORCA_OUTPUT]

ORCA_OUTPUT is the text output of a bond scan, plain or gzipped; the
default is the tutorial run in doc/source/tutorials/orca/data/. The
script reads the 'Actual Energy' surface with chemparseplot, works out
the unit of the first column from the geometry ORCA printed at step 1,
and prints the scan relative to its lowest point. It exits 1 when the
surface table is missing or its unit cannot be told.
"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

import numpy as np

from chemparseplot.parse.orca import geomscan
from chemparseplot.units import Q_

ROOT = Path(__file__).resolve().parents[3]
TUTORIAL_RUN = ROOT / "doc/source/tutorials/orca/data/scanuhf.gz"
BOHR_IN_ANGSTROM = Q_(1, "bohr").to("angstrom").magnitude
# Coordinates are printed with six decimals, so a bond rebuilt from them
# is good to about 1e-6 angstrom; 1e-3 still tells 1 bohr from 1 angstrom.
LENGTH_TOLERANCE = 1e-3


def read_output(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return fh.read()
    return path.read_text()


def first_step_bond(text: str) -> tuple[tuple[int, int], float] | None:
    """Scanned atom pair and its length in angstrom at scan step 1."""
    bond = re.search(r"Bond \(\s*(\d+),\s*(\d+)\):\s+range=", text)
    step = re.search(r"RELAXED SURFACE SCAN STEP\s+1\s", text)
    if bond is None or step is None:
        return None
    block = re.search(
        r"CARTESIAN COORDINATES \(ANGSTROEM\)\n-+\n((?:\s*\w+(?:\s+-?\d+\.\d+){3}\n)+)",
        text[step.end() :],
    )
    if block is None:
        return None
    xyz = np.array([[float(v) for v in row.split()[1:]] for row in block[1].splitlines()])
    i, j = int(bond[1]), int(bond[2])
    return (i, j), float(np.linalg.norm(xyz[i] - xyz[j]))


def scan(text: str) -> tuple[Q_, Q_, str]:
    """Distances and energies of the scan, the distances in their real unit."""
    distance, energy = geomscan.extract_energy_data(text, "Actual")
    if len(distance) == 0:
        sys.exit("no \"The Calculated Surface using the 'Actual Energy'\" table found")
    found = first_step_bond(text)
    if found is None:
        sys.exit("no bond scan with a step-1 geometry found; check the unit by hand")
    (i, j), length = found
    first = float(distance.magnitude[0])
    if abs(length - first) < LENGTH_TOLERANCE:
        unit = "angstrom"
    elif abs(length - first * BOHR_IN_ANGSTROM) < LENGTH_TOLERANCE:
        unit = "bohr"
    else:
        sys.exit(f"step 1 has atoms {i},{j} {length:.6f} A apart; the table says {first}")
    print(f"step 1: atoms {i} and {j} are {length:.6f} angstrom apart")
    print(f"table, first row: {first:.8f}, so the column is in {unit}")
    print(f"chemparseplot labels it {distance.units}")
    return Q_(distance.magnitude, unit).to("angstrom"), energy, unit


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else TUTORIAL_RUN
    text = read_output(path)
    distance, energy, _unit = scan(text)
    _, scf_energy = geomscan.extract_energy_data(text, "SCF")
    same = np.array_equal(scf_energy.magnitude, energy.magnitude)
    scf = "the same" if same else "different"
    print(f"{len(distance)} points; the SCF table is {scf}")
    lowest = int(np.argmin(energy.magnitude))
    relative = (energy - energy[lowest]).to("eV").magnitude
    print(f"{'step':>4} {'r / angstrom':>13} {'E / hartree':>14} {'E - Emin / eV':>14}")
    for k, (r, e, de) in enumerate(
        zip(distance.magnitude, energy.magnitude, relative, strict=True), 1
    ):
        mark = "  <- lowest" if k - 1 == lowest else ""
        print(f"{k:>4} {r:>13.6f} {e:>14.8f} {de:>14.6f}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
