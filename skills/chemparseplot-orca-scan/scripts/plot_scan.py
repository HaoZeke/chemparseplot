"""Draw an ORCA bond scan with chemparseplot.plot.geomscan.plot_energy_paths.

    python scripts/plot_scan.py [ORCA_OUTPUT [PNG]]

Needs matplotlib, cmcrameri, scipy and rgpycrumbs next to chemparseplot.
The spline behind plot_energy_paths wants increasing distances, and ORCA
scans often run from long to short, so the points are sorted first.
Without PNG the figure is drawn in a temporary directory, checked to be
a PNG file and removed.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from scan_table import TUTORIAL_RUN, read_output, scan

from chemparseplot.plot.geomscan import plot_energy_paths
from chemparseplot.plot.structs import EnergyPath

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def draw(source: Path, out: Path) -> None:
    distance, energy, _unit = scan(read_output(source))
    order = np.argsort(distance.magnitude)
    path_data = EnergyPath("scan", distance[order], energy[order])
    plotter = plot_energy_paths([path_data], {"distance": "angstrom", "energy": "eV"})
    plotter.fig.savefig(out)
    if out.read_bytes()[:8] != PNG_SIGNATURE:
        sys.exit(f"{out} is not a PNG file")
    axes = f"{plotter.ax.get_xlabel()} against {plotter.ax.get_ylabel()}"
    print(f"wrote {out.name}: {axes}")


def main(argv: list[str]) -> int:
    source = Path(argv[1]) if len(argv) > 1 else TUTORIAL_RUN
    if len(argv) > 2:
        draw(source, Path(argv[2]))
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        draw(source, Path(tmp) / "scan.png")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
