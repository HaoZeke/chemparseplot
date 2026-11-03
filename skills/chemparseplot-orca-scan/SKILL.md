---
name: chemparseplot-orca-scan
description: Read an ORCA relaxed surface scan (a `%geom Scan` over a bond) with chemparseplot and plot its energy curve. Covers `chemparseplot.parse.orca.geomscan.extract_energy_data`, the two surface tables ORCA prints, the bohr label it puts on distances ORCA printed in ångström, the table name it matches case by case, and `chemparseplot.plot.geomscan.plot_energy_paths`, which needs sorted distances and packages no chemparseplot extra installs. Use when asked for the energies, the minimum or a figure of an ORCA scan, or when a scan comes out 1.89 times too short.
license: MIT
compatibility: scripts/scan_table.py needs chemparseplot alone (numpy and pint). scripts/plot_scan.py also needs matplotlib and cmcrameri (the plot extra), rgpycrumbs and scipy. Python 3.11 or newer. Checked on Linux against ORCA 5.0.4 output.
---

# ORCA relaxed surface scans with chemparseplot

## What ORCA writes

The ORCA tutorial of chemparseplot,
`doc/source/tutorials/orca/geomscan.md`, scans the bond of H2 with

```text
!OPT UHF def2-SVP
%geom Scan
 B 0 1 = 7.5589039543, 0.2116708996, 33
 end
end
*xyzfile 0 1 h2_base.xyz
```

`B 0 1` is the bond between atoms 0 and 1 (ORCA counts from 0), stepped
from the first value to the second in 33 constrained optimisations.
Near the end of the output ORCA prints two tables with one row per
step, the coordinate and then the energy in hartree:

```text
The Calculated Surface using the 'Actual Energy'
   7.55890395  -0.74239862
   7.32930292  -0.74349939
...
The Calculated Surface using the SCF energy
...
```

The output of that run is in the repository as
`doc/source/tutorials/orca/data/scanuhf.gz`.

## Read it

`extract_energy_data` takes the text of the output, not a path, and
the table name, `"Actual"` or `"SCF"`. It returns two pint quantities:

```python
import gzip

from chemparseplot.parse.orca import geomscan

with gzip.open("../../doc/source/tutorials/orca/data/scanuhf.gz", "rt") as fh:
    text = fh.read()
distance, energy = geomscan.extract_energy_data(text, "Actual")
print(len(distance), distance.units, energy.units)
assert len(distance) == 33
```

which prints `33 bohr hartree`. Run from this directory in a
chemparseplot checkout, the block reads the tutorial run.
`chemparseplot.api.extract_orca_geomscan_energy(text)` is the same
function with `"Actual Energy"` as the default table.

## The length unit

The distances come back labelled bohr, whatever ORCA printed. ORCA
printed ångström here: at step 1 the Cartesian coordinates in the
output put the two atoms 7.558904 Å apart, and the first row of the
table is 7.55890395. So `distance.to("angstrom")` gives 4.0 Å where the
bond was 7.56 Å, 1.89 times too short. The tutorial does exactly that
conversion, and the comment in its input, "Converted from Angstrom to
Bohr", is the same mistake made on the way in: ORCA read the numbers as
ångström and scanned from 7.56 Å to 0.21 Å.

Relabel the numbers instead of converting them:

```python norun
from chemparseplot.units import Q_

distance = Q_(distance.magnitude, "angstrom")
```

`scripts/scan_table.py` does not take the label on trust. It rebuilds
the scanned bond from the geometry of step 1 and keeps the unit that
matches the first row, to 1e-3 Å, and it stops when neither ångström
nor bohr matches. Only bond scans were checked. The same label is put
on the first column of any scan, so for an angle or a dihedral look at
the output by hand.

## Which table

- The name is pasted into a regular expression and matched as
  written: `"actual"`, `"ACTUAL"` and `"scf"` find nothing. A name that
  matches nothing returns two empty arrays, with no error or warning,
  so check the length.
- `"Actual"` is the total energy the optimiser used and `"SCF"` the SCF
  energy alone. In the tutorial run, plain UHF, the two tables are the
  same; `scripts/scan_table.py` compares them.
- Only one-coordinate scans were checked. The pattern reads two
  numbers per row.

## Plot it

```python norun
import numpy as np

from chemparseplot.plot.geomscan import plot_energy_paths
from chemparseplot.plot.structs import EnergyPath

order = np.argsort(distance.magnitude)
plotter = plot_energy_paths(
    [EnergyPath("H2 scan", distance[order], energy[order])],
    {"distance": "angstrom", "energy": "eV"},
)
plotter.fig.savefig("scan.png")
```

`plot_energy_paths` takes a list of `EnergyPath(label, distance,
energy)`, converts each to the units given, draws a spline through the
points and returns a `BasePlotter` with `.fig` and `.ax`.

- Sort first. ORCA scans run in the order of the input, here from long
  to short, and the spline then fails with `ValueError: Error on input
  data`.
- It imports `rgpycrumbs.interpolation` when called, and that needs
  scipy. Neither is in any chemparseplot extra, `[all]` included:

  ```bash
  pip install "chemparseplot[plot]" rgpycrumbs "scipy>=1.11"
  ```

  Without rgpycrumbs the call stops with `ModuleNotFoundError`; without
  scipy, rgpycrumbs raises `ImportError: Module 'scipy.interpolate' is
  required`.
- `BasePlotter` calls `plt.style.use("bmh")`, which changes the
  matplotlib style for every later figure of the session.
- `TwoDimPlot`, which the tutorial uses next, takes `XYData(label, x,
  y)`, not `EnergyPath`; given an `EnergyPath` it raises
  `AttributeError: 'EnergyPath' object has no attribute 'x'`.

## Check

```bash
python scripts/scan_table.py [ORCA_OUTPUT]
python scripts/plot_scan.py [ORCA_OUTPUT [PNG]]
```

Both read the tutorial run when no file is given, plain or gzipped.
`scripts/scan_table.py` prints how it settled the unit, then the scan
in ångström, hartree and eV above the lowest point:

```text
step 1: atoms 1 and 0 are 7.558904 angstrom apart
table, first row: 7.55890395, so the column is in angstrom
chemparseplot labels it bohr
33 points; the SCF table is the same
step  r / angstrom    E / hartree  E - Emin / eV
   1      7.558904    -0.74239862      10.384969
...
  31      0.670873    -1.12403918       0.000000  <- lowest
```

`scripts/plot_scan.py` sorts the points, draws them with
`plot_energy_paths` and checks that the file is a PNG image. Without
PNG it draws into a temporary directory, which is how CI runs it.
