# Quickstart

This guide walks through installing `chemparseplot` and running a minimal
parsing and plotting example.

## Installation

The simplest route is via `pip`:

```bash
pip install chemparseplot
```

For plotting support, add the `plot` extra:

```bash
pip install "chemparseplot[plot]"
```

Or install everything at once:

```bash
pip install "chemparseplot[all]"
```

This pulls in [`rgpycrumbs`](https://github.com/HaoZeke/rgpycrumbs)
automatically, which provides the computational backend (surface fitting,
interpolation, structure analysis).

### Development install

```bash
git clone https://github.com/HaoZeke/chemparseplot
cd chemparseplot
uv sync --all-extras
```

## Parsing an ORCA geometry scan

`chemparseplot` ships parsers for several computational chemistry codes.
Here is a minimal example extracting energies from an ORCA geometry scan
output:

```python
from chemparseplot.parse.orca import geomscan

# Read the ORCA output (plain text)
with open("scan_output.out") as f:
    orca_text = f.read()

# Extract actual energies with units attached
distance, energy = geomscan.extract_energy_data(orca_text, "Actual")

# Units are pint Quantities
print(distance.units)  # bohr
print(energy.units)    # hartree

# Convert freely
print(distance.to("angstrom"))
```

## Plotting with unit awareness

The `TwoDimPlot` class handles unit-aware axes and scientific colormaps:

```python
from chemparseplot.plot.structs import TwoDimPlot, EnergyPath

plot = TwoDimPlot()
plot.set_units("angstrom", "hartree")

path = EnergyPath("H2 scan", distance, energy)
plot.add_data(path)
plot.show_plot("H2 Bond Length Scan")
```

Changing units redraws the axes automatically:

```python
plot.set_units("bohr", "electron_volt")
plot.fig  # updated figure
```

## Parsing eOn saddle searches

```python
from chemparseplot.parse.eon.saddle_search import parse_eon_saddle

result = parse_eon_saddle(
    eresp="results_saddle.dat",
    rloc="saddle/"
)
print(result.status)     # EONSaddleStatus
print(result.energy)     # saddle point energy
print(result.method)     # Dimer, GPRD, or LBFGS
```

## Next steps

- Browse the [tutorials](tutorials/index) for full worked examples
- See the [features](features) page for all supported engines
- Check the [API reference](apidocs/index) for detailed module documentation
