# Architecture and Design

This document explains the design decisions and architecture of `chemparseplot`.

## Overview

`chemparseplot` is a parsing and plotting library for computational chemistry outputs. It sits between raw quantum chemistry code outputs and publication-quality visualizations.

## Design Philosophy

### 1. Unit-Aware Throughout

All physical quantities use [`pint`](https://pint.readthedocs.io/) for automatic unit conversion and dimensional analysis:

```python
from chemparseplot.units import Q_

energy = Q_(-123.456, "hartree")
print(energy.to("kcal/mol"))  # Automatic conversion
```

**Why pint?** Manual unit tracking is error-prone. Pint ensures:
- Dimensional consistency checks at runtime
- Automatic conversion between common units (hartree, eV, kcal/mol)
- Clear error messages when incompatible units are combined

### 2. Computation Delegation

Heavy computational tasks are delegated to [`rgpycrumbs`](https://github.com/HaoZeke/rgpycrumbs):

```python
# chemparseplot parses
from chemparseplot.parse.orca import geomscan
energy_data = geomscan.extract_energy_data(orca_output)

# rgpycrumbs computes
from rgpycrumbs.surfaces import get_surface_model
model = get_surface_model("tps")(x_data, energy_data)
```

**Why delegation?**
- Single responsibility: chemparseplot focuses on I/O and visualization
- Avoids dependency bloat: JAX, SciPy live in rgpycrumbs
- Reusability: rgpycrumbs can be used independently

### 3. Parser-Plotter Separation

The library is organized into two main subpackages:

```
chemparseplot/
├── parse/     # Extract structured data from outputs
│   ├── orca/  # ORCA parsers
│   ├── eon/   # eOn parsers
│   └── ...
└── plot/      # Create visualizations
    ├── neb.py       # NEB plotting
    ├── geomscan.py  # Geometry scan plotting
    └── ...
```

**Why separation?**
- Parsers can be used without plotting (e.g., data analysis pipelines)
- Plotters can accept data from multiple sources (not just parsers)
- Easier testing: parsers and plotters have different test requirements

## Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
| ORCA/eOn/   | --> | chemparseplot| --> | matplotlib/ |
| Sella output|     |   parsers    |     |  plotnine   |
└─────────────┘     └──────────────┘     └─────────────┘
                           |
                           v
                    ┌──────────────┐
                    | rgpycrumbs   |
                    | (optional)   |
                    └──────────────┘
```

1. **Input**: Raw text output from quantum chemistry codes
2. **Parsing**: Regular expressions and structured extraction → `Q_` quantities with units
3. **Optional computation**: Surface fitting, interpolation via rgpycrumbs
4. **Plotting**: matplotlib/plotnine with scientific color maps (Crameri)

## Supported Engines

| Engine | Version | Parsers | Plotters |
|--------|---------|---------|----------|
| ORCA | 5.x | ✓ geomscan, NEB | ✓ |
| eOn | 2.x | ✓ saddle, NEB | ✓ |
| Sella | 2.x | ✓ saddle | - |
| ChemGP | - | ✓ HDF5 | ✓ |

## Package Structure

```
chemparseplot/
├── __init__.py
├── units.py           # pint unit registry
├── parse/
│   ├── __init__.py
│   ├── converter.py   # Unit conversion helpers
│   ├── file_.py       # File discovery utilities
│   ├── neb_utils.py   # Common NEB parsing utilities
│   ├── patterns.py    # Regular expression patterns
│   ├── grammar/          # optional parsimonious track (XYZ, ORCA text)
│   │   ├── xyz.py
│   │   └── orca_text.py
│   ├── orca/
│   │   ├── __init__.py
│   │   ├── geomscan.py
│   │   └── neb/
│   │       └── interp.py
│   ├── eon/
│   │   ├── __init__.py
│   │   ├── neb.py
│   │   ├── saddle_search.py
│   │   ├── gprd.py
│   │   └── minimization.py
│   ├── sella/
│   │   └── saddle_search.py
│   └── trajectory/
│       ├── __init__.py
│       ├── hdf5.py
│       └── neb.py
└── plot/
    ├── __init__.py
    ├── theme.py       # Scientific color maps
    ├── structs.py     # Structure rendering
    ├── geomscan.py
    ├── neb.py
    └── chemgp.py
```

## Key Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| pint | Unit handling | ✓ hard |
| numpy | Numerical operations | ✓ hard |
| matplotlib | Plotting backend | optional (`[plot]`) |
| polars | DataFrames for NEB / trajectory tables | optional (`[neb]`) |
| ase | Atoms object handling | optional (`[neb]`) |
| readcon (>=0.7) | CON/convel I/O and metadata-native energies | optional (`[neb]`) |
| rgpycrumbs | Interpolation, surfaces, RMSD alignment | optional (compute delegation; install alongside or via tests/pixi) |
| h5py | ChemGP / trajectory HDF5 | optional (`[neb]`) |
| parsimonious | Grammar/AST text parsers (XYZ, ORCA energy/coords) | optional (`[grammar]`) |



## ORCA backends (OPI proxy)

ORCA NEB and other structured products are exposed through **stable public
functions** under `chemparseplot.parse.orca` (for example `parse_orca_neb`).

| Backend | When | Consumer-facing? |
|---------|------|------------------|
| **OPI** (`opi` package / `chemparseplot[opi]`) | ORCA 6.1+ structured output | **No** — internal only |
| **legacy** (`.interp` / text) | Older ORCA or OPI missing | Via same public API (`backend="legacy"` or auto) |

Applications and **wailord** (batch shell) must call chemparseplot APIs, not
`import opi`. Text-heavy ORCA sections without a structured API are a separate
**grammar/AST** track inside chemparseplot (parsimonious-class parsers), not a
reason for a second ORCA SDK.

### Grammar track (`chemparseplot[grammar]`)

| Entry point | Role |
|-------------|------|
| `chemparseplot.parse.grammar.parse_xyz_text` / `parse_xyz_file` | XYZ frame grammar |
| `chemparseplot.parse.grammar.parse_orca_text_summary` | Final energies + last Cartesian (Å) |
| `chemparseplot.api.parse_xyz` / `parse_orca_final_energy` | Stable library shims |

Install: `pip install 'chemparseplot[grammar]'`. Without parsimonious these
APIs raise `ImportError` with an install hint (same pattern as OPI).

## Versioning

chemparseplot uses semantic versioning via `hatch-vcs`:
- **Major**: Breaking API changes
- **Minor**: New features (parsers, plotters)
- **Patch**: Bug fixes

Version is derived from git tags automatically.

## Related Projects

- **[rgpycrumbs](https://github.com/HaoZeke/rgpycrumbs)**: Core computational library
- **[pychum](https://github.com/HaoZeke/pychum)**: Input file generation
- **[eOn](https://eondocs.org/)**: Saddle point search code (parser target)
- **[ORCA](https://orcaforum.kofo.mpg.de/)**: Quantum chemistry code (parser target)
- **[readcon-core](https://github.com/lode-org/readcon-core)** (PyPI `readcon`): CON/convel codec used by eOn parsers
