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
| pint | Unit handling | ✓ |
| numpy | Numerical operations | ✓ |
| rgpycrumbs | Computational delegation | ✓ |
| matplotlib | Plotting backend | ✓ |
| polars | DataFrames for NEB data | ✓ |
| ase | Atoms object handling | ✓ |

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
