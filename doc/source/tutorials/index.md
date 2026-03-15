# Tutorial Introduction

Here we focus on standard workflows for `chemparseplot`.

## Available Tutorials

```{toctree}
:maxdepth: 2
:caption: Chemistry Engines

chemgp/index
orca/geomscan
eon/saddle
```

## Quick Start

### Parsing ORCA Output

```python
from chemparseplot.parse.orca import geomscan

# Extract geometry scan data
energy_data = geomscan.extract_energy_data(orca_output, "Actual")
```

### Parsing eOn Saddle Searches

```python
from chemparseplot.parse.eon.saddle_search import parse_eon_saddle
from rgpycrumbs.basetypes import SpinID

result = parse_eon_saddle(results_dir, SpinID(mol_id="test", spin=0))
print(f"Barrier: {result.barrier}")
```

### Plotting NEB Profiles

```python
from chemparseplot.plot.neb import plot_energy_path
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
rc = np.array([0.0, 1.0, 2.0, 3.0])
energy = np.array([-123.5, -123.4, -123.3, -123.5])
f_para = np.array([0.1, 0.05, -0.05, -0.1])

plot_energy_path(ax, rc, energy, f_para, "blue", alpha=0.8, zorder=10)
```
