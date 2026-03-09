---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# ChemGP 1D Plots (Convergence, RFF, Profile, FPS)

This tutorial produces four plotnine-based figures from ChemGP HDF5 data:
a convergence curve, an RFF quality comparison, an energy profile, and
an FPS scatter plot.

```{code-cell} ipython3
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from chemparseplot.plot.chemgp import (
    plot_convergence_curve,
    plot_rff_quality,
    plot_energy_profile,
    plot_fps_projection,
)
```

## Load sample data

The sample HDF5 file contains a `table/convergence` group with columns
`oracle_calls`, `max_fatom`, and `method` for three NEB methods.

```{code-cell} ipython3
DATA = Path("data")

# --- Convergence data ---
with h5py.File(DATA / "sample_convergence.h5") as f:
    tbl = f["table/convergence"]
    conv_df = pd.DataFrame({
        "oracle_calls": tbl["oracle_calls"][:],
        "max_fatom": tbl["max_fatom"][:],
        "method": [m.decode() if isinstance(m, bytes) else m for m in tbl["method"][:]],
    })
    conv_tol = float(f.attrs["conv_tol"])

conv_df.head()
```

## Convergence curve

`plot_convergence_curve` draws oracle calls on the x-axis and a force metric
on the y-axis, with log-scale by default. Pass `conv_tol` to add a
convergence threshold line.

```{code-cell} ipython3
fig = plot_convergence_curve(conv_df, conv_tol=conv_tol)
fig
```

### Per-method thresholds

Pass a dict to `conv_tol` to draw per-method dashed lines in matching colors:

```{code-cell} ipython3
fig = plot_convergence_curve(
    conv_df,
    conv_tol={"GP-NEB": 0.5, "AIE": 0.3, "OIE": 0.3},
)
fig
```

## RFF quality

`plot_rff_quality` takes a DataFrame with `d_rff`, `energy_mae`, and
`gradient_mae` columns, plus exact GP baselines. Here we generate synthetic
sweep data.

```{code-cell} ipython3
# Synthetic RFF sweep: MAE decreases as D_rff grows
rng = np.random.default_rng(7)
d_vals = [50, 100, 200, 300, 500]
rff_df = pd.DataFrame({
    "d_rff": d_vals,
    "energy_mae": [0.5, 0.25, 0.12, 0.08, 0.06],
    "gradient_mae": [1.2, 0.6, 0.3, 0.18, 0.12],
})

fig = plot_rff_quality(rff_df, exact_e_mae=0.04, exact_g_mae=0.08)
fig
```

## Energy profile

`plot_energy_profile` draws NEB image energies. The sample data below has
two methods with 7 images each.

```{code-cell} ipython3
n_img = 7
idx = list(range(n_img))
profile_df = pd.DataFrame({
    "image": idx * 2,
    "energy": [0.0, 0.2, 0.5, 0.8, 0.5, 0.2, 0.0,
               0.0, 0.15, 0.4, 0.75, 0.4, 0.15, 0.0],
    "method": ["NEB"] * n_img + ["GP-NEB"] * n_img,
})

fig = plot_energy_profile(profile_df)
fig
```

## FPS projection

`plot_fps_projection` shows PCA coordinates of FPS-selected vs pruned points.

```{code-cell} ipython3
rng = np.random.default_rng(42)
sel_pc1 = rng.normal(0, 1, 15)
sel_pc2 = rng.normal(0, 1, 15)
prn_pc1 = rng.normal(0, 1.5, 40)
prn_pc2 = rng.normal(0, 1.5, 40)

fig = plot_fps_projection(sel_pc1, sel_pc2, prn_pc1, prn_pc2)
fig
```

## Next steps

- [2D surface plots tutorial](surfaces.md) covers contour, GP progression, NLL landscape, variance overlay, trust region, and sensitivity
- [rgpycrumbs plt-gp CLI](https://rgpycrumbs.rgoswami.me/tools/chemgp/plt_gp.html) for batch figure generation from HDF5 files
