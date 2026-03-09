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

# ChemGP 2D Plots (Surfaces, GP Progression, NLL, Variance, Trust, Sensitivity)

This tutorial produces six matplotlib-based figures from ChemGP HDF5 data:
a PES contour, GP surrogate progression, NLL landscape, variance overlay,
trust region illustration, and hyperparameter sensitivity grid.

```{code-cell} ipython3
import h5py
import numpy as np
from pathlib import Path

from chemparseplot.plot.chemgp import (
    plot_surface_contour,
    plot_gp_progression,
    plot_nll_landscape,
    plot_variance_overlay,
    plot_trust_region,
    plot_hyperparameter_sensitivity,
)
```

## Load sample data

Two HDF5 fixtures provide the grids and metadata:

- `sample_surface.h5`: 40x40 double-well PES, GP progression grids, NEB path,
  stationary points, training points, variance grid
- `sample_nll.h5`: 25x25 NLL grid with gradient norm overlay and MAP optimum

```{code-cell} ipython3
DATA = Path("data")

# --- Surface data ---
with h5py.File(DATA / "sample_surface.h5") as f:
    eg = f["grids/energy"]
    x_range = np.linspace(*eg.attrs["x_range"], eg.attrs["x_length"])
    y_range = np.linspace(*eg.attrs["y_range"], eg.attrs["y_length"])
    energy = eg[:]
    X, Y = np.meshgrid(x_range, y_range)

    # Variance grid
    variance = f["grids/variance"][:]

    # NEB path
    path_x = f["paths/neb/x"][:]
    path_y = f["paths/neb/y"][:]

    # Stationary points
    min_x = f["points/minima/x"][:]
    min_y = f["points/minima/y"][:]
    sad_x = f["points/saddles/x"][:]
    sad_y = f["points/saddles/y"][:]

    # Training points
    train_x = f["points/training/x"][:]
    train_y = f["points/training/y"][:]

    # GP progression grids
    gp_grids = {}
    for n_train in [5, 10, 20]:
        gp_mean = f[f"grids/gp_mean_{n_train}"][:]
        tx = f[f"points/training_{n_train}/x"][:]
        ty = f[f"points/training_{n_train}/y"][:]
        gp_grids[n_train] = {"gp_mean": gp_mean, "train_x": tx, "train_y": ty}

print(f"Grid shape: {energy.shape}, {len(gp_grids)} GP snapshots loaded")
```

## PES contour with NEB path

`plot_surface_contour` draws filled contours with optional path and point
overlays. Pass `paths` as a dict of label -> (xs, ys) and `points` with
special keys `"minima"`, `"saddles"`, or `"endpoints"`.

```{code-cell} ipython3
fig = plot_surface_contour(
    X, Y, energy,
    paths={"NEB": (path_x, path_y)},
    points={
        "minima": (min_x, min_y),
        "saddles": (sad_x, sad_y),
    },
    clamp_lo=-1.0,
    clamp_hi=5.0,
    contour_step=0.5,
)
fig
```

## GP surrogate progression

`plot_gp_progression` shows how the GP approximation improves as more training
points are added. Each panel is a separate training size.

```{code-cell} ipython3
fig = plot_gp_progression(
    gp_grids,
    true_energy=energy,
    x_range=x_range,
    y_range=y_range,
    clamp_lo=-1.0,
    clamp_hi=5.0,
    n_cols=3,
    width=12.0,
    height=4.0,
)
fig
```

## NLL landscape

`plot_nll_landscape` visualizes the MAP negative log-likelihood in
hyperparameter space. The colormap is reversed (warm = low NLL = good fit).
An optional gradient norm overlay adds dashed contours.

```{code-cell} ipython3
with h5py.File(DATA / "sample_nll.h5") as f:
    ng = f["grids/nll"]
    nll_x = np.linspace(*ng.attrs["x_range"], ng.attrs["x_length"])
    nll_y = np.linspace(*ng.attrs["y_range"], ng.attrs["y_length"])
    NX, NY = np.meshgrid(nll_x, nll_y)
    nll = ng[:]
    grad_norm = f["grids/gradient_norm"][:]
    optimum = (float(f.attrs["log_sigma2"]), float(f.attrs["log_theta"]))

fig = plot_nll_landscape(
    NX, NY, nll,
    grid_grad_norm=grad_norm,
    optimum=optimum,
)
fig
```

## Variance overlay

`plot_variance_overlay` draws the energy surface with diagonal hatching over
high-variance regions and a magenta boundary contour at the 75th percentile
of the variance.

```{code-cell} ipython3
fig = plot_variance_overlay(
    X, Y, energy, variance,
    train_points=(train_x, train_y),
    stationary={
        "min0": (float(min_x[0]), float(min_y[0])),
        "min1": (float(min_x[1]), float(min_y[1])),
        "saddle0": (float(sad_x[0]), float(sad_y[0])),
    },
    clamp_lo=-1.0,
    clamp_hi=5.0,
)
fig
```

## Trust region

`plot_trust_region` illustrates GP prediction quality inside and outside a
trust region. Points inside the trust boundary (magenta dotted lines) have
low uncertainty; a hypothetical bad step outside shows where the GP diverges
from the oracle.

```{code-cell} ipython3
# 1D slice through the surface at y=0
x_slice = x_range
e_true = (x_slice**2 - 1)**2  # double well at y=0
e_pred = e_true + 0.3 * np.sin(3 * x_slice)  # GP with some error
e_std = 0.1 + 0.5 * np.abs(x_slice)  # uncertainty grows away from center
in_trust = (np.abs(x_slice) < 1.2).astype(float)
slice_train_x = np.array([-0.8, -0.3, 0.0, 0.3, 0.8])

fig = plot_trust_region(
    x_slice, e_true, e_pred, e_std, in_trust,
    train_x=slice_train_x,
)
fig
```

## Hyperparameter sensitivity

`plot_hyperparameter_sensitivity` draws a 3x3 grid. Columns are lengthscale
values, rows are signal variance values. Each panel shows the GP mean,
confidence band, and true surface along a 1D slice.

```{code-cell} ipython3
# Build 9 panels: 3 lengthscales x 3 signal variances
panels = {}
ls_values = [0.05, 0.3, 2.0]
sv_values = [0.1, 1.0, 100.0]

for j, ls in enumerate(ls_values, 1):
    for i, sv in enumerate(sv_values, 1):
        # Simulate GP with varying hyperparameters
        pred = e_true * (1 + 0.1 * (3 - j))  # lengthscale affects smoothness
        std = np.full_like(x_slice, 0.05 * sv)  # variance scales uncertainty
        panels[f"gp_ls{j}_sv{i}"] = {"E_pred": pred, "E_std": std}

fig = plot_hyperparameter_sensitivity(x_slice, e_true, panels)
fig
```

## Next steps

- [1D plots tutorial](convergence.md) covers convergence curves, RFF quality, energy profiles, and FPS scatter
- [rgpycrumbs plt-gp CLI](https://rgpycrumbs.rgoswami.me/tools/chemgp/plt_gp.html) for batch generation from the command line
- [HDF5 schema reference](https://rgpycrumbs.rgoswami.me/tools/chemgp/hdf5_schema.html) for the expected data layout
