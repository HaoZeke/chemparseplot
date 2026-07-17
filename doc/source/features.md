# Features

::::{grid} 1 2 2 4
:gutter: 2

:::{grid-item-card} Parse
:class-card: sd-shadow-sm

Extract structured tables and ASE structures from engine outputs.
:::

:::{grid-item-card} Units
:class-card: sd-shadow-sm

`pint` quantities end-to-end — no silent eV/kcal mix-ups.
:::

:::{grid-item-card} Surfaces
:class-card: sd-shadow-sm

Gradient-enhanced GPs via `rgpycrumbs.surfaces`, optional `SurfaceFitConfig`.
:::

:::{grid-item-card} CLI suite
:class-card: sd-shadow-sm

Call from `rgpycrumbs eon plt-*` with TOML `--config` for dense option sets.
:::
::::

## Data flow

```{mermaid}
flowchart LR
  OUT[Engine outputs] --> PARSE[chemparseplot.parse]
  PARSE --> COORDS[RMSD / energy arrays]
  COORDS --> FIT{SurfaceFitConfig auto_thin?}
  FIT -->|default off| FULL[Full cloud fit]
  FIT -->|opt-in| THIN[Even subsample + endpoints]
  FULL --> GP[rgpycrumbs.surfaces]
  THIN --> GP
  GP --> FIG[Publication figure]
```

```{important}
`auto_thin` defaults to **false**. Enable via `SurfaceFitConfig` or the
matching keys in rgpycrumbs plot TOML when dense force-eval movies make
`grad_imq` non-finite.
```

- **Parsing** computational chemistry output files into structured data
- **Plotting** with [scientific color maps](https://www.fabiocrameri.ch/colourmaps/)
  (camera-ready)
- **Unit preserving** throughout via `pint`
- **Computation** delegated to [`rgpycrumbs`](https://github.com/HaoZeke/rgpycrumbs)
  for surface fitting, interpolation, and structure analysis
- **Concurrent** RMSD landscape calculations via `ThreadPoolExecutor`

## Supported Engines


### ORCA (5.x)

- Geometry scan (`OPT`) energy profiles
- Nudged elastic band (`NEB`) path visualization over the linearized reaction
  coordinate

### eOn

- Saddle search parsing (Dimer, GPRD, LBFGS methods) with status tracking
- NEB path energy profiles with RMSD landscape projections

### Sella

- Saddle point optimization result parsing

### ChemGP

GP-based optimization visualization from HDF5 output (`chemparseplot.plot.chemgp`):

- **1D plotnine charts**: convergence curves, RFF quality sweeps, NEB energy
  profiles, FPS subset scatter (PCA projection)
- **2D matplotlib surfaces**: PES contour with NEB path overlay, GP surrogate
  progression panels, MAP-NLL landscape, variance overlay with hatching,
  trust region illustration, hyperparameter sensitivity grid

### Trajectory Formats

- HDF5 trajectories from ChemGP output (pre-computed forces and reaction
  coordinates)
- Generic ASE-readable formats (extxyz, `.traj`) for NEB analysis with
  tangent force computation and RMSD landscape coordinates

- **eOn ConFrame series** (`parse.eon.frame_series`): energies and trajectory DTOs from stamped frames for object-plot adapters.
