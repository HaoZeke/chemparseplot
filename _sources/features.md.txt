# Features

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

### Trajectory Formats

- HDF5 trajectories from ChemGP output (pre-computed forces and reaction
  coordinates)
- Generic ASE-readable formats (extxyz, `.traj`) for NEB analysis with
  tangent force computation and RMSD landscape coordinates
