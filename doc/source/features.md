# Features

- **Parsing** computational chemistry output files into structured data
- **Plotting** with [scientific color maps](https://www.fabiocrameri.ch/colourmaps/)
    - Camera ready
- **Unit preserving** throughout via `pint`
- **Computation** delegated to [`rgpycrumbs`](https://github.com/HaoZeke/rgpycrumbs)
  for surface fitting, interpolation, and structure analysis

## Supported Engines [WIP]

- ORCA (**5.x**)
    - Scanning energies over a degree of freedom (`OPT` scans)
    - Nudged elastic band (`NEB`) visualizations (over the "linearized" reaction
      coordinate)
- eOn
    - Saddle search parsing and visualization
    - NEB path energy profiles with surface models
