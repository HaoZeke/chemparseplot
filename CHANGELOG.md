# Changelog

<!-- towncrier release notes start -->

## [1.7.0](https://github.com/HaoZeke/chemparseplot/tree/1.7.0) - 2026-04-07

### Changed

- Replace ``ase.io.read(format="eon")`` with ``readcon.read_con_as_ase()`` for
all ``.con`` file parsing in eOn trajectory modules. Adds ``readcon>=0.7.0``
to the ``neb`` extra. ([#readcon-v2](https://github.com/HaoZeke/chemparseplot/issues/readcon-v2))

### Fixed

- Lazified all top-level rgpycrumbs imports to resolve circular dependency in CI environments.
- Resolved all ruff lint errors: added ``__all__`` for re-exports, fixed import ordering, removed unused imports.


## [1.5.3](https://github.com/HaoZeke/chemparseplot/tree/1.5.3) - 2026-03-26

### Fixed

- `plot_landscape_surface`: accept external `basis` parameter so callers can
  pass a global projection basis from the full dataset, preventing coordinate
  mismatches when surface data is filtered to a subset (e.g. last NEB step).
- `plot_landscape_path_overlay`: same `basis` parameter for consistent
  projection across surface, path overlay, and additional-con markers.
- Y-grid for optimization landscapes now covers the full data d-range
  (`max(x_span/2, |d_max|, |d_min|) * 1.1`) instead of forcing symmetric
  `x_span/2`, which truncated contours for trajectories with large lateral
  deviation.
- Cap GP noise at user-provided smoothing value in `GradientIMQ._fit()`.
  The gradient-enhanced IMQ MLL systematically overestimates noise, destroying
  basin accuracy in NEB landscapes.

### Added

- Heteroscedastic noise support in `_grad_imq_solve` via `noise_per_obs`
  parameter. Enables convergence-based step reweighting where early
  (unconverged) NEB steps get higher noise than the converged path.
- Configurable xyzrender preset via `config` parameter in `_render_xyzrender`,
  `_render_atoms`, `plot_structure_strip`, and `plot_structure_inset`. Presets:
  `paton` (ball-and-stick, default), `bubble` (space-filling, good for
  surfaces), `flat`, `tube`, `wire`, `skeletal`.
- `BaseGradientSurface` accepts `noise_per_obs` kwarg for heteroscedastic GP.

## [1.5.0](https://github.com/HaoZeke/chemparseplot/tree/1.5.0) - 2026-03-23

### Added

- Generalized (s, d) reaction valley projection for single-ended methods (dimer,
  minimization). New `chemparseplot.parse.projection` module extracts the shared
  projection math (`ProjectionBasis`, `compute_projection_basis`, `project_to_sd`,
  `inverse_sd_to_ab`).
- Dimer/saddle search trajectory parser (`chemparseplot.parse.eon.dimer_trajectory`)
  reading eOn `climb.dat` TSV and `climb` movie files.
- Minimization trajectory parser (`chemparseplot.parse.eon.min_trajectory`) reading
  eOn minimization `.dat` and movie files.
- Single-ended visualization module (`chemparseplot.plot.optimization`) with
  `plot_optimization_landscape`, `plot_optimization_profile`,
  `plot_convergence_panel`, and `plot_dimer_mode_evolution`.
- OCI-NEB/RONEB overlay functions: `plot_mmf_peaks_overlay` for MMF peak markers,
  `plot_neb_evolution` for band evolution across iterations.
- Four rendering backends for `plot_structure_strip` and `plot_structure_inset`:
  `xyzrender` (default, ball-and-stick with paton preset), `solvis` (PyVista,
  transparent background), `ovito` (OVITO off-screen), and `ase` (space-filling).
  Unified dispatch via `_render_atoms()`.
- `perspective_tilt` parameter for Rodrigues off-axis rotation to reveal atoms
  hidden by orthographic projection overlap.
- Unified `rotation` parameter across all backends (xyzrender uses `--no-orient`
  + pre-rotated atoms, solvis/ovito pre-rotate positions).
- `col_spacing`, `show_dividers`, `divider_color`, `divider_style` parameters
  for `plot_structure_strip`.
- `calculate_landscape_coords()` in `neb_utils` now accepts explicit `ref_a`/`ref_b`
  reference structures (defaults to first/last for backward compatibility).

### Changed

- Default rendering backend changed from `ase` to `xyzrender` for ball-and-stick
  visualization with bonds visible.

### Fixed

- Fixed infinite recursion in `plot/__init__.py` lazy imports (replaced
  `from chemparseplot.plot import X` with `importlib.import_module`).
- Fixed `plot/structs.py` isinstance typo (`|` vs `,` operator).
- Fixed flat variance contour crash (`v_range < 1e-10` now skips contours).
- Fixed solvis renderer API (hex colors, actor_name, transparent background).

## [1.4.0](https://github.com/HaoZeke/chemparseplot/tree/1.4.0) - 2026-03-15

### Added

- Diataxis documentation structure: tutorials (ORCA NEB, eOn saddle), how-to guides
(parsing, figure creation, install, troubleshooting), explanations (architecture,
NEB design, lazy imports), and reference (API docs, glossary). ([#docs_tutorials](https://github.com/HaoZeke/chemparseplot/issues/docs_tutorials))
- Migrated ChemGP and PLUMED modules from rgpycrumbs: ``parse/chemgp_hdf5`` (HDF5 I/O),
``parse/chemgp_jsonl`` (JSONL parsers), ``parse/plumed`` (HILLS parsing, FES reconstruction,
minima finding), ``plot/plumed`` (FES visualization), ``scripts/plot_gp`` (ChemGP CLI),
``scripts/plt_neb`` (NEB landscape CLI). ([#scope_migration](https://github.com/HaoZeke/chemparseplot/issues/scope_migration))


## [v1.3.0](https://github.com/HaoZeke/chemparseplot/tree/v1.3.0) - 2026-03-09

### Added

- ChemGP plotnine visualization module with convergence curves, RFF quality, hyperparameter sensitivity, trust region, FPS projection, and energy profile plots. ([#chemgp-plotnine](https://github.com/HaoZeke/chemparseplot/issues/chemgp-plotnine))
- Log-scale NLL landscape plot for revealing basin structure in hyperparameter optimization. ([#nll-logscale](https://github.com/HaoZeke/chemparseplot/issues/nll-logscale))
- Re-export RUHI theme from ``chemparseplot.plot`` package with ``setup_publication_theme`` helper. ([#ruhi-reexport](https://github.com/HaoZeke/chemparseplot/issues/ruhi-reexport))
- Matplotlib contourf surface plots (GP progression, NLL landscape, variance overlay) with RUHI colormap matching Julia CairoMakie originals. ([#surface-contourf](https://github.com/HaoZeke/chemparseplot/issues/surface-contourf))

### Changed

- CI-generated README pushed to orphan ``readme`` branch instead of main. ([#ci-readme](https://github.com/HaoZeke/chemparseplot/issues/ci-readme))
- Lazy-import geomscan/structs submodules to avoid mandatory cmcrameri dependency at import time. ([#lazy-imports](https://github.com/HaoZeke/chemparseplot/issues/lazy-imports))

### Miscellaneous

- Taplo TOML formatting, stop tracking autodoc2 generated files. ([#taplo-autodoc](https://github.com/HaoZeke/chemparseplot/issues/taplo-autodoc))


## [v1.2.0](https://github.com/HaoZeke/chemparseplot/tree/v1.2.0) - 2026-02-24

### Added

- Add HDF5 NEB reader for ChemGP output (`chemparseplot.parse.trajectory.hdf5`).
Reads `neb_result.h5` and `neb_history.h5` files with pre-computed parallel
forces and reaction coordinates. Also declares `h5py` as a dependency in the
`neb` optional extra. ([#9](https://github.com/HaoZeke/chemparseplot/issues/9))
- Added optional xyzrender backend for structure strip rendering via renderer parameter. ([#10](https://github.com/HaoZeke/chemparseplot/issues/10))
- Add generic trajectory NEB parser for ASE-readable formats (extxyz, .traj) via `chemparseplot.parse.trajectory.neb`.

### Changed

- Import `NYSTROM_THRESHOLD` from `rgpycrumbs.surfaces` instead of defining a local constant. The public API is unchanged; the threshold value (1000) is now maintained in one place. ([#11](https://github.com/HaoZeke/chemparseplot/issues/11))
- RMSD-R and RMSD-P landscape calculations now run concurrently for improved performance.
- Uniform structure strip sizing via common bounding box across all rendered images.

### Miscellaneous

- Added versionadded directives to all public API docstrings and Zenodo DOI badge.
- Applied ruff linting fixes across the codebase: replaced print with logging, fixed boolean positional args, exception formatting, and timezone awareness.


## [v1.1.0](https://github.com/HaoZeke/chemparseplot/tree/v1.1.0) - 2026-02-21

### Added

- Support for Nystrom-approximated gradient kernels in surface models.
- Uncertainty visualization with variance contours for NEB paths.
- Orthogonal projection to RMSD(R) for path plots.
- Symmetric projection handling.

### Changed

- Smoothed variance calculations and relative windowing for uncertainty plots.
- Expanded grid coverage in path plots to include extra points.
- Consistent scaling across zoomed plots.

### Fixed

- Documentation footer layout and link improvements.
- Added Plausible analytics and TurtleTech footer badges to documentation.


## [v1.0.1](https://github.com/HaoZeke/chemparseplot/tree/v1.0.1) - 2026-02-17

### Added

- CI workflow to auto-generate ``readme.md`` from ``readme_src.org`` on push to main.
- Downstream users page (``used_by.md``) listing public projects that depend on ``chemparseplot``.


## [v1.0.0](https://github.com/HaoZeke/chemparseplot/tree/v1.0.0) - 2026-02-17

### Removed

- ``chemparseplot.basetypes`` module (moved to ``rgpycrumbs.basetypes``). ([#7](https://github.com/HaoZeke/chemparseplot/issues/7))
- ``chemparseplot.analyze.dist`` and ``chemparseplot.analyze.use_ira`` modules (moved to ``rgpycrumbs``). ([#7](https://github.com/HaoZeke/chemparseplot/issues/7))
- ``chemparseplot.plot._surfaces`` and ``chemparseplot.plot._aids`` modules (moved to ``rgpycrumbs.surfaces``). ([#7](https://github.com/HaoZeke/chemparseplot/issues/7))
- PDM configuration and lock file. ([#7](https://github.com/HaoZeke/chemparseplot/issues/7))

### Added

- ``rgpycrumbs>=1.0.0`` as a core dependency for shared data types, interpolation, and surface fitting. ([#7](https://github.com/HaoZeke/chemparseplot/issues/7))
- Add the ruhi and batlow themes
- Added a general regex handling mechanism for columns of numbers
- Augmenting NEB coverage
- Can parse the output of an ORCA geometry scan over a single dimension
- NEB visualization helpers
- Parses the result of ORCA NEB calculations via the `interp` file

### Changed

- Imports now use ``rgpycrumbs`` for base types (``nebiter``, ``nebpath``, ``DimerOpt``, ``MolGeom``, ``SaddleMeasure``, ``SpinID``), interpolation (``spline_interp``), surface fitting (``get_surface_model``), and parsers (``BLESS_LOG``, ``_NUM``, ``tail``). ([#7](https://github.com/HaoZeke/chemparseplot/issues/7))
- ``uv``-first development workflow with ``[tool.uv.sources]`` for ``rgpycrumbs`` git resolution. ``pixi`` retained as minimal config for conda-gated features. ([#7](https://github.com/HaoZeke/chemparseplot/issues/7))
- CI simplified: ``uv sync`` handles all dependencies including ``rgpycrumbs`` via source override; removed manual ``pip install`` step. ([#7](https://github.com/HaoZeke/chemparseplot/issues/7))
- Migrated build tooling from PDM to ``uv`` with ``hatchling``+``hatch-vcs``. ([#7](https://github.com/HaoZeke/chemparseplot/issues/7))
