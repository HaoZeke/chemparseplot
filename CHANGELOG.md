# Changelog

<!-- towncrier release notes start -->

## [1.9.9](https://github.com/HaoZeke/chemparseplot/tree/1.9.9) - 2026-07-15

### Added

- Optional ``auto_thin`` / ``max_surface_points`` on
``plot_landscape_surface`` and ``render_single_ended_landscape`` (default
off). When enabled, dense observation clouds are evenly subsampled for the
GP surface fit only (endpoints kept; viewport and scatter stay full). ([#auto_thin_surface](https://github.com/HaoZeke/chemparseplot/issues/auto_thin_surface))
- Grammar/AST track for text-heavy formats via optional ``parsimonious``
(``chemparseplot[grammar]``): XYZ frames and ORCA final-energy / Cartesian
coordinate extractors under ``chemparseplot.parse.grammar``, with stable
shims ``chemparseplot.api.parse_xyz`` and ``parse_orca_final_energy``. ([#grammar_track](https://github.com/HaoZeke/chemparseplot/issues/grammar_track))


## [1.9.8](https://github.com/HaoZeke/chemparseplot/tree/v1.9.8) - 2026-07-09

### Added

- ``chemparseplot.api`` stable surface: ``extract_orca_geomscan_energy`` (ORCA
  geomscan text → typed pint Quantities), energy helpers, and ``suite_pins()``
  via the rgpkgs hub config.
- Docs: suite ``rgpkgs.toml`` / ``~/.config/rgpkgs`` shared config; feature
  extras documented as transitional toward the hub uv design.

## [1.9.7](https://github.com/HaoZeke/chemparseplot/tree/v1.9.7) - 2026-07-07

### Fixed

- Single-ended landscapes: bold axis labels, tight save with pad, relative-energy
  colorbars (readable ticks), optional title, and 1:1 (s, d) panel limits.
- Endpoint annotations use a high-contrast white box with black edge.

## [1.9.6](https://github.com/HaoZeke/chemparseplot/tree/v1.9.6) - 2026-07-07

### Added

- ``plot_structure_strip(..., prefer_single_row=False)`` honours ``max_cols`` for
  multi-row galleries (e.g. 12 structures → two rows of six larger molecules).

## [1.9.5](https://github.com/HaoZeke/chemparseplot/tree/v1.9.5) - 2026-07-07

### Fixed

- ``landscape_half_span(..., equal_metric=True)`` sets ``|d|`` half-span to at least
  half the *s* window so equal-aspect panels are true 1:1 Å (``Δs = Δd``).

## [1.9.4](https://github.com/HaoZeke/chemparseplot/tree/v1.9.4) - 2026-07-07

### Fixed

- Colorbar label uses explicit ``labelpad`` so equal-aspect (s, d) maps keep a full energy label.
- Strip ``savefig`` pad increased slightly so axis labels are not clipped on wide equal-aspect figures.

## [1.9.3](https://github.com/HaoZeke/chemparseplot/tree/v1.9.3) - 2026-07-07

### Added

- ``mark_saddle_point`` draws a high-contrast gold star (optional vertical guide) for SP markers.

### Fixed

- Landscape half-span is path-driven (``d`` extent), not ``s/2``, so 2D frames no longer pad with empty bands.
- Strip figure saves use a smaller ``pad_inches`` to crop unused canvas.

## [1.9.0](https://github.com/HaoZeke/chemparseplot/tree/v1.9.0) - 2026-07-07

### Added

- Public ``chemparseplot.parse.eon.con_io`` helpers route all CON I/O through readcon (metadata-native energies).

### Changed

- ``neb`` / test extras require ``readcon>=0.13.1`` for eOn 2.16 ``con_spec_version=2`` frame metadata.
- Energy unit conversions route through pint quantities; single-ended landscape pipeline centralized.

### Fixed

- Avoid recursion when lazy-loading ``chemparseplot.parse``; vectorize stitch RMSD path.


## [1.8.0](https://github.com/HaoZeke/chemparseplot/tree/1.8.0) - 2026-06-27

### Added

- NEB and optimization plot APIs accept energy unit conversions for unit-aware profile and landscape plots. ([#energy-units](https://github.com/HaoZeke/chemparseplot/issues/energy-units))
- Metadata-native eOn trajectory support prefers CON metadata over ``.dat`` tables, with typed parser results for trajectories, ChemGP, PLUMED, Sella, and ORCA NEB overlays. ([#metadata-native-eon](https://github.com/HaoZeke/chemparseplot/issues/metadata-native-eon))
- ``stitch_neb_segments`` combines multiple NEB segments into one full path for end-to-end visualization and analysis. ([#stitch-neb](https://github.com/HaoZeke/chemparseplot/issues/stitch-neb))

### Changed

- Parser and plot modules return shared typed, unit-aware result records; NEB plotting helpers and single-ended APIs are centralized. ([#typed-parsers](https://github.com/HaoZeke/chemparseplot/issues/typed-parsers))

### Fixed

- RMSD landscape coordinates require IRA; projected path visibility and minimization prefix defaults improved for eOn outputs. ([#ira-landscape](https://github.com/HaoZeke/chemparseplot/issues/ira-landscape))
- xyzrender structure strips size and clear axes correctly for single-ended and NEB plots (measured budgets, matte cropping, label clearance). ([#strip-layout](https://github.com/HaoZeke/chemparseplot/issues/strip-layout))

### Miscellaneous

- Migrated documentation deployment from GitHub Pages to Cloudflare Pages.
- Added OIDC trusted publishing workflow for PyPI releases, gated on tests, lints, and docs.



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
