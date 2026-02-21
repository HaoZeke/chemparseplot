# Changelog

<!-- towncrier release notes start -->

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
