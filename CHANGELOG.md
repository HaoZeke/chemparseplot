# Changelog

<!-- towncrier release notes start -->

## [1.14.1](https://github.com/HaoZeke/chemparseplot/tree/1.14.1) - 2026-11-08

### Added

- `llms-full.txt` at the repository root has the manual in one Markdown file, the pages in the order of the site's navigation, converted from the Org and MyST sources by `python scripts/llms/build_llms.py` (settings in `scripts/llms/llms.toml`).
Its header names what it leaves out: the API reference, and any page listed under `full_exclude`.
`--check` exits 1 while the file is stale, and 2 when the site map `doc/source/llms.txt` links a page that does not exist.
- `skills/` has three task guides, each a `SKILL.md` with scripts that check what it says against the code.
`chemparseplot-orca-scan` reads a relaxed surface scan from the quantum chemistry program (ORCA) and plots it.
It shows that the distances `extract_energy_data` labels bohr are in ångström.
`chemparseplot-add-parser` covers adding a parser with its tests, fragment and docs pages.
`chemparseplot-docs` finds the source of a page of the manual and covers the docs build.
`python scripts/skills/run_skills.py` checks every guide and runs its examples.

### Fixed

- Journal figures set `image.cmap` to `ruhi_diverging`.
`plot_landscape_surface`, `plot_optimization_landscape`, `render_single_ended_landscape`, `plot_mmf_peaks_overlay` and `plot_surface_contour` default to that map, and the ChemGP energy surfaces use it.
`ruhi_seq` stays registered for the light-to-dark series.
- The Ruhi theme's profile and landscape colour maps are `ruhi_diverging` again: teal `#004D40`, sky `#1E88E5`, magenta `#D81B60`, coral `#FF655D`, sunshine `#F1DB4B`.
`ruhi_seq` stays registered.

### Miscellaneous

- The `Skills` workflow (`.github/workflows/ci_skills.yml`) fails when `llms-full.txt` is stale or `doc/source/llms.txt` links a page that does not exist, runs the generator's tests, and runs the examples and scripts of every guide under `skills/`.
- `AGENTS.md` lists the install, test, lint and documentation commands, and the common tasks with the guide for each.
It also lists the pitfalls.
Among them, `extract_energy_data` labels scan distances bohr that the quantum chemistry program (ORCA) printed in ångström.
`uv.lock` is out of date, and the docs export needs Emacs 30 and an ox-rst that the same Emacs compiled.
One pitfall lists the documentation pages that disagree with the code.


## [1.14.0](https://github.com/HaoZeke/chemparseplot/tree/1.14.0) - 2026-11-04

### Added

- A tutorial runs a small nudged elastic band with ASE and draws its energy profile for two journals.
A how-to covers journal figures, and the FAQ says how to keep a figure one column wide.
An explanation traces each size, line width and colour of the journal style to its source.
The reference gains entries for the journal, palette and ChemGP surface modules.
- `chemparseplot.plot.journal` sizes and styles figures for five journals.
The `acs` preset covers the Journal of Chemical Theory and Computation and the Journal of Chemical Information and Modeling.
`elsevier` covers Computer Physics Communications and `rsc` Physical Chemistry Chemical Physics.
`aip` covers The Journal of Chemical Physics.
`journal_figure` makes a figure of the exact single- or double-column width.
Its panels are half again as wide as they are tall unless the data give them a shape.
Text is set at the body, `\small` and `\footnotesize` sizes of each journal's two-column page, in the theme's face.
For `ruhi` that face is Atkinson Hyperlegible, and the faces the journal names stand in where it is not installed.
Line widths, tick lengths, gaps and marker sizes follow from the text size.
They follow the lettering standard ISO 3098-1 and the line standard ISO 128-2 of the International Organization for Standardization (ISO).
`series_style` pairs each series' Ruhi colour with a line type from `LINE_TYPES`, where the ISO 128-2 types with one kind of mark come before those with two.
`label_panels` letters the panels, and `label_lines` and `label_hline` label lines at their ends.
`smart_legend` places a legend where it covers no data.
`sci_axis` writes numbers from 0.1 to 1000 plain.
`relative_error` and `log_error_axis` put relative errors on log scales.
`save_figure` writes PDF, SVG and raster files at the exact figure size, with TrueType fonts embedded and no timestamps.
`setup_publication_theme` takes `journal` to apply the same style.
- `chemparseplot.plot.palette` defines the Ruhi colours and the checks behind their use.
`palette_report` gives each colour's contrast on the background by the Web Content Accessibility Guidelines 2.1.
It gives the smallest CIEDE2000 difference between any two colours for normal vision and three simulated deficiencies.
The simulation follows Machado and co-workers, and the Okabe-Ito palette is the reference.
`colormap_report` gives the CIELAB lightness profile of a colour map.
`ruhi_seq` is a perceptually ordered map whose lightness is monotone for every vision type.
`ruhi_div` is monotone on each side of its light grey centre.
- `plot_neb_profile` draws the energy profile of a nudged elastic band in the journal style.
The curve is a cubic Hermite interpolant whose slopes are minus the forces along the path.
It is sampled once per output pixel.
The barrier is written above the peak.
Earlier bands run from light to dark, with an iteration colour bar.
A reference energy is drawn as a dashed, labelled line.
- `plot_relative_error` plots the relative error of a series against a reference value on a log scale.
Decades are labelled as powers of ten, and exact results are drawn at machine epsilon.

### Changed

- The default colour maps are Ruhi maps.
`ruhi_seq` replaces viridis in the nudged elastic band and optimization landscapes and their path overlays.
It also replaces coolwarm in `plot_mmf_peaks_overlay`, which takes `cmap`.
It replaces the `ruhi_diverging` blend, whose lightness is not monotone, in the theme and the ChemGP surfaces.
`chemparseplot.plot.landscape.plot_landscape_surface` uses `ruhi_seq` in place of that blend too.
Earlier bands in `plot_neb_evolution` and Wiberg bond orders use only the part of `ruhi_seq` with a 3:1 contrast on the background.
`mark_saddle_point` draws its marker in a Ruhi colour.
`plot_orca_neb_profile` and `plot_orca_neb_energy_profile` mark their key points in Ruhi colours.
`plot_orca_neb_energy_profile` also draws its line in the first Ruhi line colour by default.
`setup_global_theme` sets the Ruhi line colours as the colour cycle and the theme's map as `image.cmap`.
- `plot_energy_path` takes the image marker size and edge width from the `lines.markersize` and `lines.markeredgewidth` rcParams.
The journal style sets both; they were fixed at 6 pt and 0.5 pt before.
With matplotlib's defaults the edge is 1 pt.
It takes `markersize`, `markeredgewidth`, `markeredgecolor` and `samples`.
Without `samples` the curve has a point for every pixel column at `savefig.dpi`, where it had 300 points.
- `plot_single_ended_convergence` draws in the journal style, with panel letters.
The convergence measure and the step size are on log scales.
Lines are labelled at their ends or in a legend off the data, and `threshold` draws a dashed, labelled threshold.
`plot_convergence_panel` puts the step size on a log scale too and marks only the last iteration.
`plot_single_ended_profile` gives each trajectory a Ruhi colour and its own line type.
It takes line width and marker size from the rcParams, where they were fixed numbers.
Like `plot_convergence_panel`, it marks only the last iteration, so markers no longer hide the line types.
The line types are those of the drawing standard ISO 128-2 of the International Organization for Standardization (ISO).
`save_standard_figure` saves constrained-layout figures at their exact size.
- `plot_surface_contour` draws the energy landscape in the journal style at the column width, with the `ruhi_seq` map.
The bands, contour lines and colour-bar ticks share one set of levels.
Each contour line is black or white, whichever contrasts more with the bands beside it.
Minima are lettered from A and saddles numbered from S1.
It takes `journal`, `columns`, `cmap`, `xlabel`, `ylabel` and `zlabel`.

### Fixed

- `chemparseplot.plot.chemgp` imports plotnine and pandas only when a line chart is first drawn.
The surface plots import and draw with matplotlib alone, and the ChemGP surfaces tutorial runs to the end.

### Miscellaneous

- Image comparisons check the nudged elastic band profile, the energy landscape and the convergence figures.
The baseline images come from effective medium theory runs of Au on Al(100).
`tests/plot/data/au_al100.py` records the runs.


## [1.13.2](https://github.com/HaoZeke/chemparseplot/tree/1.13.2) - 2026-10-31

### Fixed

- The Wiberg edge reach is two and a half covalent-radius sums, so forming H-H contacts of dissociation saddles (1.3 to 1.6 A) keep their edge while separated radical centres stay unbonded.


## [1.13.1](https://github.com/HaoZeke/chemparseplot/tree/1.13.1) - 2026-10-31

### Fixed

- Reaction panels draw no Wiberg edge between atoms farther apart than twice their covalent-radius sum (`max_contact`), so a closed-shell density at a diradical-like geometry no longer shows separated radical centres as bonded; reaction atlas rows grow with their bond annotations and the colour bar keeps its own margin.


## [1.13.0](https://github.com/HaoZeke/chemparseplot/tree/1.13.0) - 2026-10-31

### Added

- `plot_reaction_gallery` lays out one structure per reaction in a grid of tiles, each titled with the reaction's name, an optional chemistry line and its measured bond-order annotations.


## [1.12.0](https://github.com/HaoZeke/chemparseplot/tree/1.12.0) - 2026-10-30

### Added

- Added `chemparseplot.parse.surrogate.mlneb.parse_mlneb_run`, which reads ASE/CatLearn ML-NEB output (`predicted.traj`, `evaluated.traj`, `ml_summary.txt`) into the same records as the gpr_optim parser, and `read_search` to pick a parser by producer.
- Added `chemparseplot.parse.surrogate`: producer-independent records of surrogate-assisted saddle searches (band, search and single-ended histories, campaign tables) and a parser for gpr_optim campaign records.
- Added `chemparseplot.plot.surrogate`: band profile with posterior mean and true evaluations, band evolution, convergence against oracle calls, model diagnostics, single-ended curvature and force, and campaign dumbbell, matrix, wall-time, scaling and efficiency figures. They take only the records of `chemparseplot.parse.surrogate`.
- Added a how-to and a features entry for the surrogate-search records and figures.
- The campaign wall figure (`plot_cases_wall`) reads an optional `pipeline_wall_sd` column. It draws that sample standard deviation as a horizontal bar and writes it beside the release time.
- The surrogate profile and landscape draw the strip of structures `rgpycrumbs eon plt-neb` draws (reactant, saddle or climbing image, product, or every image, or N evenly spaced ones), through `plot_structure_strip`. Atom types come from `band.numbers`, read from the cell (`band/band.con`, `saddle/pos.con`) or `read_atom_types(FILE)`; without any the call raises a `ValueError` naming what was looked for.
- `chemparseplot.plot.rings` draws the faces `ringNetwork` returns and the bridges between them with xyzrender. The command is `rgpycrumbs geom plt-rings`.
- `chemparseplot.plot.rings` tracks a query across frames. A block hop is a change of the terminal ring's atoms. A junction frame keeps that ring. The command is `rgpycrumbs geom plt-rings-track`.

### Changed

- CON I/O requires readcon 0.17.1.
`chemparseplot.api` now provides `load_trajectory`, `plot_con_overview`, and `plot_con_overlay`.
`.con` reads go through `parse.eon.con_io`, with precision omitted and canonical left off.
- Documentation is authored in org-mode and exported to RST at build.
Shibuya Ecosystem nav and intersphinx point at the rgpycrumbs and pychum docs
sites.
- The surrogate landscape (`chemparseplot.plot.surrogate.plot_reduced_landscape`) is drawn with the NEB landscape functions: the progress/deviation plane, the GP energy surface fitted to the oracle evaluations, the evaluations, the final path, the climbing image and the reported saddle, each in the legend. `plot_landscape_surface` takes `variance_label` to set the variance contour text.
The surrogate landscape fades the GP surface where the relative variance exceeds a threshold (default 0.95), and `plot_landscape_surface` takes `fade_variance_above` for the same purpose.

### Fixed

- Cost-frontier panels label their log axes with plain major ticks, so minor-tick labels no longer overprint; axes narrower than one decade get unit subdivisions.


## [1.11.0](https://github.com/HaoZeke/chemparseplot/tree/1.11.0) - 2026-08-10

### Added

- ``plot_structure_strip`` accepts a separate caption axis so multiline gallery
labels cannot overlap rendered structures.


## [1.10.2](https://github.com/HaoZeke/chemparseplot/tree/1.10.2) - 2026-08-09

### Fixed

- Sketch-map landmark pinning is now guaranteed: pinned rows are unioned into the landmark set after farthest-point selection, which otherwise skips points near an already-kept reference.


## [1.10.1](https://github.com/HaoZeke/chemparseplot/tree/1.10.1) - 2026-08-09

### Fixed

- Structure-strip label bands size to the tallest multiline label, so two-line gallery captions no longer overlap the rendered structures.


## [1.10.0](https://github.com/HaoZeke/chemparseplot/tree/1.10.0) - 2026-08-08

### Added

- Sketch-map landscape figures (``chemparseplot.plot.landscape``: coordination-number and SOAP descriptors, the published dimlandmark/dimred schedule, ruhi contour surfaces, margin structure insets) and approximate disconnectivity graphs (``chemparseplot.plot.disconnectivity``: superbasin merge trees over descriptor k-NN graphs with spectral basin colouring) for global-optimization minima databases.


## 1.9.17

### Fixed

- Lazy ``parse.eon.frame_series`` so bare ``import chemparseplot.parse`` no
  longer requires the ``neb`` / ``readcon`` extra (tutorial + pure imports).
- Tests: xyzrender missing-package checks use import blocking (Python API),
  not ``shutil.which``; landscape save / strip height expectations match
  current tight-crop layout.

## 1.9.16

### Fixed

- Test: CLI vs Python API xyzrender strip images must be pixel-identical
  (same package version).

## 1.9.15

### Fixed

- Structure strips: use xyzrender **Python API** via ``ensure_import`` /
  ``RGPYCRUMBS_AUTO_DEPS`` (no PATH binary / hard env pin).

## 1.9.14

### Added

- ``parse.eon.frame_series``: ConFrame sequence → energies / NEB path arrays /
  ``MinTrajectoryData`` / ``DimerTrajectoryData`` for object-plot adapters.

## 1.9.13

### Fixed

- `plot/neb.py`: add `from __future__ import annotations` so `SurfaceFitConfig`
  return annotations do not raise `NameError` on Python 3.11+ (broke 1.9.10–1.9.12).

## [1.9.12](https://github.com/HaoZeke/chemparseplot/tree/1.9.12) - 2026-07-15

### Documentation

- Switch Sphinx theme from library to **Shibuya** with ``sphinx-design``
  (grids, cards) and ``sphinxcontrib-mermaid`` diagrams.
- Index / features / quickstart: ecosystem mermaid, card grids, auto_thin tip.
- Org howto/architecture: mermaid data-flow for landscape fits and suite stack.


## [1.9.11](https://github.com/HaoZeke/chemparseplot/tree/1.9.11) - 2026-07-15

### Documentation

- Full orgmode pass for dense landscape fits: ``SurfaceFitConfig`` /
  ``auto_thin`` (default off) in reference plotting API, create-NEB-figures
  howto, FAQ, troubleshooting, and glossary.


## [1.9.10](https://github.com/HaoZeke/chemparseplot/tree/1.9.10) - 2026-07-15

### Added

- ``SurfaceFitConfig`` dataclass (``from_mapping`` for TOML tables) and
``surface_fit=`` on ``plot_landscape_surface`` /
``render_single_ended_landscape``. Prefer this over growing keyword lists;
``auto_thin`` still defaults to off. ([#surface_fit_config](https://github.com/HaoZeke/chemparseplot/issues/surface_fit_config))


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
