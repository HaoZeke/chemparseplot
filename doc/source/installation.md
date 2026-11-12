# Installation

The easiest way to install `chemparseplot` is via `pip`:

```bash
pip install chemparseplot
```

This will automatically install [`rgpycrumbs`](https://github.com/HaoZeke/rgpycrumbs)
as a dependency, which provides the computational modules used by `chemparseplot`
(surface fitting, interpolation, data types, etc.).

## Optional extras (transitional)

```bash
# Plotting support (matplotlib, cmcrameri colormaps)
pip install "chemparseplot[plot]"

# NEB analysis (ase, h5py, polars, readcon). CON/convel I/O is readcon-core.
pip install "chemparseplot[neb]"

# Everything transitional
pip install "chemparseplot[all]"
```

The `neb` extra installs [readcon-core](https://github.com/lode-org/readcon-core)
(PyPI: `readcon`) for CON and convel files.

Structure strips use **xyzrender** via the Python API
(`ensure_import("xyzrender")` when `RGPYCRUMBS_AUTO_DEPS=1`); there is no
`[xyzrender]` extra. Prefer bare install + AUTO_DEPS / suite pins for new work.

Lennard-Jones clusters use **solvis**.
Install it with `pip install "chemparseplot[solvis]"`.
The test extra installs that stack too, including pyvista and vtk.
The VTK wheel draws through GLX.
When the process has no display, a cluster render starts Xvfb.
Without a display, EGL, or OSMesa, the screenshot segfaults.
Structure captions use Atkinson Hyperlegible when that font is installed.
Without it they fall back to the matplotlib sans-serif face.

## Development install

Local clones of the GitHub repo are best served setting up the maximally
reproducible development environment:

```bash
git clone https://github.com/HaoZeke/chemparseplot
cd chemparseplot
# Install with all dev dependencies
uv sync --all-extras
```

## Auxiliary Software

Since `chemparseplot` is meant to facilitate working with the results of
computational chemistry / atomic physics codes, some links to commonly used
tools are enumerated below. Note that technically it is the **outputs** of these
codes which is required, not the codes themselves.

- `ORCA` can be obtained (freely) after [registering on their forum](https://orcaforum.kofo.mpg.de/app.php/portal)
- `ASE`, the [atomic simulation environment](https://wiki.fysik.dtu.dk/ase/), is also on PyPI and can be installed via `pip`
- `eON` is [freely available](https://eondocs.org/) and on `conda-forge`.


## Plot stack and lazy deps

Structure strips use the **xyzrender Python API** via
`rgpycrumbs.ensure_import` when `RGPYCRUMBS_AUTO_DEPS=1` (no PATH binary).
Feature extras (`plot`, `neb`, …) remain transitional; prefer bare install +
suite pins / AUTO_DEPS for new work. ConFrame series helpers live in
`chemparseplot.parse.eon.frame_series` (v1.9.14+).
