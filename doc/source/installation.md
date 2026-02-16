# Installation

The easiest way to install `chemparseplot` is via `pip`:

```bash
pip install chemparseplot
# For plotting
pip install matplotlib
```

This will automatically install [`rgpycrumbs`](https://github.com/HaoZeke/rgpycrumbs)
as a dependency, which provides the computational modules used by `chemparseplot`
(surface fitting, interpolation, data types, etc.).

Local clones of the GitHub repo are best served setting up the maximally
reproducible development environment:

```bash
# Probably in $HOME/Git/Github
git clone git@github.com:HaoZeke/chemparseplot
cd chemparseplot
# For a reproducible python version
pixi shell
pdm install -dG:all
```

## Auxiliary Software

Since `chemparseplot` is meant to facilitate working with the results of
computational chemistry / atomic physics codes, some links to commonly used
tools are enumerated below. Note that technically it is the **outputs** of these
codes which is required, not the codes themselves.

- `ORCA` can be obtained (freely) after [registering on their forum](https://orcaforum.kofo.mpg.de/app.php/portal)
- `ASE`, the [atomic simulation environment](https://wiki.fysik.dtu.dk/ase/), is also on PyPI and can be installed via `pip`
- `eON` is [freely available](https://eondocs.org/) and on `conda-forge`. 
