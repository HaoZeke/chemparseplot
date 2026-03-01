
# Table of Contents

-   [About](#orgaffc63a)
    -   [Installation](#org4762a59)
    -   [Ecosystem Overview](#org11ff012)
    -   [Features](#orgb59a5ac)
        -   [Supported Engines](#orgcdef401)
    -   [Documentation](#orgfb3785a)
    -   [Contributing](#org93b51c2)
-   [License](#orgfeaa34c)
-   [Acknowledgments](#org63516b7)



<a id="orgaffc63a"></a>

# About

![img](branding/logo/chemparseplot_logo.png)

[![Tests](https://github.com/HaoZeke/chemparseplot/actions/workflows/build_test.yml/badge.svg)](https://github.com/HaoZeke/chemparseplot/actions/workflows/build_test.yml)
[![Linting](https://github.com/HaoZeke/chemparseplot/actions/workflows/ci_prek.yml/badge.svg)](https://github.com/HaoZeke/chemparseplot/actions/workflows/ci_prek.yml)
[![Docs](https://github.com/HaoZeke/chemparseplot/actions/workflows/build_docs.yml/badge.svg)](https://github.com/HaoZeke/chemparseplot/actions/workflows/build_docs.yml)
[![PyPI](https://img.shields.io/pypi/v/chemparseplot)](https://pypi.org/project/chemparseplot/)
[![Python](https://img.shields.io/pypi/pyversions/chemparseplot)](https://pypi.org/project/chemparseplot/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![One Good Tutorial docs checklist v1: adopted](https://onegoodtutorial.org/badge/adopted-v1.svg)](https://onegoodtutorial.org/about/badge/?v=1)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![DOI](https://zenodo.org/badge/725730118.svg)](https://doi.org/10.5281/zenodo.18529752)

A **pure-python**<sup><a id="fnr.1" class="footref" href="#fn.1" role="doc-backlink">1</a></sup> parsing and plotting library for computational
chemistry outputs. `chemparseplot` extracts structured data from quantum
chemistry codes (ORCA, eOn, Sella, ChemGP) and produces publication-quality,
unit-aware visualizations with [scientific color maps](https://www.fabiocrameri.ch/colourmaps/).

Computational tasks (surface fitting, structure analysis, interpolation) are
handled by [`rgpycrumbs`](https://github.com/HaoZeke/rgpycrumbs), which is a required dependency. `chemparseplot` parses
output files, delegates heavy computation to `rgpycrumbs`, and produces
publication-quality plots.


<a id="org4762a59"></a>

## Installation

    pip install chemparseplot
    # With plotting support
    pip install "chemparseplot[plot]"
    # Everything
    pip install "chemparseplot[all]"

For development:

    git clone https://github.com/HaoZeke/chemparseplot
    cd chemparseplot
    uv sync --all-extras

See the [installation guide](https://chemparseplot.rgoswami.me/installation.html) and [quickstart](https://chemparseplot.rgoswami.me/quickstart.html) for details.


<a id="org11ff012"></a>

## Ecosystem Overview

`chemparseplot` is part of the `rgpycrumbs` suite of interlinked libraries.

![img](branding/logo/ecosystem.png)


<a id="orgb59a5ac"></a>

## Features

-   **Parsing** computational chemistry output files into structured data
-   **Plotting** with [scientific color maps](https://www.fabiocrameri.ch/colourmaps/) (camera-ready)
-   **Unit preserving** throughout via `pint`
-   **Computation** delegated to [`rgpycrumbs`](https://github.com/HaoZeke/rgpycrumbs) for surface fitting, interpolation,
    and structure analysis


<a id="orgcdef401"></a>

### Supported Engines

-   ORCA (**5.x**)
    -   Geometry scan (`OPT`) energy profiles
    -   Nudged elastic band (`NEB`) path visualization
-   eOn
    -   Saddle search parsing (Dimer, GPRD, LBFGS methods)
    -   NEB path energy profiles with landscape projections
-   Sella
    -   Saddle point optimization result parsing
-   Trajectory formats
    -   HDF5 trajectories (ChemGP output with pre-computed forces)
    -   Generic ASE-readable formats (extxyz, .traj) for NEB analysis


<a id="orgfb3785a"></a>

## Documentation

Full documentation is at <https://chemparseplot.rgoswami.me>. This includes:

-   A [quickstart guide](https://chemparseplot.rgoswami.me/quickstart.html)
-   [Tutorials](https://chemparseplot.rgoswami.me/tutorials/index.html) for common workflows
-   [API reference](https://chemparseplot.rgoswami.me/apidocs/index.html)


<a id="org93b51c2"></a>

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](https://github.com/HaoZeke/chemparseplot/blob/main/CONTRIBUTING.md) for development setup and
guidelines, and our [Code of Conduct](https://github.com/HaoZeke/chemparseplot/blob/main/CODE_OF_CONDUCT.md).

For bug reports or questions, open an issue on [GitHub](https://github.com/HaoZeke/chemparseplot/issues).


<a id="orgfeaa34c"></a>

# License

MIT. However, this is an academic resource, so **please cite** as much as possible
via:

-   The [Zenodo DOI](https://doi.org/10.5281/zenodo.18529752) for general use.
-   The `wailord` paper for ORCA usage


<a id="org63516b7"></a>

# Acknowledgments

This project builds on work supported by the University of Iceland and the
Icelandic Research Fund. `chemparseplot` relies on [`rgpycrumbs`](https://github.com/HaoZeke/rgpycrumbs) for computational
modules.


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> To distinguish it from my other thin-python wrapper projects
