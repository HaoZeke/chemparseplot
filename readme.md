
# Table of Contents

-   [About](#org92cf9f4)
    -   [Ecosystem Overview](#orgacc123d)
    -   [Features](#org41d4f06)
        -   [Supported Engines [WIP]](#orgd3d7c1b)
    -   [Rationale](#org7083dde)
-   [License](#orgae8a918)



<a id="org92cf9f4"></a>

# About

![img](branding/logo/chemparseplot_logo.png)

[![Tests](https://github.com/HaoZeke/chemparseplot/actions/workflows/build_test.yml/badge.svg)](https://github.com/HaoZeke/chemparseplot/actions/workflows/build_test.yml)
[![PyPI](https://img.shields.io/pypi/v/chemparseplot)](https://pypi.org/project/chemparseplot/)
[![Python](https://img.shields.io/pypi/pyversions/chemparseplot)](https://pypi.org/project/chemparseplot/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![One Good Tutorial docs checklist v1: adopted](https://onegoodtutorial.org/badge/adopted-v1.svg)](https://onegoodtutorial.org/about/badge/?v=1)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![DOI](https://zenodo.org/badge/725730118.svg)](https://doi.org/10.5281/zenodo.18529752)

A **pure-python**<sup><a id="fnr.1" class="footref" href="#fn.1" role="doc-backlink">1</a></sup> project to provide unit-aware uniform visualizations
of common computational chemistry tasks. Essentially this means we provide:

-   Parsers for various computational chemistry software outputs
-   Plotting scripts for specific workflows

Computational tasks (surface fitting, structure analysis, interpolation) are
handled by [`rgpycrumbs`](https://github.com/HaoZeke/rgpycrumbs), which is a required dependency. `chemparseplot` parses
output files, delegates heavy computation to `rgpycrumbs`, and produces
publication-quality plots.

This is a spin-off from `wailord` ([here](https://wailord.xyz)) which is meant to handle aggregated
runs in a specific workflow, while here the goal is to do no input handling and
very pragmatic output parsing, with the goal of generating uniform plots.


<a id="orgacc123d"></a>

## Ecosystem Overview

`chemparseplot` is part of the `rgpycrumbs` suite of interlinked libraries.

![img](branding/logo/ecosystem.png)


<a id="org41d4f06"></a>

## Features

-   [Scientific color maps](https://www.fabiocrameri.ch/colourmaps/) for the plots
    -   Camera ready
-   Unit preserving
    -   Via `pint`


<a id="orgd3d7c1b"></a>

### Supported Engines [WIP]

-   ORCA (**5.x**)
    -   Scanning energies over a degree of freedom (`OPT` scans)
    -   Nudged elastic band (`NEB`) visualizations (over the "linearized" reaction
        coordinate)


<a id="org7083dde"></a>

## Rationale

`wailord` is for production runs, however often there is a need to collect
"spot" calculation visualizations, which should nevertheless be uniform, i.e.
either Bohr/Hartree or Angstron/eV or whatever.

Also I couldn't find (m)any scripts using the scientific colorschemes.


<a id="orgae8a918"></a>

# License

MIT. However, this is an academic resource, so **please cite** as much as possible
via:

-   The [Zenodo DOI](https://doi.org/10.5281/zenodo.18529752) for general use.
-   The `wailord` paper for ORCA usage


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> To distinguish it from my other thin-python wrapper projects
