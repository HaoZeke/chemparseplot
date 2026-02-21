
# Table of Contents

-   [About](#org41f780c)
    -   [Ecosystem Overview](#org8d84664)
    -   [Features](#orgcd8c670)
        -   [Supported Engines [WIP]](#orgf54561c)
    -   [Rationale](#orgc44cf84)
-   [License](#orgf123668)



<a id="org41f780c"></a>

# About

![img](branding/logo/chemparseplot_logo.png)

[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

A **pure-python**<sup><a id="fnr.butwhy" class="footref" href="#fn.butwhy" role="doc-backlink">1</a></sup> project to provide unit-aware uniform visualizations
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


<a id="org8d84664"></a>

## Ecosystem Overview

`chemparseplot` is part of the `rgpycrumbs` suite of interlinked libraries.

![img](branding/logo/ecosystem.png)


<a id="orgcd8c670"></a>

## Features

-   [Scientific color maps](https://www.fabiocrameri.ch/colourmaps/) for the plots
    -   Camera ready
-   Unit preserving
    -   Via `pint`


<a id="orgf54561c"></a>

### Supported Engines [WIP]

-   ORCA (**5.x**)
    -   Scanning energies over a degree of freedom (`OPT` scans)
    -   Nudged elastic band (`NEB`) visualizations (over the "linearized" reaction
        coordinate)


<a id="orgc44cf84"></a>

## Rationale

`wailord` is for production runs, however often there is a need to collect
"spot" calculation visualizations, which should nevertheless be uniform, i.e.
either Bohr/Hartree or Angstron/eV or whatever.

Also I couldn't find (m)any scripts using the scientific colorschemes.


<a id="orgf123668"></a>

# License

MIT. However, this is an academic resource, so **please cite** as much as possible
via:

-   The Zenodo DOI for general use.
-   The `wailord` paper for ORCA usage


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> To distinguish it from my other thin-python wrapper projects
