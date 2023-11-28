
# Table of Contents

1.  [About](#org933ae35)
    1.  [Features](#org29aeadc)
        1.  [Supported Engines [WIP]](#org326cafe)
    2.  [Rationale](#orgb6526b7)
2.  [License](#org3d31168)


<a id="org933ae35"></a>

# About

[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

A **pure-python**<sup><a id="fnr.1" class="footref" href="#fn.1" role="doc-backlink">1</a></sup> project to provide unit-aware uniform visualizations
of common computational chemistry tasks. Essentially this means we provide:

-   Plotting scripts for specific workflows
-   Parsers for various software outputs

This is a spin-off from `wailord` ([here](https://wailord.xyz)) which is meant to handle aggregated
runs in a specific workflow, while here the goal is to do no input handling and
very pragmatic output parsing, with the goal of generating uniform plots.


<a id="org29aeadc"></a>

## Features

-   [Scientific color maps](https://www.fabiocrameri.ch/colourmaps/) for the plots
    -   Camera ready
-   Unit preserving
    -   Via `pint`


<a id="org326cafe"></a>

### Supported Engines [WIP]

-   ORCA (**5.x**)
    -   Scanning energies over a degree of freedom (`OPT` scans)
    -   Nudged elastic band (`NEB`) visualizations (over the "linearized" reaction
        coordinate)


<a id="orgb6526b7"></a>

## Rationale

`wailord` is for production runs, however often there is a need to collect
"spot" calculation visualizations, which should nevertheless be uniform, i.e.
either Bohr/Hartree or Angstron/eV or whatever.

Also I couldn't find (m)any scripts using the scientific colorschemes.


<a id="org3d31168"></a>

# License

MIT. However, this is an academic resource, so **please cite** as much as possible
via:

-   The Zenodo DOI for general use.
-   The `wailord` paper for ORCA usage


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> To distinguish it from my other thin-python wrapper projects
