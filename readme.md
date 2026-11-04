
# Table of Contents

-   [About](#org5ed400d)
    -   [Installation](#orgd7eda52)
    -   [Ecosystem Overview](#orgbbc1557)
    -   [Features](#org6fcd299)
        -   [Supported Engines](#orgc247628)
    -   [Documentation](#orgb030d7a)
    -   [Contributing](#org189ed07)
    -   [Release Process](#org535cdf9)
-   [License](#orgfb43acb)
-   [Acknowledgments](#org5cea173)

> Canonical source note: this Org file is authoritative for contributor-facing documentation.
> Rendered Markdown files are derived artifacts and should not be edited separately.


<a id="org5ed400d"></a>

# About

![img](branding/logo/chemparseplot_logo.png)

[![Tests](https://github.com/HaoZeke/chemparseplot/actions/workflows/build_test.yml/badge.svg)](https://github.com/HaoZeke/chemparseplot/actions/workflows/build_test.yml) [![Linting](https://github.com/HaoZeke/chemparseplot/actions/workflows/ci_prek.yml/badge.svg)](https://github.com/HaoZeke/chemparseplot/actions/workflows/ci_prek.yml) [![Docs](https://github.com/HaoZeke/chemparseplot/actions/workflows/build_docs.yml/badge.svg)](https://github.com/HaoZeke/chemparseplot/actions/workflows/build_docs.yml) [![PyPI](https://img.shields.io/pypi/v/chemparseplot)](https://pypi.org/project/chemparseplot/) [![Python](https://img.shields.io/pypi/pyversions/chemparseplot)](https://pypi.org/project/chemparseplot/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![One Good Tutorial docs checklist v1: adopted](https://onegoodtutorial.org/badge/adopted-v1.svg)](https://onegoodtutorial.org/about/badge/?v=1) [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-blueviolet.svg)](https://github.com/pypa/hatch) [![DOI](https://zenodo.org/badge/725730118.svg)](https://doi.org/10.5281/zenodo.18529752)

A **pure-python**<sup><a id="fnr.butwhy" class="footref" href="#fn.butwhy" role="doc-backlink">1</a></sup> parsing and plotting library for computational chemistry outputs.
`chemparseplot` extracts structured data from the output of the quantum chemistry program (ORCA), eOn, Sella and ChemGP.
It produces publication-quality, unit-aware visualizations with [scientific color maps](https://www.fabiocrameri.ch/colourmaps/).

Computational tasks (surface fitting, structure analysis, interpolation) are handled by [`rgpycrumbs`](https://github.com/HaoZeke/rgpycrumbs), which is a required dependency.
`chemparseplot` parses output files, delegates heavy computation to `rgpycrumbs`, and produces publication-quality plots.


<a id="orgd7eda52"></a>

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


<a id="orgbbc1557"></a>

## Ecosystem Overview

`chemparseplot` is part of the `rgpycrumbs` suite of interlinked libraries.

![img](branding/logo/ecosystem.png)


<a id="org6fcd299"></a>

## Features

-   **Parsing** computational chemistry output files into structured data.
-   **Plotting** with [scientific color maps](https://www.fabiocrameri.ch/colourmaps/) (camera-ready).
-   **Unit preserving** throughout via `pint`.
-   **Computation** delegated to [`rgpycrumbs`](https://github.com/HaoZeke/rgpycrumbs) for surface fitting, interpolation, and structure analysis.


<a id="orgc247628"></a>

### Supported Engines

-   ORCA (**5.x**)
    -   Geometry scan (`OPT`) energy profiles.
    -   Nudged elastic band (NEB) path visualization.
-   eOn
    -   Saddle search parsing for the dimer and Gaussian process regression dimer (GPRD) methods, with their optimizers such as `lbfgs`.
    -   NEB energy profiles and 2D surface projections of the path.
-   Sella
    -   Saddle point optimization result parsing.
-   Trajectory formats
    -   HDF5 trajectories (ChemGP output with pre-computed forces).
    -   Generic ASE-readable formats (extxyz, .traj) for NEB analysis.
-   Multi-segment NEB stitching (`stitch_neb_segments`) for continuous bands across chained minimizations (v1.8+).
-   Metadata-native eOn `.con` frames (prefer JSON metadata energies over `.dat` tables) with typed parser results shared by NEB and single-ended tools.
-   Unit-aware NEB/optimization plot helpers (`convert_neb_values`, centralized strip rendering) consumed by `rgpycrumbs` CLIs.


<a id="orgb030d7a"></a>

## Documentation

Full documentation is at <https://chemparseplot.rgoswami.me>.
This includes:

-   A [quickstart guide](https://chemparseplot.rgoswami.me/quickstart.html)
-   [Tutorials](https://chemparseplot.rgoswami.me/tutorials/index.html) for common workflows
-   [API reference](https://chemparseplot.rgoswami.me/apidocs/index.html)

The map of the site, [llms.txt](https://chemparseplot.rgoswami.me/llms.txt), lists the pages with a line on what each covers, and [llms-full.txt](https://github.com/HaoZeke/chemparseplot/blob/main/llms-full.txt) has the manual in one Markdown file and names what it leaves out.
[AGENTS.md](https://github.com/HaoZeke/chemparseplot/blob/main/AGENTS.md) has the build, test and documentation commands and the mistakes that are easy to make, and [skills/](https://github.com/HaoZeke/chemparseplot/tree/main/skills) has task guides whose scripts CI runs: reading an ORCA scan, adding a parser, and finding the source of a docs page.


<a id="org189ed07"></a>

## Contributing

Contributions are welcome.
See [CONTRIBUTING.md](https://github.com/HaoZeke/chemparseplot/blob/main/CONTRIBUTING.md) for development setup and guidelines, and the [Code of Conduct](https://github.com/HaoZeke/chemparseplot/blob/main/CODE_OF_CONDUCT.md).

For bug reports or questions, open an issue on [GitHub](https://github.com/HaoZeke/chemparseplot/issues).


<a id="org535cdf9"></a>

## Release Process

Versions are derived automatically from Git tags via `hatch-vcs`.
We keep `towncrier` for release-note aggregation and use `cocogitto` (`cog`) for the semantic version/tag step.
The actual publish step is already handled by the tag-triggered `.github/workflows/release.yml` workflow.

    # 1. Run the same checks the tag workflow expects
    uv sync --extra test --extra plot --extra release
    uv run pytest --cov=chemparseplot tests
    uv run ruff check .
    uv run ruff format --check .
    uv run sphinx-build doc/source/ doc/build
    
    # 2. Preview the next semantic version from Conventional Commits
    cog bump --dry-run --auto
    
    # 3. Aggregate towncrier fragments into the changelog
    #    towncrier headings use X.Y.Z, while the git tag stays vX.Y.Z
    uvx towncrier build --version "1.7.1"
    
    # 4. Commit the release notes (historically: release: vX.Y.Z)
    git add CHANGELOG.md doc/release/upcoming_changes
    git commit -m "release: v1.7.1"
    
    # 5. Apply the release tag (hatch-vcs reads the version from tags)
    cog bump --auto
    
    # 6. Push main and tags; CI publishes from the tag
    git push origin main --tags


<a id="orgfb43acb"></a>

# License

`chemparseplot` is released under the Massachusetts Institute of Technology (MIT) License.
This is an academic resource, so **please cite** it as much as possible via:

-   The [Zenodo digital object identifier (DOI)](https://doi.org/10.5281/zenodo.18529752) for general use.
-   The `wailord` paper for ORCA usage.


<a id="org5cea173"></a>

# Acknowledgments

This project builds on work supported by the University of Iceland and the Icelandic Research Fund.
`chemparseplot` relies on [`rgpycrumbs`](https://github.com/HaoZeke/rgpycrumbs) for computational modules.


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> To distinguish it from my other thin-python wrapper projects
