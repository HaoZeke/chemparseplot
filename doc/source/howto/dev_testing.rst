====================================
How-to: Run Tests and Check Coverage
====================================



Problem
-------

You want to run tests and check coverage across all optional dependency sets.

Quick Start
-----------

Single Environment (uv)
~~~~~~~~~~~~~~~~~~~~~~~

For basic development without the heavier optional plotting stacks:

.. code:: shell

    cd chemparseplot
    uv sync --extra test
    uv run pytest tests/ --ignore=tests/tutorials

This installs the base test floor used by the main parser and NEB suites. The remaining coverage still needs the heavier optional stacks: jax, pandas, and plotnine.

Full Coverage (pixi)
~~~~~~~~~~~~~~~~~~~~

pixi manages multiple environments with different optional deps. Each env runs its test subset, then results combine:

.. code:: shell

    pixi run cov

This runs:

.. table::

    +-------------+--------------------------------+-------------------------------------------+
    | Environment | Deps                           | What it tests                             |
    +=============+================================+===========================================+
    | test        | ase, polars, scipy, matplotlib | Core parsers, projection, NEB, trajectory |
    +-------------+--------------------------------+-------------------------------------------+
    | plot        | + jax, cmcrameri               | Surface fitting, landscape rendering      |
    +-------------+--------------------------------+-------------------------------------------+
    | chemgp      | + pandas, plotnine             | ChemGP HDF5/JSONL parsers, plotnine plots |
    +-------------+--------------------------------+-------------------------------------------+
    | plumed      | + pandas                       | PLUMED FES parsing and reconstruction     |
    +-------------+--------------------------------+-------------------------------------------+

The ``cov`` task cleans stale files, runs all four env tasks, then calls ``coverage combine`` and ``coverage report --fail-under=90``.

Individual Environments
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

    pixi run -e test test-base     # base tests only
    pixi run -e plot test-plot     # with jax + cmcrameri
    pixi run -e chemgp test-chemgp # with pandas + plotnine
    pixi run -e plumed test-plumed # plumed tests only

Coverage Report
~~~~~~~~~~~~~~~

After any test run:

.. code:: shell

    uv run coverage report --show-missing

Or for HTML:

.. code:: shell

    uv run coverage html
    # open htmlcov/index.html

How Coverage Combine Works
--------------------------

1. Each pixi env task runs ``pytest --cov`` which writes ``.coverage``

2. The task renames it to ``.coverage.<env>`` (e.g. ``.coverage.test``)

3. ``coverage combine`` merges all ``.coverage.*`` into one ``.coverage``

4. ``coverage report`` reads the combined file

The ``[tool.coverage.run]`` in ``pyproject.toml`` has ``parallel = true`` and ``source_pkgs = ["chemparseplot"]``. The ``[tool.coverage.paths]`` maps different install prefixes back to the source tree.

Writing Tests
-------------

Test Markers
~~~~~~~~~~~~

.. code:: python

    import pytest

    @pytest.mark.neb       # needs ase, polars, h5py
    @pytest.mark.pure      # needs only numpy

Smoke Tests for Plot Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot functions are tested by creating a figure, calling the function, and checking axes properties:

.. code:: python

    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    def test_some_plot():
        fig, ax = plt.subplots()
        some_plot_function(ax, data)
        assert len(ax.lines) > 0
        plt.close(fig)

Quality Guidelines
------------------

- New modules must have tests in the same PR

- Coverage threshold: 90% (enforced by ``pixi run cov``)

- Plot functions: at least smoke tests (figure creation + basic assertions)

- Parsers: synthetic input data + output schema validation

- Keep base-env tests free of the heavier optional stacks (jax, pandas, plotnine)

See Also
--------

- `Plotting API Reference <../reference/plotting_api.rst>`_

- `Contributing Guide <../../CONTRIBUTING.md>`_
