=======================
Contributing Guidelines
=======================


.. contents::


Contributing to chemparseplot
-----------------------------

Thank you for contributing to chemparseplot!

Getting Started
~~~~~~~~~~~~~~~

1. Fork and Clone
^^^^^^^^^^^^^^^^^

.. code:: bash

    git clone https://github.com/your-username/chemparseplot
    cd chemparseplot

2. Set Up Development Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    # Using uv (recommended)
    uv sync --all-extras

    # Or using pip
    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[all,test]"

3. Run Tests
^^^^^^^^^^^^

.. code:: bash

    # Run all tests
    uv run pytest

    # Run specific test categories
    uv run pytest -m pure
    uv run pytest -m neb

Code Style
~~~~~~~~~~

Formatting
^^^^^^^^^^

Code is formatted with ruff:

.. code:: bash

    uv run ruff format .
    uv run ruff check .

Type Hints
^^^^^^^^^^

All public functions must have type hints:

.. code:: python

    def parse_orca_neb(
        basename: str,
        working_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Parse ORCA NEB calculation."""
        ...

Target: 90%+ type hint coverage.

Documentation
^^^^^^^^^^^^^

All public functions must have docstrings:

.. code:: python

    def plot_energy_profile(
        data: dict[str, Any],
        output: Path,
        width: float = 5.37,
    ) -> None:
        """Plot NEB energy profile.

        Parameters
        ----------
        data
            Parsed NEB data from parse_orca_neb()
        output
            Output file path
        width
            Figure width in inches

        Example
        -------
        >>> data = parse_orca_neb("job")
        >>> plot_energy_profile(data, "profile.pdf")
        """
        ...

Pull Request Process
~~~~~~~~~~~~~~~~~~~~

1. Create Branch
^^^^^^^^^^^^^^^^

.. code:: bash

    git checkout -b feature/your-feature-name

2. Make Changes
^^^^^^^^^^^^^^^

- Write tests for new features

- Update documentation

- Follow code style guidelines

3. Run CI Checks Locally
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    # Run tests
    uv run pytest

    # Check formatting
    uv run ruff format --check .
    uv run ruff check .

    # Test tutorials
    uv run python3 /tmp/test_tutorial.py

4. Submit PR
^^^^^^^^^^^^

1. Push to your fork

2. Create PR on GitHub

3. Fill out PR template

4. Wait for CI checks

5. Address reviewer comments

Adding New Parsers
~~~~~~~~~~~~~~~~~~

Directory Structure
^^^^^^^^^^^^^^^^^^^

.. code:: text

    chemparseplot/parse/<code>/
    |-- __init__.py
    |-- <method>.py      # Main parser
    `-- test_<method>.py # Tests

Parser Requirements
^^^^^^^^^^^^^^^^^^^

1. Return compatible data format:

.. code:: python

    {
        "energies": np.array([...]),
        "rmsd_r": np.array([...]),
        "rmsd_p": np.array([...]),
        "converged": bool,
        "n_images": int,
    }

1. Use lazy imports for optional dependencies:

.. code:: python

    from rgpycrumbs._aux import ensure_import

    def _get_optional_dep():
        global _dep
        if _dep is None:
            _dep = ensure_import("optional_package")
        return _dep

1. Add tests with mock data

2. Update documentation

Adding New Plot Types
~~~~~~~~~~~~~~~~~~~~~

Directory Structure
^^^^^^^^^^^^^^^^^^^

.. code:: text

    chemparseplot/plot/
    |-- <plot_type>.py      # Plotting function
    `-- test_<plot_type>.py # Tests

Plot Requirements
^^^^^^^^^^^^^^^^^

1. Accept standard data format

2. Use theme system:

.. code:: python

    from chemparseplot.plot.theme import get_theme
    theme = get_theme("ruhi")

1. Support customization (width, height, dpi)

2. Add to documentation

Testing Requirements
~~~~~~~~~~~~~~~~~~~~

Test Categories
^^^^^^^^^^^^^^^

- ``@pytest.mark pure`` - Pure Python tests (no optional deps)

- ``@pytest.mark neb`` - NEB-related tests

- ``@pytest.mark plot`` - Plotting tests

Test Coverage
^^^^^^^^^^^^^

Target: 80%+ coverage for new code

.. code:: bash

    uv run pytest --cov=chemparseplot

Documentation Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~

New Features Must Have
^^^^^^^^^^^^^^^^^^^^^^

1. Tutorial (if user-facing)

2. API reference

3. How-to guide (if applicable)

4. FAQ entry (if addresses common issue)

Documentation Style
^^^^^^^^^^^^^^^^^^^

- Org-mode format

- ASCII-only (no Unicode)

- Working code examples

- Cross-references with ``[[file:...][...]]``

Reporting Bugs
~~~~~~~~~~~~~~

Create issue on GitHub with:

- chemparseplot version

- Python version

- ORCA/eOn version

- Minimal reproducible example

- Full error message

- Expected vs actual behavior

Requesting Features
~~~~~~~~~~~~~~~~~~~

Create issue on GitHub with:

- Feature description

- Use case

- Example input/output

- Priority (nice-to-have vs critical)

Code of Conduct
~~~~~~~~~~~~~~~

Please be respectful and inclusive in all interactions.
See `Code of Conduct <../code_of_conduct.rst>`_.

Questions?
~~~~~~~~~~

- `FAQ <../howto/faq.rst>`_ - Common questions

- `Troubleshooting <../howto/troubleshooting.rst>`_ - Common problems

- `GitHub Discussions <https://github.com/HaoZeke/chemparseplot/discussions>`_ - General questions

See Also
~~~~~~~~

- `Best Practices <best_practices.rst>`_

- `How-to Guides <../howto/index.rst>`_

- `Reference <../reference/index.rst>`_
