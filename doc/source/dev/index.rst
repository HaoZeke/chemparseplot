=======================
Developer Documentation
=======================


.. contents::


Developer Documentation
-----------------------

Technical documentation for developers working on chemparseplot.

Architecture
~~~~~~~~~~~~

.. table::

    +------------------------------------------------------+--------------------------------------------------------+
    | Document                                             | Description                                            |
    +======================================================+========================================================+
    | `Parsing Architecture <parsing_architecture.rst>`_   | How parsers extract data from quantum chemistry output |
    +------------------------------------------------------+--------------------------------------------------------+
    | `Plotting Architecture <plotting_architecture.rst>`_ | Separation of data parsing and visualization           |
    +------------------------------------------------------+--------------------------------------------------------+
    | `OPI Integration <opi_integration.rst>`_             | ORCA Python Interface integration design               |
    +------------------------------------------------------+--------------------------------------------------------+

Patterns
~~~~~~~~

.. table::

    +--------------------------------------------------+-----------------------------------------------+
    | Pattern                                          | Description                                   |
    +==================================================+===============================================+
    | `Lazy Imports <lazy_imports.rst>`_               | Defer optional dependency loading             |
    +--------------------------------------------------+-----------------------------------------------+
    | `Unified Data Format <unified_data_format.rst>`_ | Compatible data format across different codes |
    +--------------------------------------------------+-----------------------------------------------+
    | `Diataxis Documentation <diataxis_docs.rst>`_    | Documentation structure and organization      |
    +--------------------------------------------------+-----------------------------------------------+

Contributing
~~~~~~~~~~~~

- `Contributing Guidelines <../contributing.rst>`_

- `Release Process <../release.rst>`_

- `Testing Guidelines <../testing.rst>`_

Quick Start for Developers
--------------------------

Setup\*\*
~~~~~~~~~

.. code:: bash

    git clone https://github.com/HaoZeke/chemparseplot
    cd chemparseplot
    uv sync --all-extras

Run Tests\*\*
~~~~~~~~~~~~~

.. code:: bash

    uv run pytest -m pure  # Pure Python tests
    uv run pytest -m neb   # NEB-related tests

Add New Parser\*\*
~~~~~~~~~~~~~~~~~~

1. Create ``chemparseplot/parse/<code>/<method>.py``

2. Return data compatible with existing format

3. Add tests in ``tests/parse/<code>/``

4. Update documentation

See `Parsing Architecture <parsing_architecture.rst>`_ for details.

Related
-------

- `Main Documentation <../index.rst>`_

- `Tutorials <../tutorials/index.rst>`_

- `GitHub Repository <https://github.com/HaoZeke/chemparseplot>`_
