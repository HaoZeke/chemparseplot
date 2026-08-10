==================
Installation Guide
==================


.. contents::


Installation Guide
------------------

Install chemparseplot using your preferred package manager.

Requirements
~~~~~~~~~~~~

- Python 3.10, 3.11, or 3.12

- pip, uv, pixi, or conda

Quick Install
~~~~~~~~~~~~~

Using pip
^^^^^^^^^

.. code:: bash

    # Basic installation
    pip install chemparseplot

    # With plotting support
    pip install "chemparseplot[plot]"

    # With all optional dependencies
    pip install "chemparseplot[all]"

Using uv
^^^^^^^^

.. code:: bash

    uv add chemparseplot

Using pixi
^^^^^^^^^^

.. code:: bash

    pixi add chemparseplot

Using conda
^^^^^^^^^^^

.. code:: bash

    conda install -c conda-forge chemparseplot

Platform-Specific Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linux
^^^^^

No special requirements. All features work on Linux.

.. code:: bash

    # Recommended: use uv for fastest installation
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv add chemparseplot

macOS
^^^^^

No special requirements. All features work on macOS.

.. code:: bash

    # Using Homebrew Python
    brew install python
    pip install chemparseplot

Windows
^^^^^^^

Most features work on Windows. Note:

- OPI requires ORCA 6.1+ for Windows

- Some parallel features work best with WSL2

.. code:: bash

    # Using pip
    pip install chemparseplot

    # Or using conda
    conda install -c conda-forge chemparseplot

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

Plotting (Recommended)
^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    pip install "chemparseplot[plot]"

Includes:

- matplotlib

- cmcrameri (scientific colormaps)

NEB Processing
^^^^^^^^^^^^^^

.. code:: bash

    pip install "chemparseplot[neb]"

Includes:

- ase (Atomic Simulation Environment)

- h5py (HDF5 support)

- polars (fast dataframes)

ORCA 6.1+ Support
^^^^^^^^^^^^^^^^^

.. code:: bash

    pip install orca-pi

Official ORCA Python Interface for parsing ORCA 6.1+ JSON output.

All Dependencies
^^^^^^^^^^^^^^^^

.. code:: bash

    pip install "chemparseplot[all]"

Verification
~~~~~~~~~~~~

Test your installation:

.. code:: bash

    python -c "import chemparseplot; print(chemparseplot.__version__)"

Expected output: version number (e.g., "0.2.0")

Troubleshooting
~~~~~~~~~~~~~~~

"ModuleNotFoundError: No module named 'chemparseplot'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure you installed in the correct Python environment:

.. code:: bash

    # Check which Python is being used
    which python
    python --version

    # Install in correct environment
    /path/to/python -m pip install chemparseplot

"ImportError: No module named 'matplotlib'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install plotting dependencies:

.. code:: bash

    pip install "chemparseplot[plot]"

"ImportError: No module named 'opi'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install OPI for ORCA 6.1+ support:

.. code:: bash

    pip install orca-pi

Or use legacy parser for ORCA < 6.1.

Installation fails with permission error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use user installation:

.. code:: bash

    pip install --user chemparseplot

Or use virtual environment:

.. code:: bash

    python -m venv .venv
    source .venv/bin/activate
    pip install chemparseplot

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For contributing to chemparseplot:

.. code:: bash

    # Clone repository
    git clone https://github.com/HaoZeke/chemparseplot
    cd chemparseplot

    # Install in editable mode with all dependencies
    uv sync --all-extras

    # Run tests
    uv run pytest

See: `Contributing Guidelines <../dev/contributing.rst>`_

Next Steps
~~~~~~~~~~

- `Tutorials <../tutorials/index.rst>`_ - Learn how to use chemparseplot

- `FAQ <faq.rst>`_ - Common questions

- `Reference <../reference/index.rst>`_ - API documentation

See Also
~~~~~~~~

- `Official Documentation <https://chemparseplot.rgoswami.me>`_

- `GitHub Repository <https://github.com/HaoZeke/chemparseplot>`_

- `Glossary <../reference/glossary.rst>`_ - Terms and definitions
