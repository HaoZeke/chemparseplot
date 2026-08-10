=====================
Architecture Overview
=====================


.. contents::


Ecosystem position
------------------

.. mermaid::

   flowchart TB
     subgraph rgpkgs["rgpkgs suite"]
       RGP[rgpycrumbs]
       CPP[chemparseplot]
       PYC[pychum]
     end
     ENG[eOn / ORCA / Sella] --> CPP
     CPP -->|parse + plot APIs| USER[Notebooks / docs recipes]
     RGP -->|CLI plt-* + surfaces| USER
     RGP -->|plot.toml config| CPP
     PYC -->|inputs| ENG

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: chemparseplot
      Library: parsers, landscape APIs, ``SurfaceFitConfig``.

   .. grid-item-card:: rgpycrumbs
      CLI dispatch, GP surfaces, TOML plot config.

Architecture Overview
---------------------

This document explains the high-level architecture of chemparseplot.

System Overview
~~~~~~~~~~~~~~~

.. code:: text

                    chemparseplot Architecture
                    ==========================

    +------------------+     +------------------+     +------------------+
    |   Input Files    |     |  chemparseplot   |     |   Output Files   |
    +------------------+     +------------------+     +------------------+
    |                  |     |                  |     |                  |
    | ORCA output      |---->|   PARSERS        |     | PDF figures      |
    | eOn output       |     |   - orca/        |---->| HDF5 data        |
    | Sella output     |     |   - eon/         |     | JSON data        |
    | ChemGP HDF5      |     |   - sella/       |     |                  |
    |                  |     |   - trajectory/  |     |                  |
    +------------------+     +--------+---------+     +------------------+
                                      |
                                      v
                             +--------+---------+
                             |   PLOTTING       |
                             |   - chemgp.py    |
                             |   - neb.py       |
                             |   - geomscan.py  |
                             +--------+---------+
                                      |
                                      v
                             +--------+---------+
                             |   UTILITIES      |
                             |   - units.py     |
                             |   - converter.py |
                             |   - patterns.py  |
                             +------------------+

Module Structure
~~~~~~~~~~~~~~~~

.. code:: text

    chemparseplot/
    |-- __init__.py          # Package initialization
    |-- units.py             # Pint unit registry
    |-- util.py              # Utilities with lazy imports
    |-- parse/               # PARSERS
    |   |-- __init__.py
    |   |-- orca/            # ORCA output parsing
    |   |   |-- __init__.py
    |   |   |-- geomscan.py
    |   |   `-- neb/
    |   |       |-- __init__.py
    |   |       |-- interp.py
    |   |       `-- opi_parser.py  # OPI-based parser
    |   |-- eon/             # eOn output parsing
    |   |   |-- __init__.py
    |   |   |-- neb.py
    |   |   `-- saddle_search.py
    |   |-- sella/           # Sella output parsing
    |   |   `-- saddle_search.py
    |   |-- trajectory/      # Trajectory parsing
    |   |   |-- __init__.py
    |   |   |-- hdf5.py
    |   |   `-- neb.py
    |   |-- file_.py         # File utilities
    |   |-- patterns.py      # Regex patterns
    |   |-- converter.py     # Unit conversions
    |   `-- neb_utils.py     # NEB utilities
    `-- plot/                # PLOTTING
        |-- __init__.py
        |-- theme.py         # Plot themes (ruhi, etc.)
        |-- structs.py       # Structure rendering
        |-- chemgp.py        # ChemGP plotting
        |-- neb.py           # NEB plotting
        `-- geomscan.py      # Geometry scan plotting

Data Flow
~~~~~~~~~

.. code:: text

    ORCA/eOn Output
         |
         v
    +---------+
    | PARSER  |  (parse/orca/, parse/eon/)
    +----+----+
         |
         v
    Structured Data (dict/DataFrame)
         |
         v
    +---------+
    | PLOTTER |  (plot/neb.py, plot/chemgp.py)
    +----+----+
         |
         v
    PDF/PNG Figure

Key Design Decisions
~~~~~~~~~~~~~~~~~~~~

Lazy Imports
^^^^^^^^^^^^

Optional dependencies loaded on first use:

.. code:: python

    from rgpycrumbs._aux import ensure_import

    def _get_opi():
        global _opi
        if _opi is None:
            _opi = ensure_import("opi.output.core")
        return _opi

Benefits:

- Smaller installation

- Graceful degradation

- Auto-install option

Unified Data Format
^^^^^^^^^^^^^^^^^^^

All parsers return compatible data structures:

.. code:: python

    {
        "energies": np.array([...]),  # eV
        "rmsd_r": np.array([...]),    # Angstrom
        "rmsd_p": np.array([...]),    # Angstrom
        "grad_r": np.array([...]),    # eV/Angstrom
        "grad_p": np.array([...]),    # eV/Angstrom
        "converged": bool,
        "n_images": int,
    }

Benefits:

- Same plotting code for all codes

- Easy to add new parsers

- Consistent API

Separation of Concerns
^^^^^^^^^^^^^^^^^^^^^^

- Parsers: Extract data from output files

- Plotters: Create visualizations

- Utilities: Unit handling, conversions

Benefits:

- Testable components

- Reusable code

- Clear boundaries

Dependencies
~~~~~~~~~~~~

.. code:: text

    Required:
    - numpy
    - pint (units)

    Optional (lazy-loaded):
    - matplotlib (plotting)
    - cmcrameri (colormaps)
    - ase (structure handling)
    - polars (dataframes)
    - h5py (HDF5)
    - orca-pi (ORCA 6.1+ parsing)
    - jax (GP surface fitting, via rgpycrumbs)

Integration with rgpycrumbs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

chemparseplot delegates heavy computation to rgpycrumbs:

.. code:: text

    chemparseplot          rgpycrumbs
    =============          ==========
    parse.orca.neb  ---->  (none)
    parse.eon.neb   ---->  (none)
    plot.neb        ---->  surfaces (GP fitting)
    plot.chemgp     ---->  interpolation

See Also
~~~~~~~~

- `Reference Documentation <../reference/index.rst>`_

- `Developer Documentation <../dev/index.rst>`_

- `Lazy Import Pattern <lazy_imports.rst>`_

- `Glossary <../reference/glossary.rst>`_
