==========================
Frequently Asked Questions
==========================


.. contents::


Frequently Asked Questions
--------------------------

Common questions about chemparseplot with answers and solutions.

Installation
~~~~~~~~~~~~

How do I install chemparseplot?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    # Basic installation
    pip install chemparseplot

    # With plotting dependencies
    pip install "chemparseplot[plot]"

    # With all optional dependencies
    pip install "chemparseplot[all]"

See: `Installation Guide <install.rst>`_

How do I install with conda/pixi?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    # Using pixi
    pixi add chemparseplot

    # Using conda (via conda-forge)
    conda install -c conda-forge chemparseplot

I get "ModuleNotFoundError: No module named 'chemparseplot'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure you installed in the correct Python environment:

.. code:: bash

    # Check Python path
    python -c "import sys; print(sys.executable)"

    # Install in correct environment
    /path/to/python -m pip install chemparseplot

Usage
~~~~~

How do I parse ORCA NEB output?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from pathlib import Path
    from chemparseplot.parse.orca.neb import parse_orca_neb

    data = parse_orca_neb("job", working_dir=Path("calculation"))
    print(f"Energies: {data['energies']}")
    print(f"Barrier: {data['barrier_forward']:.2f} eV")

See: `ORCA NEB Tutorial <../tutorials/orca_neb.rst>`_

How do I create an energy profile plot?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from chemparseplot.plot.neb import plot_orca_neb_energy_profile

    plot_orca_neb_energy_profile(data, "profile.pdf")

See: `ORCA NEB Tutorial <../tutorials/orca_neb.rst>`_

How do I parse eOn NEB output?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from chemparseplot.parse.eon.neb import aggregate_neb_landscape_data

    data = aggregate_neb_landscape_data(dat_paths, con_paths, y_col=2)

See: `Parse eOn NEB Calculations <parse_eon_neb.rst>`_

Errors
~~~~~~

"ImportError: No module named 'opi'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OPI (ORCA Python Interface) is required for ORCA 6.1+ parsing:

.. code:: bash

    pip install orca-pi

Or use legacy parser for ORCA < 6.1:

.. code:: python

    from chemparseplot.parse.orca.neb import parse_orca_neb_fallback

    data = parse_orca_neb_fallback("job", Path("calc"))

"RMSD coordinates required for landscape plot"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Landscape plots require geometry output. Ensure your calculation includes:

.. code:: text

    %output
      Print[P_Molden] true
      Print[MOs] true
    end

"Contour levels must be increasing"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Energy data may have numerical issues. Check for:

- Duplicate energy values

- NaN or Inf values

- Very small energy differences

Fix by filtering or smoothing data.

Performance
~~~~~~~~~~~

How can I speed up batch plotting?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use parallel processing:

.. code:: bash

    rgpycrumbs chemgp batch -c config.toml -j 4

The ``-j 4`` flag uses 4 parallel workers.

Why is landscape plotting slow?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Surface fitting is computationally expensive. Options:

- Use fewer points (downsample)

- Opt-in ``auto_thin`` / ``SurfaceFitConfig`` (default off) to cap fit size

- Use simpler interpolation method (RBF instead of GP)

- Use Nystrom approximation for large datasets

How do I thin dense minimization movies without changing defaults?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``SurfaceFitConfig`` (TOML key names match rgpycrumbs plot config):

.. code:: python

    from chemparseplot.plot.neb import SurfaceFitConfig

    cfg = SurfaceFitConfig.from_mapping({"auto_thin": True, "max_surface_points": 64})

Default ``auto_thin`` is ``False`` so existing scripts keep full-cloud fits.

Compatibility
~~~~~~~~~~~~~

Which ORCA versions are supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ORCA 6.1+: Full support via OPI

- ORCA < 6.1: Limited support via .interp file parsing

Which Python versions are supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Python 3.10, 3.11, 3.12

Does chemparseplot work on Windows?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes, but some features may have limited support:

- Plotting works on all platforms

- OPI requires ORCA 6.1+ (Windows available)

- Some parallel features work best on Linux/macOS

Development
~~~~~~~~~~~

How do I contribute?
^^^^^^^^^^^^^^^^^^^^

See: `Contributing Guidelines <../dev/contributing.rst>`_

How do I report a bug?
^^^^^^^^^^^^^^^^^^^^^^

Create an issue on GitHub: `GitHub Issues <https://github.com/HaoZeke/chemparseplot/issues>`_

Include:

- chemparseplot version

- Python version

- ORCA/eOn version

- Minimal reproducible example

- Error message

How do I request a feature?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create an issue on GitHub with:

- Feature description

- Use case

- Example input/output

- Priority (nice-to-have vs critical)

See Also
~~~~~~~~

- `Glossary <../reference/glossary.rst>`_ - Definitions of terms

- `Installation Guide <install.rst>`_ - Detailed installation

- `Troubleshooting <troubleshooting.rst>`_ - Common problems and solutions

- `Tutorials <../tutorials/index.rst>`_ - Learn how to use chemparseplot
