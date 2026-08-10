======================
ORCA NEB API Reference
======================


.. contents::


Module: ``chemparseplot.parse.orca.neb``
----------------------------------------

``parse_orca_neb(basename, working_dir=None)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parse ORCA NEB calculation using OPI (ORCA 6.1+).

Parameters
^^^^^^^^^^

.. table::

    +-----------------+----------+----------+---------------------------------------+
    | Parameter       | Type     | Default  | Description                           |
    +=================+==========+==========+=======================================+
    | ``basename``    | ``str``  | -        | ORCA job basename (without extension) |
    +-----------------+----------+----------+---------------------------------------+
    | ``working_dir`` | ``Path`` | ``None`` | Working directory (default: cwd)      |
    +-----------------+----------+----------+---------------------------------------+

Returns
^^^^^^^

``dict`` - Structured NEB data (see `Data Format <../howto/parse_orca_neb.rst>`_)

Raises
^^^^^^

.. table::

    +-----------------------+-----------------------------+
    | Exception             | Condition                   |
    +=======================+=============================+
    | ``ImportError``       | OPI (orca-pi) not installed |
    +-----------------------+-----------------------------+
    | ``FileNotFoundError`` | ORCA output files not found |
    +-----------------------+-----------------------------+

Example
^^^^^^^

.. code:: python

    from pathlib import Path
    from chemparseplot.parse.orca.neb import parse_orca_neb

    data = parse_orca_neb("job", working_dir=Path("calculation"))
    print(f"Energies: {data['energies']}")
    print(f"Barrier: {data['barrier_forward']:.2f} eV")

Notes
^^^^^

- Requires ORCA 6.1+ with JSON output

- Uses lazy loading via ``ensure_import()``

- OPI package installed on first use if ``RGPYCRUMBS_AUTO_DEPS=1``

``parse_orca_neb_fallback(basename, working_dir=None)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parse ORCA NEB using legacy .interp file parsing (ORCA < 6.1).

Parameters
^^^^^^^^^^

Same as ``parse_orca_neb()``

Returns
^^^^^^^

``dict`` or ``None`` - NEB data if successful, ``None`` if parsing fails

Notes
^^^^^

- Parses ``.interp`` files from ORCA NEB calculations

- Less robust than OPI-based parsing

- Use for ORCA versions < 6.1

``HAS_OPI``
~~~~~~~~~~~

``bool`` - True if OPI package is available (lazy-loaded).

Module: ``chemparseplot.plot.neb``
----------------------------------

``plot_orca_neb_energy_profile(neb_data, output, ...)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot ORCA NEB energy profile using eOn-style plotting.

Parameters
^^^^^^^^^^

.. table::

    +---------------+---------------------------------+---------------+----------------------------------------+
    | Parameter     | Type                            | Default       | Description                            |
    +===============+=================================+===============+========================================+
    | ``neb_data``  | ``OrcaNebResult`` / ``Mapping`` | -             | Typed result from ``parse_orca_neb()`` |
    +---------------+---------------------------------+---------------+----------------------------------------+
    | ``output``    | ``Path``                        | -             | Output file path                       |
    +---------------+---------------------------------+---------------+----------------------------------------+
    | ``width``     | ``float``                       | ``5.37``      | Figure width (inches)                  |
    +---------------+---------------------------------+---------------+----------------------------------------+
    | ``height``    | ``float``                       | ``5.37``      | Figure height (inches)                 |
    +---------------+---------------------------------+---------------+----------------------------------------+
    | ``dpi``       | ``int``                         | ``200``       | Output resolution                      |
    +---------------+---------------------------------+---------------+----------------------------------------+
    | ``method``    | ``str``                         | ``'hermite'`` | Interpolation method                   |
    +---------------+---------------------------------+---------------+----------------------------------------+
    | ``smoothing`` | ``Any``                         | ``None``      | Smoothing parameters                   |
    +---------------+---------------------------------+---------------+----------------------------------------+

Returns
^^^^^^^

``None`` - Saves plot to ``output``

Example
^^^^^^^

.. code:: python

    from chemparseplot.parse.orca.neb import parse_orca_neb
    from chemparseplot.plot.neb import plot_orca_neb_energy_profile

    data = parse_orca_neb("job")
    plot_orca_neb_energy_profile(data, "profile.pdf")

``plot_orca_neb_landscape(neb_data, output, ...)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot ORCA NEB 2D landscape using eOn-style plotting.

Parameters
^^^^^^^^^^

.. table::

    +------------------+---------------------------------+-------------------+----------------------------------------+
    | Parameter        | Type                            | Default           | Description                            |
    +==================+=================================+===================+========================================+
    | ``neb_data``     | ``OrcaNebResult`` / ``Mapping`` | -                 | Typed result from ``parse_orca_neb()`` |
    +------------------+---------------------------------+-------------------+----------------------------------------+
    | ``output``       | ``Path``                        | -                 | Output file path                       |
    +------------------+---------------------------------+-------------------+----------------------------------------+
    | ``width``        | ``float``                       | ``5.37``          | Figure width (inches)                  |
    +------------------+---------------------------------+-------------------+----------------------------------------+
    | ``height``       | ``float``                       | ``5.37``          | Figure height (inches)                 |
    +------------------+---------------------------------+-------------------+----------------------------------------+
    | ``dpi``          | ``int``                         | ``200``           | Output resolution                      |
    +------------------+---------------------------------+-------------------+----------------------------------------+
    | ``method``       | ``str``                         | ``'grad_matern'`` | Surface interpolation                  |
    +------------------+---------------------------------+-------------------+----------------------------------------+
    | ``project_path`` | ``bool``                        | ``True``          | Project to reaction valley             |
    +------------------+---------------------------------+-------------------+----------------------------------------+

Returns
^^^^^^^

``None`` - Saves plot to ``output``

Raises
^^^^^^

.. table::

    +----------------+--------------------------------+
    | Exception      | Condition                      |
    +================+================================+
    | ``ValueError`` | RMSD coordinates not available |
    +----------------+--------------------------------+

Notes
^^^^^

- Requires ``rmsd_r`` and ``rmsd_p`` in ``neb_data``

- Uses same plotting functions as eOn NEB for consistency

See Also
--------

- `Tutorial: ORCA NEB Parsing and Plotting <../tutorials/orca_neb.rst>`_

- `How-to: Parse ORCA NEB Calculations <../howto/parse_orca_neb.rst>`_

- `Plotting API Reference <plotting_api.rst>`_
