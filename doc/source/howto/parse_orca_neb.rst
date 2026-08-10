===================================
How-to: Parse ORCA NEB Calculations
===================================



Problem
-------

You have an ORCA NEB calculation and want to extract energies, geometries, and barriers.

Solution
--------

For ORCA 6.1+ (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use OPI-based parser:

.. code:: python

    from pathlib import Path
    from chemparseplot.parse.orca.neb import parse_orca_neb

    data = parse_orca_neb("job", working_dir=Path("calculation"))

    # Access parsed data
    energies = data["energies"]  # eV
    n_images = data["n_images"]
    barrier = data["barrier_forward"]
    converged = data["converged"]

For ORCA < 6.1
~~~~~~~~~~~~~~

Use legacy .interp file parser:

.. code:: python

    from pathlib import Path
    from chemparseplot.parse.orca.neb import parse_orca_neb_fallback

    data = parse_orca_neb_fallback("job", Path("calculation"))

    if data is None:
        print("Parsing failed - check .interp file exists")

Automatic Fallback
~~~~~~~~~~~~~~~~~~

Let the parser choose automatically:

.. code:: python

    from chemparseplot.parse.orca.neb import parse_orca_neb, HAS_OPI

    if HAS_OPI:
        data = parse_orca_neb("job", Path("calc"))
    else:
        data = parse_orca_neb_fallback("job", Path("calc"))

Data Format
-----------

Returned dictionary contains:

.. table::

    +---------------------+----------------------------+----------------------------------+
    | Key                 | Type                       | Description                      |
    +=====================+============================+==================================+
    | ``energies``        | ``np.ndarray``             | Energies in eV                   |
    +---------------------+----------------------------+----------------------------------+
    | ``rmsd_r``          | ``np.ndarray`` or ``None`` | RMSD from reactant               |
    +---------------------+----------------------------+----------------------------------+
    | ``rmsd_p``          | ``np.ndarray`` or ``None`` | RMSD from product                |
    +---------------------+----------------------------+----------------------------------+
    | ``grad_r``          | ``np.ndarray`` or ``None`` | Gradients (RMSD-r)               |
    +---------------------+----------------------------+----------------------------------+
    | ``grad_p``          | ``np.ndarray`` or ``None`` | Gradients (RMSD-p)               |
    +---------------------+----------------------------+----------------------------------+
    | ``forces``          | ``list`` or ``None``       | Force vectors                    |
    +---------------------+----------------------------+----------------------------------+
    | ``converged``       | ``bool``                   | Normal termination               |
    +---------------------+----------------------------+----------------------------------+
    | ``n_images``        | ``int``                    | Number of NEB images             |
    +---------------------+----------------------------+----------------------------------+
    | ``barrier_forward`` | ``float`` or ``None``      | Forward barrier (eV)             |
    +---------------------+----------------------------+----------------------------------+
    | ``barrier_reverse`` | ``float`` or ``None``      | Reverse barrier (eV)             |
    +---------------------+----------------------------+----------------------------------+
    | ``source``          | ``str``                    | ``'opi'`` or ``'legacy_interp'`` |
    +---------------------+----------------------------+----------------------------------+
    | ``orca_version``    | ``str``                    | ORCA version string              |
    +---------------------+----------------------------+----------------------------------+

See Also
--------

- `Tutorial: ORCA NEB Parsing and Plotting <../tutorials/orca_neb.rst>`_

- `ORCA NEB API Reference <../reference/orca_neb_api.rst>`_

- `How-to: Parse eOn NEB Calculations <parse_eon_neb.rst>`_
