==================================
How-to: Parse eOn NEB Calculations
==================================



Problem
-------

You have eOn NEB output (CON/DAT file pairs) and want to extract energies, RMSD coordinates, and gradients for plotting.

Solution
--------

Load Structures and Compute RMSD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from pathlib import Path
    from chemparseplot.parse.eon.neb import (
        NebOverlayStructure,
        load_structures_and_calculate_additional_rmsd,
    )

    bundle = load_structures_and_calculate_additional_rmsd(
        con_file=Path("neb.con"),
        additional_con=[
            (Path("saddle.con"), "Saddle Point"),
            (Path("minima.con"), "Local Minimum"),
        ],
        ira_kmax=14.0,
        sp_file=Path("sp.con"),  # explicit saddle point (optional)
    )

This returns:

- ``bundle.atoms_list``: ASE Atoms objects from the main trajectory

- ``bundle.additional_structures``: list of ``NebOverlayStructure`` records

- ``bundle.saddle_point``: explicit saddle-point overlay, or ``None``

.. code:: python

    for overlay in bundle.additional_structures:
        print(overlay.label, overlay.r, overlay.p)

    if bundle.saddle_point is not None:
        print(
            "Explicit saddle:",
            bundle.saddle_point.label,
            bundle.saddle_point.r,
            bundle.saddle_point.p,
        )

Landscape-style RMSD coordinates require IRA. If ``ira_mod`` is unavailable,
``load_structures_and_calculate_additional_rmsd()`` and other landscape helpers
now raise ``ImportError`` instead of silently switching to order-dependent
alignment.

Aggregate Landscape Data from Multiple Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

eOn NEB runs produce one DAT/CON pair per optimization step. Aggregate them into a single Polars DataFrame:

.. code:: python

    from pathlib import Path
    import ira_mod
    from chemparseplot.parse.eon.neb import aggregate_neb_landscape_data

    dat_paths = sorted(Path("neb_run").glob("*.dat"))
    con_paths = sorted(Path("neb_run").glob("*.con"))

    df = aggregate_neb_landscape_data(
        all_dat_paths=dat_paths,
        all_con_paths=con_paths,
        y_data_column=1,          # column index for energy in DAT files
        ira_instance=ira_mod.IRA(),  # provide a real IRA backend
        ira_kmax=14.0,
    )

    print(df.columns)  # ['r', 'p', 'grad_r', 'grad_p', 'z', 'step']
    print(df.shape)

Use Parquet Caching
~~~~~~~~~~~~~~~~~~~

Landscape aggregation is expensive (RMSD calculation per image). Cache the result:

.. code:: python

    df = aggregate_neb_landscape_data(
        all_dat_paths=dat_paths,
        all_con_paths=con_paths,
        y_data_column=1,
        ira_instance=ira_mod.IRA(),
        cache_file=Path("landscape_cache.parquet"),
        force_recompute=False,    # set True to bypass cache
        ira_kmax=14.0,
    )

On subsequent runs, the function loads from ``landscape_cache.parquet`` and validates the schema (checks for ``p`` and ``grad_r`` columns).

Augment with External NEB Paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add data from other NEB runs to improve landscape surface fits:

.. code:: python

    from ase.io import read as ase_read

    ref_atoms = ase_read("reactant.con")
    prod_atoms = ase_read("product.con")

    df = aggregate_neb_landscape_data(
        all_dat_paths=dat_paths,
        all_con_paths=con_paths,
        y_data_column=1,
        ira_instance=ira_mod.IRA(),
        cache_file=Path("landscape_cache.parquet"),
        augment_dat="extra_run/*.dat",
        augment_con="extra_run/*.con",
        ref_atoms=ref_atoms,
        prod_atoms=prod_atoms,
    )

Augmented paths get ``step``-1= in the DataFrame, distinguishing them from the primary path.

Compute 1D Profile RMSD
~~~~~~~~~~~~~~~~~~~~~~~

For simple energy profiles (no landscape), compute RMSD from reactant only:

.. code:: python

    from chemparseplot.parse.eon.neb import compute_profile_rmsd

    atoms_list = ase_read("neb.con", index=":")
    df_profile = compute_profile_rmsd(
        atoms_list,
        cache_file=Path("profile_cache.parquet"),
        force_recompute=False,
        ira_kmax=14.0,
    )
    # df_profile has column: 'r' (RMSD from first frame)

Estimate RBF Smoothing
~~~~~~~~~~~~~~~~~~~~~~

Before fitting a landscape surface, estimate a smoothing parameter:

.. code:: python

    from chemparseplot.parse.eon.neb import estimate_rbf_smoothing

    smoothing = estimate_rbf_smoothing(df)
    print(f"Estimated smoothing: {smoothing:.4f}")

This calculates the median inter-image distance in (r, p) space for each step.

Data Format
-----------

The aggregated DataFrame uses this schema:

.. table::

    +------------+-------------+--------------------------------------------+
    | Column     | Type        | Description                                |
    +============+=============+============================================+
    | ``r``      | ``Float64`` | RMSD from reactant (Angstrom)              |
    +------------+-------------+--------------------------------------------+
    | ``p``      | ``Float64`` | RMSD from product (Angstrom)               |
    +------------+-------------+--------------------------------------------+
    | ``grad_r`` | ``Float64`` | Synthetic gradient in R direction          |
    +------------+-------------+--------------------------------------------+
    | ``grad_p`` | ``Float64`` | Synthetic gradient in P direction          |
    +------------+-------------+--------------------------------------------+
    | ``z``      | ``Float64`` | Energy or eigenvalue                       |
    +------------+-------------+--------------------------------------------+
    | ``step``   | ``Int64``   | Optimization step index (-1 for augmented) |
    +------------+-------------+--------------------------------------------+

See Also
--------

- `Tutorial: ORCA NEB Parsing and Plotting <../tutorials/orca_neb.rst>`_

- `eOn NEB API Reference <../reference/eon_neb_api.rst>`_

- `How-to: Create Publication NEB Figures <create_neb_figures.rst>`_

- `How-to: Parse ORCA NEB Calculations <parse_orca_neb.rst>`_

In-memory ConFrame sequences (object plot)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When frames already exist as ``readcon.ConFrame`` (for example
``neb.path_frames()`` from pyeonclient, or a multi-frame CON read with
``readcon``), build plot series without re-reading job directories:

.. code:: python

    from chemparseplot.parse.eon.frame_series import (
        energies_from_frames,
        neb_path_arrays,
        min_trajectory_from_frames,
        dimer_trajectory_from_frames,
    )

    energies = energies_from_frames(frames)
    path = neb_path_arrays(frames)  # atoms_list, energies, optional NEB stamps
    # min_traj = min_trajectory_from_frames(frames)
    # dimer_traj = dimer_trajectory_from_frames(frames)

Wire into rgpycrumbs with:

.. code:: python

    from rgpycrumbs.eon import plot

    plot(frames, kind="neb", plot_type="profile", output_file="1d.pdf")
