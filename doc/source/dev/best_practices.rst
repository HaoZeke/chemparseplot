==============
Best Practices
==============


.. contents::


Best Practices for Using chemparseplot
--------------------------------------

Parsing
~~~~~~~

Always Check Convergence
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    data = parse_orca_neb("job")
    if not data["converged"]:
        print("Warning: Calculation did not converge!")

Use Context Managers for Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from pathlib import Path

    work_dir = Path("calculation")
    data = parse_orca_neb("job", working_dir=work_dir)

Handle Missing Data Gracefully
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    energies = data.get("energies")
    if energies is None or len(energies) == 0:
        print("No energy data available")
        return

Plotting
~~~~~~~~

Use Consistent Figure Sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # Standard single plot
    width, height = 5.37, 5.37  # inches

    # Wide plot for comparisons
    width, height = 7.0, 5.0

    # Tall plot for profiles
    width, height = 5.0, 7.0

Use Scientific Colormaps
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # Good choices
    cmap = "cmc.batlow"      # Perceptually uniform
    cmap = "cmc.berlin"      # Diverging
    cmap = "cmc.oslo"        # Sequential

    # Avoid
    cmap = "jet"             # Not perceptually uniform
    cmap = "rainbow"         # Misleading

Set Appropriate DPI
^^^^^^^^^^^^^^^^^^^

.. code:: python

    # For presentations
    dpi = 150

    # For publications
    dpi = 300

    # For web
    dpi = 72

Performance
~~~~~~~~~~~

Use Parallel Processing for Batches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    # Sequential (slow for many files)
    rgpycrumbs chemgp batch -c config.toml

    # Parallel (4 workers)
    rgpycrumbs chemgp batch -c config.toml -j 4

    # Parallel (8 workers, fast)
    rgpycrumbs chemgp batch -c config.toml -j 8

Cache Parsed Data
^^^^^^^^^^^^^^^^^

.. code:: python

    import pickle
    from pathlib import Path

    cache_file = Path("data_cache.pkl")

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
    else:
        data = parse_orca_neb("job")
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

Downsample Large Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    import numpy as np

    # Original data with 1000 points
    energies = data["energies"]

    # Downsample to 100 points
    indices = np.linspace(0, len(energies)-1, 100, dtype=int)
    energies_ds = energies[indices]

Error Handling
~~~~~~~~~~~~~~

Catch Specific Exceptions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from chemparseplot.parse.orca.neb import parse_orca_neb

    try:
        data = parse_orca_neb("job")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install orca-pi")
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
    except ValueError as e:
        print(f"Invalid data: {e}")

Validate Input Data
^^^^^^^^^^^^^^^^^^^

.. code:: python

    def validate_neb_data(data: dict) -> bool:
        """Validate NEB data structure."""
        required_keys = ["energies", "n_images", "converged"]
        for key in required_keys:
            if key not in data:
                return False

        if len(data["energies"]) != data["n_images"]:
            return False

        return True

Documentation
~~~~~~~~~~~~~

Include Examples in Docstrings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    def plot_profile(data, output):
        """Plot energy profile.

        Parameters
        ----------
        data : dict
            Parsed NEB data
        output : Path
            Output file path

        Example
        -------
        >>> data = parse_orca_neb("job")
        >>> plot_profile(data, "profile.pdf")
        """
        ...

Use Cross-References
^^^^^^^^^^^^^^^^^^^^

.. code:: org

    See [[file:../howto/parse_orca_neb.org][Parse ORCA NEB Calculations]] for details.

Testing
~~~~~~~

Test with Mock Data
^^^^^^^^^^^^^^^^^^^

.. code:: python

    import numpy as np

    mock_data = {
        "energies": np.array([0.0, 0.5, 1.0, 0.5, 0.0]),
        "n_images": 5,
        "converged": True,
    }

    plot_profile(mock_data, "test_profile.pdf")

Test Edge Cases
^^^^^^^^^^^^^^^

.. code:: python

    # Empty data
    plot_profile({"energies": [], "n_images": 0}, "empty.pdf")

    # Single image
    plot_profile({"energies": [0.0], "n_images": 1}, "single.pdf")

    # Very large barrier
    plot_profile({"energies": [0.0, 100.0, 0.0], "n_images": 3}, "large.pdf")

Version Control
~~~~~~~~~~~~~~~

Pin Dependencies in Production
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: toml

    # pyproject.toml
    [project]
    dependencies = [
        "chemparseplot>=0.2.0,<0.3.0",
    ]

Use Semantic Versioning
^^^^^^^^^^^^^^^^^^^^^^^

- MAJOR.MINOR.PATCH

- MAJOR: Breaking changes

- MINOR: New features (backward compatible)

- PATCH: Bug fixes

Security
~~~~~~~~

Validate File Paths
^^^^^^^^^^^^^^^^^^^

.. code:: python

    from pathlib import Path

    # Good: Validate path
    output = Path(user_input).resolve()
    if not output.is_relative_to(working_dir):
        raise ValueError("Path outside working directory")

Don't Execute Untrusted Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # BAD: Never do this
    exec(user_input)

    # GOOD: Use safe parsing
    import tomllib
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

See Also
~~~~~~~~

- `Contributing Guidelines <contributing.rst>`_

- `How-to Guides <../howto/index.rst>`_

- `Reference <../reference/index.rst>`_
