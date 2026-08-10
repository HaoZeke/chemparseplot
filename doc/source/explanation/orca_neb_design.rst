===========================
ORCA NEB Integration Design
===========================



Overview
--------

This document explains the design decisions behind ORCA NEB integration in chemparseplot.

Why OPI?
--------

Problem: Fragile Text Parsing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previous ORCA parsing used regex-based text extraction:

.. code:: python

    # Old approach - fragile
    pattern = r"Iteration:\s*(?P<iteration>\d+)\s*\n.*?" + THREE_COL_NUM
    match = re.search(pattern, text, re.DOTALL)

Issues:

- Breaks when ORCA output format changes

- Hard to maintain

- No type safety

- Manual unit conversion

Solution: OPI JSON Parsing
~~~~~~~~~~~~~~~~~~~~~~~~~~

OPI (ORCA Python Interface) is the official FACCTs package for ORCA 6.1+:

.. code:: python

    # New approach - robust
    from opi.output.core import Output
    output = Output("job")
    output.parse()
    energies = [output.get_final_energy(i) for i in range(output.num_results_gbw)]

Benefits:

- Official support from ORCA developers

- JSON-based (stable format)

- Type-safe with Pydantic models

- Automatic unit handling

- Actively maintained

Why Lazy Loading?
-----------------

Problem: Hard Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

Making OPI a hard dependency would:

- Force all users to install orca-pi

- Break installations without ORCA

- Increase package size

Solution: Lazy Loading via ``ensure_import()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from rgpycrumbs._aux import ensure_import

    _opi_output = None

    def _get_opi_output():
        global _opi_output
        if _opi_output is None:
            _opi_output = ensure_import("opi.output.core").Output
        return _opi_output

Benefits:

- No hard dependency

- Auto-installs if ``RGPYCRUMBS_AUTO_DEPS=1``

- Clear error messages if missing

- Follows existing chemparseplot pattern

Why Compatible Data Format?
---------------------------

Problem: Code Duplication
~~~~~~~~~~~~~~~~~~~~~~~~~

eOn NEB and ORCA NEB produce similar data:

- Energy profiles

- RMSD coordinates

- Gradients

Without compatible format:

- Separate plotting code for each

- Duplicate effort

- Inconsistent figures

Solution: Unified Data Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both parsers return compatible dict:

.. code:: python

    # eOn NEB data
    data_eon = parse_eon_neb(...)
    # ORCA NEB data
    data_orca = parse_orca_neb(...)

    # Same plotting functions work for both!
    plot_energy_path(ax, data_eon['rc'], data_eon['energies'], ...)
    plot_energy_path(ax, data_orca['rmsd_r'], data_orca['energies'], ...)

Benefits:

- Single plotting codebase

- Consistent figures across methods

- Easy to add new parsers (VASP, Quantum ESPRESSO, etc.)

- Users learn one API

Design Trade-offs
-----------------

RMSD Calculation
~~~~~~~~~~~~~~~~

- ``parse_orca_neb()`` calculates RMSD from geometries

- Requires ASE library

- Falls back to ``None`` if ASE unavailable

- Alternative: Use IRA for proper alignment (future work)

Synthetic Gradients
~~~~~~~~~~~~~~~~~~~

- Projected forces onto RMSD coordinates

- Simplified version (not full IRA)

- Sufficient for landscape visualization

- Full implementation would require IRA integration

ORCA Version Support
~~~~~~~~~~~~~~~~~~~~

- OPI: ORCA 6.1+ only

- Legacy: ORCA < 6.1 via .interp files

- Automatic fallback

- Clear version reporting in output

Future Improvements
-------------------

1. ****IRA Integration****: Proper reaction path alignment

2. ****More Parsers****: VASP, Quantum ESPRESSO, Gaussian NEB

3. ****Batch Processing****: Parse multiple calculations at once

4. ****Caching****: Cache parsed results for faster re-plots

5. ****Interactive Plots****: Plotly/Bokeh backend option

Related
-------

- `Tutorial: ORCA NEB Parsing and Plotting <../tutorials/orca_neb.rst>`_

- `How-to: Parse ORCA NEB Calculations <../howto/parse_orca_neb.rst>`_

- `ORCA NEB API Reference <../reference/orca_neb_api.rst>`_

- `Lazy Import Pattern <../explanation/lazy_imports.rst>`_
