``chemparseplot``: Chemical Parsers and Plotters
================================================

.. image:: ../../branding/logo/chemparseplot_logo.png

About
-----

A **pure-python** [1]_  project to provide unit-aware uniform visualizations
of common computational chemistry tasks. Essentially this means we provide:

- Plotting scripts for specific workflows
- Parsers for various software outputs

This is a spin-off from ``wailord`` (`here <https://wailord.xyz>`_) which is
meant to handle aggregated runs in a specific workflow, while here the goal is
to do no input handling and very pragmatic output parsing, with the goal of
generating uniform plots. For more information check the `features` page.

Documentation TOC
-----------------

.. toctree::
   :maxdepth: 2

   apidocs/index
   tutorials/index
   installation
   contributing
   features

Features
~~~~~~~~

- `Scientific color maps <https://www.fabiocrameri.ch/colourmaps/>`_ for the plots
  - Camera ready
- Unit preserving
  - Via ``pint``

These are supported for:

License
-------

MIT. However, this is an academic resource, so **please cite** as much as possible
via:

- The Zenodo DOI for general use.

- The ``wailord`` paper for ORCA usage


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. [1] To distinguish it from my other thin-python wrapper projects
