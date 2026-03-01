``chemparseplot``: Chemical Parsers and Plotters
================================================

.. image:: ../../branding/logo/chemparseplot_logo.png

About
-----

A **pure-python** parsing and plotting library for computational chemistry
outputs. ``chemparseplot`` extracts structured data from quantum chemistry codes
(ORCA, eOn, Sella, ChemGP) and produces publication-quality, unit-aware
visualizations with `scientific color maps
<https://www.fabiocrameri.ch/colourmaps/>`_.

Computational tasks (surface fitting, interpolation, structure analysis) are
delegated to `rgpycrumbs <https://github.com/HaoZeke/rgpycrumbs>`_, which is a
required dependency. For more information check the :doc:`features` page.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   installation
   features

.. toctree::
   :maxdepth: 2
   :caption: Guides

   tutorials/index
   contributing

.. toctree::
   :maxdepth: 2
   :caption: Reference

   apidocs/index
   used_by
   worklog/graphTrials

License
-------

MIT. However, this is an academic resource, so **please cite** as much as
possible via:

- The `Zenodo DOI <https://doi.org/10.5281/zenodo.18529752>`_ for general use.
- The ``wailord`` paper for ORCA usage.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
