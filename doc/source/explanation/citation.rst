=========================
How to Cite chemparseplot
=========================



How to Cite chemparseplot
-------------------------

If you use chemparseplot in your research, please cite the relevant papers based on functionality used.

Software Citation
~~~~~~~~~~~~~~~~~

cite:[goswami2024chemparseplot]

Visualization Methods
~~~~~~~~~~~~~~~~~~~~~

When using 2D reaction valley projection (landscape plotting with ``project_path=True``, ``grad_imq``, ``grad_matern``):

cite:[goswami2026valley]

The method maps NEB trajectories onto a two-dimensional projection defined by
permutation-corrected RMSD from reactant and product configurations, with a
rotated coordinate frame decomposing into reaction progress (``s``) and orthogonal
deviation (``d``).

NEB Methods
~~~~~~~~~~~

When using NEB-related parsing, visualization, or enhanced CI-NEB:

cite:[goswami2026neb]

Enhanced Climbing Image NEB method with Hessian eigenmode alignment for improved saddle point convergence.

Gaussian Process Methods
~~~~~~~~~~~~~~~~~~~~~~~~

When using GP-based surface fitting, interpolation, or saddle search acceleration:

cite:[goswami2025gpr goswami2025pruning]

- GPR-accelerated saddle point searches with efficient implementation

- Adaptive pruning for increased robustness and reduced computational overhead

Statistical Analysis
~~~~~~~~~~~~~~~~~~~~

When using Bayesian hierarchical models for performance analysis:

cite:[goswami2025bayesian]

Combined Citation
~~~~~~~~~~~~~~~~~

For papers using chemparseplot with full functionality:

cite:[goswami2024chemparseplot goswami2026valley goswami2026neb goswami2025gpr goswami2025pruning goswami2025bayesian]

Acknowledgments
~~~~~~~~~~~~~~~

.. code:: text

    Computational chemistry data was parsed and visualized using chemparseplot
    (https://github.com/HaoZeke/chemparseplot). Reaction path visualization employed
    the 2D reaction valley projection method [cite:@goswami2026valley]. NEB calculations
    used the enhanced CI-NEB method [cite:@goswami2026neb]. Gaussian process surface
    fitting followed [cite:@goswami2025gpr @goswami2025pruning].

Related Software
~~~~~~~~~~~~~~~~

ORCA
^^^^

cite:[neese2022]

eOn
^^^

cite:[peterson2016]

rgpycrumbs
^^^^^^^^^^

cite:[goswami2024rgpycrumbs]

metatensor/metatomic
^^^^^^^^^^^^^^^^^^^^

cite:[bigiMetatensorMetatomicFoundational2026]

See Also
~~~~~~~~

- `Reference Documentation <../reference/index.rst>`_

- `Tutorials <../tutorials/index.rst>`_

- `ORCA NEB Design <../explanation/orca_neb_design.rst>`_
