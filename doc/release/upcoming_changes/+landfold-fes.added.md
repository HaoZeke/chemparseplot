``parse.landfold`` / ``plot.landfold`` consume ``landfold.fes.v1`` and the
``landfold fes --csv`` grid, then fit ``rgpycrumbs.surfaces`` through
``plot_landscape_surface`` (default ``grad_imq``, same stack as NEB
landscapes). Landfold owns the invert; chemparseplot owns the figure.
