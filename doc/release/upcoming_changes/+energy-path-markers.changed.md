`plot_energy_path` takes the image marker size and edge width from the `lines.markersize` and `lines.markeredgewidth` rcParams.
The journal style sets both; they were fixed at 6 pt and 0.5 pt before.
With matplotlib's defaults the edge is 1 pt.
It takes `markersize`, `markeredgewidth`, `markeredgecolor` and `samples`.
Without `samples` the curve has a point for every pixel column at `savefig.dpi`, where it had 300 points.
