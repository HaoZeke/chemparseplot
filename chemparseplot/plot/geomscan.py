import matplotlib.pyplot as plt

from chemparseplot.plot._aids import spline_interp
from chemparseplot.plot.structs import BasePlotter


def plot_energy_paths(energy_paths, units, colormap_fraction=1.0, plotter=None):
    if plotter is None:
        plotter = BasePlotter()

    for idx, path in enumerate(energy_paths):
        distance_values = path.distance.to(units["distance"])
        energy_values = path.energy.to(units["energy"])

        distance_fine, energy_fine = spline_interp(distance_values.m, energy_values.m)

        # Determine the color for each path using the colormap
        color = plotter.colormap(idx / len(energy_paths) * colormap_fraction)
        plotter.ax.plot(distance_fine, energy_fine, color=color, label=path.label)
        plotter.ax.plot(
            distance_values, energy_values, linestyle="", marker="o", color=color
        )

    plotter.ax.set_xlabel(f"Distance ({units['distance']})")
    plotter.ax.set_ylabel(f"Energy ({units['energy']})")
    plotter.ax.legend()
    plotter.ax.minorticks_on()
    plotter.ax.set_facecolor("gray")

    # Adding a colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plotter.colormap, norm=plt.Normalize(vmin=0, vmax=len(energy_paths) - 1)
    )
    cbar = plotter.fig.colorbar(sm, ax=plotter.ax)
    cbar.set_label("Path Index")

    return plotter
