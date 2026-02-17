from collections import namedtuple

import matplotlib.pyplot as plt
from cmcrameri import cm

from rgpycrumbs.interpolation import spline_interp

# Define a namedtuple for energy paths
EnergyPath = namedtuple("EnergyPath", ["label", "distance", "energy"])
XYData = namedtuple("XYData", ["label", "x", "y"])


# Baseline plotting class
class BasePlotter:
    def __init__(
        self, figsize=(3.2, 2.5), dpi=200, pad=0.2, colormap=cm.batlow, style="bmh"
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.pad = pad
        self.colormap = colormap
        plt.style.use(style)
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

    def add_title(self, title=""):
        self.ax.set_title(title)
        self.fig.tight_layout(pad=self.pad)


class TwoDimPlot(BasePlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []  # Stores XYData objects
        self.x_unit = "dimensionless"
        self.y_unit = "dimensionless"
        self.x_label = "X"
        self.y_label = "Y"

    def set_labels(self, x_label, y_label):
        self.x_label = x_label
        self.y_label = y_label
        self.update_labels()

    def set_units(self, x_unit, y_unit):
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.redraw_plot()

    def update_labels(self):
        self.ax.set_xlabel(f"{self.x_label} ({self.x_unit})")
        self.ax.set_ylabel(f"{self.y_label} ({self.y_unit})")

    def redraw_plot(self):
        self.ax.clear()
        for idx, xy_data in enumerate(self.data):
            x_values = xy_data.x.to(self.x_unit).m
            y_values = xy_data.y.to(self.y_unit).m
            distance_fine, y_fine = spline_interp(x_values, y_values)

            color = self.colormap(idx / len(self.data))
            self.ax.plot(
                distance_fine, y_fine, color=color, label=xy_data.label
            )  # Label for the line plot
            self.ax.plot(
                x_values, y_values, linestyle="", marker="o", color=color
            )  # No label for markers

        # Add legend only if there are labeled data plots
        if self.data:
            self.ax.legend()

        self.ax.minorticks_on()
        self.update_labels()
        self.fig.canvas.draw_idle()

    def add_data(self, xy_data):
        self.data.append(xy_data)
        self.redraw_plot()

    def rmdat(self, labels_to_remove):
        if not isinstance(labels_to_remove | (list, set)):
            labels_to_remove = [labels_to_remove]
        self.data = [
            xy_data for xy_data in self.data if xy_data.label not in labels_to_remove
        ]
        self.redraw_plot()

    def __repr__(self):
        return f"TwoDimPlot with {len(self.data)} datasets"
