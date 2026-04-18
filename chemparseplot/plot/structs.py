from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm

_ENERGY_FACTORS = {
    "eV": 1.0,
    "kcal/mol": 23.06054783061903,
    "kJ/mol": 96.48533212331002,
}


class UnitConverted(Protocol):
    """Protocol for the result of converting a unit-aware series."""

    @property
    def m(self) -> Any: ...


class UnitSeries(Protocol):
    """Protocol for pint-like data accepted by plotting helpers."""

    def to(self, unit: str) -> UnitConverted: ...


class AxisUnits(TypedDict):
    """Canonical axis-unit mapping for 2D plot helpers."""

    distance: str
    energy: str


def to_magnitude(values: UnitSeries, unit: str) -> Any:
    """Convert a unit-aware series to magnitudes in the requested unit."""

    return values.to(unit).m


def axis_label(label: str, unit: str) -> str:
    """Format a consistent axis label with units."""

    return f"{label} ({unit})"


def convert_energy(values: Any, unit: str, *, source_unit: str = "eV") -> np.ndarray:
    """Convert energy-like values between supported presentation units."""

    factor = _ENERGY_FACTORS[unit] / _ENERGY_FACTORS[source_unit]
    return np.asarray(values, dtype=float) * factor


def convert_energy_curvature(
    values: Any, unit: str, *, source_unit: str = "eV"
) -> np.ndarray:
    """Convert eigenvalue-like values while preserving the Angstrom denominator."""

    return convert_energy(values, unit, source_unit=source_unit)


def energy_axis_label(unit: str, *, label: str = "Energy") -> str:
    """Format a canonical energy-axis label."""

    return axis_label(label, unit)


def eigenvalue_axis_label(unit: str, *, label: str = "Eigenvalue") -> str:
    """Format a canonical curvature-axis label."""

    return f"{label} ({unit}/$\\AA^2$)"


@dataclass(frozen=True, slots=True)
class EnergyPath:
    """Typed energy path with unit-aware distance and energy series.

```{versionadded} 0.0.3
```
"""

    label: str
    distance: UnitSeries
    energy: UnitSeries


@dataclass(frozen=True, slots=True)
class XYData:
    """Typed generic XY data with unit-aware axes.

```{versionadded} 0.0.3
```
"""

    label: str
    x: UnitSeries
    y: UnitSeries


# Baseline plotting class
class BasePlotter:
    """Thin wrapper around a matplotlib figure and axes with defaults.

    ```{versionadded} 0.0.3
    ```
    """

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
    """Interactive 2D plotter with unit-aware axes and spline interpolation.

    ```{versionadded} 0.0.3
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data: list[XYData] = []
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
        self.ax.set_xlabel(axis_label(self.x_label, self.x_unit))
        self.ax.set_ylabel(axis_label(self.y_label, self.y_unit))

    def redraw_plot(self):
        self.ax.clear()
        for idx, xy_data in enumerate(self.data):
            x_values = to_magnitude(xy_data.x, self.x_unit)
            y_values = to_magnitude(xy_data.y, self.y_unit)
            from rgpycrumbs.interpolation import spline_interp

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

    def add_data(self, xy_data: XYData):
        self.data.append(xy_data)
        self.redraw_plot()

    def rmdat(self, labels_to_remove):
        if not isinstance(labels_to_remove, list | set):
            labels_to_remove = [labels_to_remove]
        self.data = [
            xy_data for xy_data in self.data if xy_data.label not in labels_to_remove
        ]
        self.redraw_plot()

    def __repr__(self):
        return f"TwoDimPlot with {len(self.data)} datasets"
