"""Plot energy along the imaginary-time orbit grid."""

from __future__ import annotations

import numpy as np


def plot_orbit_energy(ax, phase, energy, reference, **kwargs):
    """Plot the measured energy above a reference in its declared unit.

    Phase is the fractional imaginary time on any ordered orbit grid.
    Energy and reference are quantities with convertible energy units.
    """
    phase = np.asarray(phase)
    values = (energy - reference).magnitude
    if phase.ndim != 1 or phase.shape != np.shape(values):
        msg = "phase and energy must have the same one-dimensional shape"
        raise ValueError(msg)
    (line,) = ax.plot(phase, values, **kwargs)
    ax.set_xlabel("Fractional imaginary time")
    ax.set_ylabel(f"Energy relative to reference ({energy.units:~})")
    return line
