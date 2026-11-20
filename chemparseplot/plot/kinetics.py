"""Rate plots for temperature-dependent kinetics."""

import numpy as np


def plot_arrhenius(ax, temperature, log10_rate_per_second, **kwargs):
    """Plot base-ten log rates against inverse temperature on an existing axis.

    Temperature is a quantity convertible to kelvin. Passing log rates avoids
    exponentiating very small tunneling rates. Matplotlib style arguments are
    forwarded to the curve, including marker styles for reference measurements.
    """
    kelvin = np.asarray(temperature.to("kelvin").magnitude, dtype=float)
    rates = np.asarray(log10_rate_per_second, dtype=float)
    if kelvin.ndim != 1 or kelvin.shape != rates.shape:
        raise ValueError("temperatures and log rates must be matching one-dimensional arrays")
    if np.any(~np.isfinite(kelvin)) or np.any(kelvin <= 0):
        raise ValueError("temperatures must be positive and finite")
    (curve,) = ax.plot(1000 / kelvin, rates, **kwargs)
    ax.set_xlabel(r"$1000\,\mathrm{K}/T$")
    ax.set_ylabel(r"$\log_{10}(k/\mathrm{s}^{-1})$")
    return curve
