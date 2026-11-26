"""Rate plots for temperature-dependent kinetics."""

import os
import tempfile

import numpy as np

_MIN_ENSEMBLE_FRAMES = 2


def plot_arrhenius(ax, temperature, log10_rate_per_second, **kwargs):
    """Plot base-ten log rates against inverse temperature on an existing axis.

    Temperature is a quantity convertible to kelvin. Passing log rates avoids
    exponentiating very small tunneling rates. Matplotlib style arguments are
    forwarded to the curve, including marker styles for reference measurements.
    """
    kelvin = np.asarray(temperature.to("kelvin").magnitude, dtype=float)
    rates = np.asarray(log10_rate_per_second, dtype=float)
    if kelvin.ndim != 1 or kelvin.shape != rates.shape:
        msg = "temperatures and log rates must be matching one-dimensional arrays"
        raise ValueError(msg)
    if np.any(~np.isfinite(kelvin)) or np.any(kelvin <= 0):
        msg = "temperatures must be positive and finite"
        raise ValueError(msg)
    (curve,) = ax.plot(1000 / kelvin, rates, **kwargs)
    ax.set_xlabel(r"$1000\,\mathrm{K}/T$")
    ax.set_ylabel(r"$\log_{10}(k/\mathrm{s}^{-1})$")
    return curve


def render_path_ensemble(
    frames, *, canvas_size=700, config="paton", opacity=0.45, color=None
):
    """Render structures along a path as one overlaid xyzrender image.

    ``frames`` is a sequence of ASE ``Atoms`` with the same atom order, such
    as the replicas of a tunnelling path. The first frame is drawn opaque
    and sets the orientation. The others are aligned to it and drawn with
    ``opacity``, coloured along the sequence unless ``color`` is given.

    Returns an RGBA array with a transparent background.
    """
    # matplotlib, ase and neb (scipy) are optional; plot_arrhenius must import
    # without them.
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from ase.io import write  # noqa: PLC0415

    from chemparseplot.plot.neb import _import_xyzrender  # noqa: PLC0415

    frames = list(frames)
    if len(frames) < _MIN_ENSEMBLE_FRAMES:
        msg = "a path ensemble needs at least two frames"
        raise ValueError(msg)
    xr = _import_xyzrender()
    with tempfile.TemporaryDirectory() as folder:
        source = os.path.join(folder, "path.xyz")
        image = os.path.join(folder, "path.png")
        write(source, frames, format="xyz")
        options = {"ensemble_opacity": opacity}
        if color is not None:
            options["ensemble_color"] = color
        molecule = xr.load(source, ensemble=True, reference_frame=0, **options)
        xr.render(
            molecule,
            config=config,
            canvas_size=canvas_size,
            hy=True,
            transparent=True,
            output=image,
        )
        return plt.imread(image)


def add_image_inset(ax, image, bounds, title=None):
    """Place an RGBA image in an inset of ``ax`` at ``bounds`` (axes fractions)."""
    inset = ax.inset_axes(bounds)
    inset.imshow(image)
    inset.axis("off")
    if title:
        inset.set_title(title, fontsize="small")
    return inset
