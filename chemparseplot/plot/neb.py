import io
import logging
from collections import namedtuple
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from ase.io import write as ase_write
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ArrowStyle
from scipy.interpolate import (
    CubicHermiteSpline,
    RBFInterpolator,
    griddata,
    splev,
    splrep,
)
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist

log = logging.getLogger(__name__)

# --- Data Structures ---
InsetImagePos = namedtuple("InsetImagePos", "x y rad")


@dataclass
class SmoothingParams:
    window_length: int = 5
    polyorder: int = 2


MIN_PATH_LENGTH = 1e-6

# --- Structure Rendering Helpers ---


def render_structure_to_image(atoms, zoom, rotation):
    """Renders an ASE atoms object to a numpy image array."""
    buf = io.BytesIO()
    ase_write(buf, atoms, format="png", rotation=rotation, show_unit_cell=0, scale=100)
    buf.seek(0)
    img_data = plt.imread(buf)
    buf.close()
    return img_data


def plot_structure_strip(
    ax,
    atoms_list,
    labels,
    zoom=0.3,
    rotation="0x,90y,0z",
    theme_color="black",
    max_cols=6,
):
    """Renders a horizontal gallery of atomic structures."""
    ax.axis("off")
    n_plot = len(atoms_list)
    n_cols = min(n_plot, max_cols)
    n_rows = (n_plot + max_cols - 1) // max_cols
    row_step = 8.5

    ax.set_xlim(-0.5, n_cols - 0.5)
    # Heuristic layout from plt_neb.py
    y_min = -((n_rows - 1) * row_step) - 0.8
    y_max = 0.6
    ax.set_ylim(y_min, y_max)

    for i, atoms in enumerate(atoms_list):
        col = i % max_cols
        row = i // max_cols
        x_pos, y_pos = col, -row * row_step

        # Image generation
        img_data = render_structure_to_image(atoms, zoom, rotation)

        # Adjust zoom for strip
        effective_zoom = zoom * 0.45
        imagebox = OffsetImage(img_data, zoom=effective_zoom)

        ab = AnnotationBbox(
            imagebox,
            (x_pos, y_pos),
            frameon=False,
            xycoords="data",
            boxcoords="offset points",
            pad=0.0,
        )
        ax.add_artist(ab)

        if labels and i < len(labels):
            ax.text(
                x_pos,
                y_pos - 0.8,
                labels[i],
                ha="center",
                va="top",
                fontsize=11,
                color=theme_color,
                fontweight="bold",
            )


def plot_structure_inset(
    ax, atoms, x, y, xybox, rad, zoom=0.4, rotation="0x,90y,0z", arrow_props=None
):
    """Plots a single structure as an annotation inset."""
    img_data = render_structure_to_image(atoms, zoom, rotation)
    imagebox = OffsetImage(img_data, zoom=zoom)

    # Default arrow properties matching plt_neb.py
    default_arrow = {
        "arrowstyle": ArrowStyle.Fancy(head_length=0.4, head_width=0.4, tail_width=0.1),
        "connectionstyle": f"arc3,rad={rad}",
        "linestyle": "-",
        "alpha": 0.8,
        "color": "black",
        "linewidth": 1.2,
    }
    if arrow_props:
        default_arrow.update(arrow_props)

    ab = AnnotationBbox(
        imagebox,
        (x, y),
        xybox=xybox,
        frameon=False,
        xycoords="data",
        boxcoords="offset points",
        pad=0.1,
        arrowprops=default_arrow,
    )
    ax.add_artist(ab)
    ab.set_zorder(80)


# --- Path Plotting Helpers ---


def plot_energy_path(
    ax, rc, energy, f_para, color, alpha, zorder, method="hermite", smoothing=None
):
    """Plots 1D energy profile with optional Hermite spline interpolation."""
    if smoothing is None:
        smoothing = SmoothingParams()

    # Derivative is negative force
    deriv = -f_para

    try:
        idx = np.argsort(rc)
        rc_s = rc[idx]
        e_s = energy[idx]

        rc_min, rc_max = rc_s.min(), rc_s.max()
        path_length = rc_max - rc_min

        # Normalize
        if path_length > MIN_PATH_LENGTH:
            rc_norm = (rc_s - rc_min) / path_length
        else:
            rc_norm = rc_s

        rc_fine_norm = np.linspace(0, 1, 300)

        if method == "hermite":
            deriv_smooth = savgol_filter(
                deriv, smoothing.window_length, smoothing.polyorder
            )
            deriv_sorted = deriv_smooth[idx]
            # Scale derivative to normalized coordinate
            deriv_scaled = deriv_sorted * path_length

            spline = CubicHermiteSpline(rc_norm, e_s, deriv_scaled)
            y_fine = spline(rc_fine_norm)
        else:
            tck = splrep(rc_norm, e_s, k=3)
            y_fine = splev(rc_fine_norm, tck)

        x_fine = rc_fine_norm * path_length + rc_min

        ax.plot(x_fine, y_fine, color=color, alpha=alpha, zorder=zorder)
        ax.plot(
            rc,
            energy,
            marker="o",
            ls="None",
            color=color,
            ms=6,
            alpha=alpha,
            zorder=zorder + 1,
            markerfacecolor=color,
            markeredgewidth=0.5,
        )

    except Exception as e:
        log.warning(f"Spline failed ({e}), plotting raw lines.")
        ax.plot(rc, energy, color=color, alpha=alpha, zorder=zorder)


def plot_eigenvalue_path(ax, rc, eigenvalue, color, alpha, zorder, grid_color="white"):
    """Plots 1D eigenvalue profile."""
    try:
        idx = np.argsort(rc)
        rc_s = rc[idx]
        ev_s = eigenvalue[idx]
    except ValueError:
        rc_s, ev_s = rc, eigenvalue

    # Standard spline
    rc_fine = np.linspace(rc.min(), rc.max(), 300)
    tck = splrep(rc_s, ev_s, k=3)
    y_fine = splev(rc_fine, tck)

    ax.plot(rc_fine, y_fine, color=color, alpha=alpha, zorder=zorder)
    ax.plot(
        rc,
        eigenvalue,
        marker="s",
        ls="None",
        color=color,
        ms=6,
        alpha=alpha,
        zorder=zorder + 1,
        markerfacecolor=color,
        markeredgewidth=0.5,
    )
    ax.axhline(0, color=grid_color, linestyle=":", linewidth=1.5, alpha=0.8, zorder=1)


def _augment_minima_points(rmsd_r, rmsd_p, z_data, radius=0.01, dE=0.02, num_pts=12):
    """
    Creates a 'collar' of synthetic points around the endpoints.
    This forces the RBF interpolator to curve upwards around these points, preventing
    artificial wells (overshooting) where the physics dictates a minimum.
    """
    # Explicitly handle endpoints to ensure they use their own energy
    indices = {0, len(z_data) - 1}
    
    aug_r, aug_p, aug_z = [rmsd_r], [rmsd_p], [z_data]

    for idx in indices:
        r0, p0, z0 = rmsd_r[idx], rmsd_p[idx], z_data[idx]

        # Generate a ring of points
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        ring_r = r0 + radius * np.cos(angles)
        ring_p = p0 + radius * np.sin(angles)
        # Force energy higher to create a bowl shape
        ring_z = np.full_like(ring_r, z0 + dE)

        aug_r.append(ring_r)
        aug_p.append(ring_p)
        aug_z.append(ring_z)

    return np.concatenate(aug_r), np.concatenate(aug_p), np.concatenate(aug_z)


def _augment_with_gradients(r, p, z, gr, gp, epsilon=0.05):
    """
    Uses projected gradients to create helper points slightly offset from the path.
    This effectively tells the RBF interpolator the local slope.

    Creates 4 helper points for every real point:
      (r +/- eps, p) and (r, p +/- eps)
    """
    if gr is None or gp is None:
        return r, p, z

    # Helper 1: Step in R
    r1, p1, z1 = r + epsilon, p, z + epsilon * gr
    r2, p2, z2 = r - epsilon, p, z - epsilon * gr

    # Helper 2: Step in P
    r3, p3, z3 = r, p + epsilon, z + epsilon * gp
    r4, p4, z4 = r, p - epsilon, z - epsilon * gp

    return (
        np.concatenate([r, r1, r2, r3, r4]),
        np.concatenate([p, p1, p2, p3, p4]),
        np.concatenate([z, z1, z2, z3, z4]),
    )


def plot_landscape_surface(
    ax,
    rmsd_r,
    rmsd_p,
    grad_r,
    grad_p,
    z_data,
    step_data=None,
    method="rbf",
    rbf_smooth=None,
    cmap="viridis",
    show_pts=True,
):
    """
    Plots the 2D landscape surface using RBF or Grid interpolation.

    Automatically augments data around minima to enforce physical wells if using RBF.
    """
    log.info(f"Generating 2D surface using {method}...")

    nx, ny = 150, 150
    # Add buffer to grid
    x_margin = (rmsd_r.max() - rmsd_r.min()) * 0.1
    y_margin = (rmsd_p.max() - rmsd_p.min()) * 0.1
    xg = np.linspace(rmsd_r.min() - x_margin, rmsd_r.max() + x_margin, nx)
    yg = np.linspace(rmsd_p.min() - y_margin, rmsd_p.max() + y_margin, ny)
    xg, yg = np.meshgrid(xg, yg)

    if method == "grid":
        zg = griddata((rmsd_r, rmsd_p), z_data, (xg, yg), method="cubic")
    else:
        # Minima Augmentation (Create Bowls)
        r_aug, p_aug, z_aug = _augment_minima_points(rmsd_r, rmsd_p, z_data)

        # Gradient Augmentation (Enforce Slope)
        # Map the gradients to the augmented array, but since augmentation
        # adds new points with unknown gradients, apply gradient augmentation
        # ONLY to the original raw points first, then combine.
        if grad_r is not None and grad_p is not None:
            r_grad, p_grad, z_grad = _augment_with_gradients(
                rmsd_r, rmsd_p, z_data, grad_r, grad_p
            )
            # Combine both augmented sets
            pts = np.column_stack(
                [np.concatenate([r_aug, r_grad]), np.concatenate([p_aug, p_grad])]
            )
            vals = np.concatenate([z_aug, z_grad])
        else:
            pts = np.column_stack([r_aug, p_aug])
            vals = z_aug

        safe_smooth = rbf_smooth if rbf_smooth is not None else 0.0
        rbf = RBFInterpolator(
            pts, vals, kernel="thin_plate_spline", smoothing=safe_smooth
        )
        zg = rbf(np.column_stack([xg.ravel(), yg.ravel()])).reshape(xg.shape)

    try:
        colormap = plt.get_cmap(cmap)
    except ValueError:
        colormap = plt.get_cmap("viridis")

    ax.contourf(xg, yg, zg, levels=20, cmap=colormap, alpha=0.75, zorder=10)
    ax.contour(xg, yg, zg, levels=15, colors="white", alpha=0.3, linewidths=0.5, zorder=11)

    if show_pts:
        # TODO(rg): will a user every want to control this?
        # Filter: Only show points that are NOT augmented (step != -1)
        if step_data is not None:
            mask = step_data != -1
            plot_r, plot_p = rmsd_r[mask], rmsd_p[mask]
        else:
            plot_r, plot_p = rmsd_r, rmsd_p
            
        ax.scatter(plot_r, plot_p, c="k", s=12, marker=".", alpha=0.6, zorder=40)


def plot_landscape_path_overlay(ax, r, p, z, cmap, z_label):
    """Overlays the colored path line on the landscape."""
    points = np.array([r, p]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    try:
        colormap = plt.get_cmap(cmap)
    except ValueError:
        colormap = plt.get_cmap("viridis")

    norm = plt.Normalize(z.min(), z.max())
    lc = LineCollection(segments, cmap=colormap, norm=norm, zorder=30)

    # Color segments by average Z of endpoints
    lc.set_array((z[:-1] + z[1:]) / 2)
    lc.set_linewidth(3)
    ax.add_collection(lc)

    ax.scatter(
        r,
        p,
        c=z,
        cmap=colormap,
        norm=norm,
        edgecolors="black",
        linewidths=0.5,
        zorder=40,
    )

    cb = ax.figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax, label=z_label
    )
    return cb
