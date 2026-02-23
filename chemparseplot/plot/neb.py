import io
import logging
import shutil
import subprocess
import tempfile
from collections import namedtuple
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from ase.io import write as ase_write
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ArrowStyle
from rgpycrumbs.surfaces import (
    NYSTROM_N_INDUCING_DEFAULT,
    NYSTROM_THRESHOLD,
    get_surface_model,
    nystrom_paths_needed,
)
from scipy import ndimage
from scipy.interpolate import (
    CubicHermiteSpline,
    splev,
    splrep,
)
from scipy.signal import savgol_filter

log = logging.getLogger(__name__)

# --- Data Structures ---
InsetImagePos = namedtuple("InsetImagePos", "x y rad")
"""Position specification for an inset structure image (x, y, rad).

```{versionadded} 0.1.0
```
"""


@dataclass
class SmoothingParams:
    """Parameters for Savitzky-Golay smoothing of NEB force profiles.

    ```{versionadded} 0.1.0
    ```
    """

    window_length: int = 5
    polyorder: int = 2


MIN_PATH_LENGTH = 1e-6

# --- Structure Rendering Helpers ---


def render_structure_to_image(atoms, zoom, rotation):  # noqa: ARG001
    """Renders an ASE Atoms object to a numpy RGBA image array.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to render.
    zoom : float
        Zoom level (used by callers for OffsetImage scaling, not by ASE).
    rotation : str
        ASE rotation string, e.g. ``"0x,90y,0z"``.

    Returns
    -------
    numpy.ndarray
        RGBA image array with shape ``(H, W, 4)`` and float dtype.

    ```{versionadded} 0.1.0
    ```
    """
    buf = io.BytesIO()
    ase_write(
        buf, atoms, format="png", rotation=rotation, show_unit_cell=0, scale=100
    )
    buf.seek(0)
    img_data = plt.imread(buf)
    buf.close()
    return img_data


def _check_xyzrender():
    """Verify that the ``xyzrender`` binary is on PATH.

    Raises
    ------
    RuntimeError
        If xyzrender is not found, with install instructions.
    """
    if shutil.which("xyzrender") is None:
        msg = (
            "xyzrender binary not found on PATH. "
            "Install with: pip install 'xyzrender>=0.1.3'"
        )
        raise RuntimeError(msg)


def _render_xyzrender(atoms, canvas_size=400):
    """Render an ASE Atoms object to a numpy RGBA array via xyzrender.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to render.
    canvas_size : int
        Output image width/height in pixels (passed as ``-S``).

    Returns
    -------
    numpy.ndarray
        RGBA image array with shape ``(H, W, 4)`` and float dtype.
    """
    from ase.io import write as _ase_write

    with tempfile.NamedTemporaryFile(
        suffix=".xyz", delete=False
    ) as xyz_fh:
        xyz_path = xyz_fh.name
    png_path = xyz_path.rsplit(".", 1)[0] + ".png"

    try:
        _ase_write(xyz_path, atoms, format="xyz")
        cmd = [
            "xyzrender",
            xyz_path,
            "-o",
            png_path,
            "-S",
            str(canvas_size),
        ]
        subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603
        img_data = plt.imread(png_path)
    finally:
        import os

        for p in (xyz_path, png_path):
            try:
                os.unlink(p)
            except OSError:
                pass
    return img_data


def plot_structure_strip(
    ax,
    atoms_list,
    labels,
    zoom=0.3,
    rotation="0x,90y,0z",
    theme_color="black",
    max_cols=6,
    renderer="ase",
):
    """Renders a horizontal gallery of atomic structures.

    Parameters
    ----------
    renderer : str
        Rendering backend: ``"ase"`` (default) or ``"xyzrender"``.

    ```{versionadded} 0.1.0
    ```

    ```{versionchanged} 1.2.0
    Added the *renderer* parameter.
    ```
    """
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

    if renderer == "xyzrender":
        _check_xyzrender()

    for i, atoms in enumerate(atoms_list):
        col = i % max_cols
        row = i // max_cols
        x_pos, y_pos = col, -row * row_step

        if renderer == "xyzrender":
            img_data = _render_xyzrender(atoms, canvas_size=400)
        else:
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
    ax,
    atoms,
    x,
    y,
    xybox,
    rad,
    zoom=0.4,
    rotation="0x,90y,0z",
    arrow_props=None,
    renderer="ase",
):
    """Plots a single structure as an annotation inset.

    Parameters
    ----------
    renderer : str
        Rendering backend: ``"ase"`` (default) or ``"xyzrender"``.

    ```{versionadded} 0.1.0
    ```

    ```{versionchanged} 1.2.0
    Added the *renderer* parameter.
    ```
    """
    if renderer == "xyzrender":
        _check_xyzrender()
        img_data = _render_xyzrender(atoms, canvas_size=400)
    else:
        img_data = render_structure_to_image(atoms, zoom, rotation)
    # Apply the same unified scaling as the strip
    effective_zoom = zoom * 0.45
    imagebox = OffsetImage(img_data, zoom=effective_zoom)

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
    """Plots 1D energy profile with optional Hermite spline interpolation.

    ```{versionadded} 0.1.0
    ```
    """
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
    """Plots 1D eigenvalue profile.

    ```{versionadded} 0.1.0
    ```
    """
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


def _augment_minima_points(rmsd_r, rmsd_p, z_data, radius=0.01, d_e=0.02, num_pts=12):
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
        ring_z = np.full_like(ring_r, z0 + d_e)

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
    method="grad_matern",
    rbf_smooth=None,
    cmap="viridis",
    show_pts=True,  # noqa: FBT002
    variance_threshold=0.05,
    project_path=True,  # noqa: FBT002
    extra_points=None,
    n_inducing=None,
):
    """Plot the 2D landscape surface.

    If project_path evaluates to True, the plot maps into
    reaction valley coordinates
    (Progress $s$ vs Orthogonal Distance $d$).

    ```{versionadded} 0.1.0
    ```

    ```{versionchanged} 1.1.0
    Added the *project_path* parameter for reaction-valley coordinate projection.
    ```
    """
    log.info(f"Generating 2D surface using {method} (Projected: {project_path})...")

    r_start, p_start = rmsd_r[0], rmsd_p[0]
    r_end, p_end = rmsd_r[-1], rmsd_p[-1]

    vec_r, vec_p = r_end - r_start, p_end - p_start
    path_norm = np.hypot(vec_r, vec_p)
    u_r, u_p = vec_r / path_norm, vec_p / path_norm
    v_r, v_p = -u_p, u_r

    # --- 1. Grid Setup (Handles both Projection and Standard RMSD) ---
    if project_path:
        s_data = (rmsd_r - r_start) * u_r + (rmsd_p - p_start) * u_p
        d_data = (rmsd_r - r_start) * v_r + (rmsd_p - p_start) * v_p
        s_min, s_max = s_data.min(), s_data.max()
        d_min, d_max = d_data.min(), d_data.max()

        if extra_points is not None and len(extra_points) > 0:
            extra_s = (extra_points[:, 0] - r_start) * u_r + (
                extra_points[:, 1] - p_start
            ) * u_p
            extra_d = (extra_points[:, 0] - r_start) * v_r + (
                extra_points[:, 1] - p_start
            ) * v_p
            s_min, s_max = min(s_min, extra_s.min()), max(s_max, extra_s.max())
            d_min, d_max = min(d_min, extra_d.min()), max(d_max, extra_d.max())

        xg_1d = np.linspace(
            s_min - (s_max - s_min) * 0.1, s_max + (s_max - s_min) * 0.1, 150
        )
        # Compute symmetric Y-grid based on X-span to fill the 1:1 plot
        x_span = xg_1d.max() - xg_1d.min()
        y_half = x_span / 2
        yg_1d = np.linspace(-y_half, y_half, 150)
        xg, yg = np.meshgrid(xg_1d, yg_1d)
    else:
        r_min, r_max = rmsd_r.min(), rmsd_r.max()
        p_min, p_max = rmsd_p.min(), rmsd_p.max()
        if extra_points is not None:
            r_min, r_max = (
                min(r_min, extra_points[:, 0].min()),
                max(r_max, extra_points[:, 0].max()),
            )
            p_min, p_max = (
                min(p_min, extra_points[:, 1].min()),
                max(p_max, extra_points[:, 1].max()),
            )

        xg_1d = np.linspace(
            r_min - (r_max - r_min) * 0.1, r_max + (r_max - r_min) * 0.1, 150
        )
        yg_1d = np.linspace(
            p_min - (p_max - p_min) * 0.1, p_max + (p_max - p_min) * 0.1, 150
        )
        xg, yg = np.meshgrid(xg_1d, yg_1d)

    if step_data is not None:
        actual_nimags = int(np.sum(step_data == step_data.max()))
    else:
        actual_nimags = None

    # --- 2. Hyperparameter Optimization ---
    if (
        method == "grad_imq"
        and len(rmsd_r) > NYSTROM_THRESHOLD
    ):
        log.warning(
            "More than %d points, switching to Nystrom",
            NYSTROM_THRESHOLD,
        )
        method = method + "_ny"
    model_class = get_surface_model(method)
    is_gradient_model = method.startswith("grad_")
    _MIN_RBF_SMOOTH = 1e-4
    h_ls = rbf_smooth if rbf_smooth and rbf_smooth > _MIN_RBF_SMOOTH else 0.5
    h_noise = 1e-2

    mask_opt = (
        (step_data == step_data.max())
        if step_data is not None
        else np.ones(len(z_data), dtype=bool)
    )
    # Build extra kwargs for Nystrom model
    _approx_kwargs = {}
    if "_ny" in method and n_inducing is not None:
        _approx_kwargs["n_inducing"] = n_inducing

    opt_kwargs = {
        "x": np.column_stack([rmsd_r, rmsd_p])[mask_opt],
        "y": z_data[mask_opt],
        "ls": h_ls,
        "smoothing": h_noise,
        "nimags": len(z_data),
        "optimize": True,
        **_approx_kwargs,
    }
    if is_gradient_model:
        opt_kwargs["gradients"] = np.column_stack([grad_r, grad_p])[mask_opt]

    try:
        learner = (
            model_class(**opt_kwargs)
            if is_gradient_model
            else model_class(
                x_obs=opt_kwargs["x"],
                y_obs=opt_kwargs["y"],
                **opt_kwargs,
            )
        )
        best_ls = getattr(learner, "ls", getattr(learner, "epsilon", h_ls))
        best_noise = getattr(learner, "noise", getattr(learner, "sm", h_noise))
    except Exception as e:
        log.warning(f"Optimization failed: {e}")
        best_ls, best_noise = h_ls, h_noise

    # --- 3. Prediction and Variance ---
    if project_path:
        grid_pts_eval = np.column_stack(
            [
                r_start + xg.ravel() * u_r + yg.ravel() * v_r,
                p_start + xg.ravel() * u_p + yg.ravel() * v_p,
            ]
        )
    else:
        grid_pts_eval = np.column_stack([xg.ravel(), yg.ravel()])

    _grad_stack = np.column_stack([grad_r, grad_p]) if grad_r is not None else None
    rbf = model_class(
        x=np.column_stack([rmsd_r, rmsd_p]),
        y=z_data,
        gradients=_grad_stack,
        length_scale=best_ls,
        smoothing=best_noise,
        optimize=False,
        nimags=actual_nimags,
        **_approx_kwargs,
    )

    zg = np.array(rbf(grid_pts_eval).reshape(xg.shape))
    var_grid = (
        np.array(rbf.predict_var(grid_pts_eval).reshape(xg.shape))
        if hasattr(rbf, "predict_var")
        else None
    )
    var_grid = ndimage.gaussian_filter(var_grid, sigma=2)  # smoothing variances

    # --- 4. Plotting ---
    ax.contourf(xg, yg, zg, levels=20, cmap=cmap, alpha=0.75, zorder=10)
    # NOTE(rg): this is not the "absolute" variance but the relative one
    if var_grid is not None:
        # Get the actual min and max variance currently in the grid
        v_min, v_max = var_grid.min(), var_grid.max()
        v_range = v_max - v_min

        # Calculate levels as 5%, 95%, and the user requested threshold of the
        # visual range with an epsilon to avoid errors for flat variances
        v_levs = [
            v_min + 0.05 * v_range + 1e-6,
            v_min + variance_threshold * v_range + 1e-6,
            v_min + 0.95 * v_range + 1e-6,
        ]

        v_con = ax.contour(
            xg,
            yg,
            var_grid,
            levels=v_levs,
            colors="black",
            linestyles="dashed",
            alpha=0.8,
            zorder=12,
        )
        ax.clabel(
            v_con,
            inline=True,
            fontsize=8,
            inline_spacing=50,
            fmt=lambda x: r"$\sigma^2 = $" + f"{x:.2g}",
        )

    if show_pts:
        plot_x = s_data if project_path else rmsd_r
        plot_y = d_data if project_path else rmsd_p

        # When Nystrom is active, fade non-inducing points so the user
        # can see which steps actually feed the surface fit.
        if "_ny" in method and step_data is not None and actual_nimags is not None:
            n_ind = n_inducing if n_inducing is not None else NYSTROM_N_INDUCING_DEFAULT
            keep = nystrom_paths_needed(n_ind, actual_nimags)
            max_step = step_data.max()
            inducing_mask = step_data >= (max_step - keep + 1)
            # Background (non-inducing) points
            ax.scatter(
                plot_x[~inducing_mask],
                plot_y[~inducing_mask],
                c="k",
                s=8,
                marker=".",
                alpha=0.15,
                zorder=39,
            )
            # Inducing points
            ax.scatter(
                plot_x[inducing_mask],
                plot_y[inducing_mask],
                c="k",
                s=12,
                marker=".",
                alpha=0.6,
                zorder=40,
            )
        else:
            ax.scatter(
                plot_x,
                plot_y,
                c="k",
                s=12,
                marker=".",
                alpha=0.6,
                zorder=40,
            )


def plot_landscape_path_overlay(
    ax,
    r,
    p,
    z,
    cmap,
    z_label,
    project_path=True,  # noqa: FBT002
):
    """Overlay the colored path line on the landscape.

    Mapped to the chosen coordinate basis.

    ```{versionadded} 0.1.0
    ```

    ```{versionchanged} 1.1.0
    Added the *project_path* parameter for reaction-valley coordinate projection.
    ```
    """
    if project_path:
        r_start, p_start = r[0], p[0]
        r_end, p_end = r[-1], p[-1]

        vec_r = r_end - r_start
        vec_p = p_end - p_start
        path_norm = np.hypot(vec_r, vec_p)

        u_r = vec_r / path_norm
        u_p = vec_p / path_norm
        v_r = -u_p
        v_p = u_r

        plot_x = (r - r_start) * u_r + (p - p_start) * u_p
        plot_y = (r - r_start) * v_r + (p - p_start) * v_p
    else:
        plot_x = r
        plot_y = p

    points = np.array([plot_x, plot_y]).T.reshape(-1, 1, 2)
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
        plot_x,
        plot_y,
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
