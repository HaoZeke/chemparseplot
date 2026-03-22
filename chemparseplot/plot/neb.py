import io
import logging
import shutil
import subprocess
import tempfile
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as tri
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

from chemparseplot.parse.projection import (
    compute_projection_basis,
    inverse_sd_to_ab,
    project_to_sd,
)

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
    ase_write(buf, atoms, format="png", rotation=rotation, show_unit_cell=0, scale=100)
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


def _apply_perspective_tilt(atoms, tilt_deg=8.0):
    """Apply a small off-axis rotation to reveal hidden atoms.

    Uses Rodrigues formula to rotate around an axis perpendicular to
    the viewing direction. This prevents atoms from occluding each
    other in orthographic projection without significantly distorting
    the view.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to tilt (modified in place).
    tilt_deg : float
        Tilt angle in degrees. 5-10 is usually enough.
    """
    theta = np.radians(tilt_deg)
    # Tilt axis: diagonal in the xy-plane (not aligned with any principal axis)
    k = np.array([1.0, 0.7, 0.0])
    k = k / np.linalg.norm(k)

    # Rodrigues rotation matrix: R = I cos(t) + (1-cos(t)) k*kT + sin(t) K
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = cos_t * np.eye(3) + (1 - cos_t) * np.outer(k, k) + sin_t * K

    center = atoms.positions.mean(axis=0)
    atoms.positions = (R @ (atoms.positions - center).T).T + center


def _parse_rotation_angles(rotation_str):
    """Parse ASE-style rotation string into (rx, ry, rz) degrees.

    E.g. ``"0x,90y,0z"`` -> ``(0, 90, 0)``.
    """
    import re

    rx, ry, rz = 0.0, 0.0, 0.0
    for part in rotation_str.replace(" ", "").split(","):
        m = re.match(r"(-?[\d.]+)([xyz])", part.strip())
        if m:
            val, axis = float(m.group(1)), m.group(2)
            if axis == "x":
                rx = val
            elif axis == "y":
                ry = val
            else:
                rz = val
    return rx, ry, rz


def _render_xyzrender(atoms, rotation="auto", canvas_size=400):
    """Render an ASE Atoms object to a numpy RGBA array via xyzrender.

    Uses the ``paton`` preset with hydrogens visible for ball-and-stick
    style rendering.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to render.
    rotation : str
        ``"auto"`` (default) uses xyzrender's auto-orientation.
        Any ASE-style string (e.g. ``"0x,90y,0z"``) disables auto-orient
        and pre-rotates the atoms.
    canvas_size : int
        Output image width/height in pixels (passed as ``-S``).

    Returns
    -------
    numpy.ndarray
        RGBA image array with shape ``(H, W, 4)`` and float dtype.
    """
    from ase.io import write as _ase_write

    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as xyz_fh:
        xyz_path = xyz_fh.name
    png_path = xyz_path.rsplit(".", 1)[0] + ".png"

    try:
        atoms = atoms.copy()
        has_custom_rotation = rotation != "auto"

        if has_custom_rotation:
            rx, ry, rz = _parse_rotation_angles(rotation)
            # Pre-rotate and disable auto-orient
            atoms.rotate(rx, "x", center="COP")
            atoms.rotate(ry, "y", center="COP")
            atoms.rotate(rz, "z", center="COP")

        _ase_write(xyz_path, atoms, format="xyz")
        cmd = [
            "xyzrender",
            xyz_path,
            "-o",
            png_path,
            "-S",
            str(canvas_size),
            "--config",
            "paton",
            "--hy",
            "-t",
        ]
        if has_custom_rotation:
            cmd.append("--no-orient")
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


def _render_atoms(
    atoms, renderer, zoom, rotation, canvas_size=400, perspective_tilt=0.0
):
    """Dispatch rendering to the selected backend.

    All backends return a numpy RGBA image array.

    Parameters
    ----------
    rotation : str
        ASE-style rotation string (e.g. ``"0x,90y,0z"``). Applied
        uniformly across all backends.
    perspective_tilt : float
        Small off-axis tilt in degrees to reveal occluded atoms.
        0 disables. 5-10 is usually enough.
    """
    if perspective_tilt > 0:
        atoms = atoms.copy()
        _apply_perspective_tilt(atoms, perspective_tilt)
    # "auto" means: xyzrender auto-orients, other backends use default side view
    effective_rotation = rotation if rotation != "auto" else "0x,90y,0z"

    if renderer == "xyzrender":
        _check_xyzrender()
        return _render_xyzrender(atoms, rotation=rotation, canvas_size=canvas_size)
    elif renderer == "solvis":
        return _render_solvis(atoms, rotation=effective_rotation, canvas_size=canvas_size)
    elif renderer == "ovito":
        return _render_ovito(atoms, rotation=effective_rotation, canvas_size=canvas_size)
    else:
        return render_structure_to_image(atoms, zoom, effective_rotation)


def _render_solvis(atoms, rotation="0x,90y,0z", canvas_size=400):
    """Render via solvis (ball-and-stick with PyVista).

    Requires: ``uvx --from solvis-tools python -c "import solvis"``

    Returns
    -------
    numpy.ndarray
        RGBA image array.
    """
    try:
        from solvis.visualization import AtomicPlotter
    except ImportError as exc:
        msg = "solvis not installed. Install with: pip install solvis-tools"
        raise RuntimeError(msg) from exc

    from ase.io import write as _ase_write
    from ase.neighborlist import natural_cutoffs, neighbor_list

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as png_fh:
        png_path = png_fh.name

    try:
        import pyvista as pv

        pv.start_xvfb()  # headless rendering without DISPLAY

        plotter = AtomicPlotter(
            interactive_mode=False,
            window_size=[canvas_size, canvas_size],
            depth_peeling=True,
            shadows=False,
        )
        # Transparent background
        plotter.plotter.set_background([1.0, 1.0, 1.0, 0.0])

        # Apply rotation
        atoms = atoms.copy()
        rx, ry, rz = _parse_rotation_angles(rotation)
        if rx != 0 or ry != 0 or rz != 0:
            atoms.rotate(rx, "x", center="COP")
            atoms.rotate(ry, "y", center="COP")
            atoms.rotate(rz, "z", center="COP")

        positions = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()

        # CPK hex colors from ASE jmol palette
        from ase.data.colors import jmol_colors

        def _rgb_to_hex(rgb):
            r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            return f"#{r:02x}{g:02x}{b:02x}"

        colors = [_rgb_to_hex(jmol_colors[z]) for z in numbers]

        plotter.add_atoms_as_spheres(positions, colors, radius=0.3)

        # Bonds via ASE neighbor list
        cutoffs = natural_cutoffs(atoms, mult=1.1)
        i_idx, j_idx = neighbor_list("ij", atoms, cutoffs)
        for bond_idx, (i, j) in enumerate(zip(i_idx, j_idx)):
            if i < j:
                plotter.add_bond(
                    positions[i], positions[j],
                    colors[i], colors[j],
                    f"bond_{bond_idx}",
                    radius=0.08,
                )

        # Render with transparency
        img_data = plotter.plotter.screenshot(
            png_path, transparent_background=True, return_img=True
        )
        plotter.close()
        # Normalize to float [0,1] if needed
        if img_data.dtype == np.uint8:
            img_data = img_data.astype(np.float32) / 255.0
    finally:
        import os

        try:
            os.unlink(png_path)
        except OSError:
            pass
    return img_data


def _render_ovito(atoms, rotation="0x,90y,0z", canvas_size=400):
    """Render via OVITO Python (high-quality off-screen rendering).

    Requires: ``pip install ovito``

    Returns
    -------
    numpy.ndarray
        RGBA image array.
    """
    try:
        from ovito.io.ase import ase_to_ovito
        from ovito.pipeline import Pipeline, StaticSource
        from ovito.vis import OpenGLRenderer, Viewport
    except ImportError as exc:
        msg = "ovito not installed. Install with: pip install ovito"
        raise RuntimeError(msg) from exc

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as png_fh:
        png_path = png_fh.name

    try:
        # Apply rotation before converting to OVITO
        atoms = atoms.copy()
        rx, ry, rz = _parse_rotation_angles(rotation)
        if rx != 0 or ry != 0 or rz != 0:
            atoms.rotate(rx, "x", center="COP")
            atoms.rotate(ry, "y", center="COP")
            atoms.rotate(rz, "z", center="COP")

        data = ase_to_ovito(atoms)
        pipeline = Pipeline(source=StaticSource(data_collection=data))
        pipeline.add_to_scene()

        vp = Viewport(type=Viewport.Type.Ortho)
        vp.zoom_all()

        vp.render_image(
            size=(canvas_size, canvas_size),
            filename=png_path,
            background=(1.0, 1.0, 1.0),
            renderer=OpenGLRenderer(),
        )
        pipeline.remove_from_scene()
        img_data = plt.imread(png_path)
    finally:
        import os

        try:
            os.unlink(png_path)
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
    renderer="xyzrender",
    col_spacing=1.5,
    show_dividers=False,  # noqa: FBT002
    divider_color="gray",
    divider_style="--",
    perspective_tilt=0.0,
) -> Any:
    """Renders a horizontal gallery of atomic structures.

    Parameters
    ----------
    renderer : str
        Rendering backend: ``"ase"``, ``"xyzrender"``, ``"solvis"``,
        or ``"ovito"``.
    col_spacing : float
        Horizontal spacing between structure images in data units.
    show_dividers : bool
        Draw vertical divider lines between structures.
    divider_color : str
        Color for divider lines.
    divider_style : str
        Linestyle for divider lines (e.g. ``"--"``, ``"-"``, ``":"``).

    ```{versionadded} 0.1.0
    ```

    ```{versionchanged} 1.2.0
    Added the *renderer* parameter.
    ```

    ```{versionchanged} 1.5.0
    Added *col_spacing*, *show_dividers*, *divider_color*, *divider_style*
    parameters. Added ``"solvis"`` and ``"ovito"`` renderer backends.
    ```
    """
    ax.axis("off")
    n_plot = len(atoms_list)
    n_cols = min(n_plot, max_cols)
    n_rows = (n_plot + max_cols - 1) // max_cols
    row_step = 8.5

    col_step = col_spacing
    ax.set_xlim(-0.5, (n_cols - 1) * col_step + 0.5)
    y_min = -((n_rows - 1) * row_step) - 0.8
    y_max = 0.6
    ax.set_ylim(y_min, y_max)

    # Adaptive font size: shrink for many items
    label_fontsize = min(11, max(7, 14 - n_plot))

    for i, atoms in enumerate(atoms_list):
        col = i % max_cols
        row = i // max_cols
        x_pos, y_pos = col * col_step, -row * row_step

        img_data = _render_atoms(
            atoms, renderer, zoom, rotation, perspective_tilt=perspective_tilt
        )

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
                fontsize=label_fontsize,
                color=theme_color,
                fontweight="bold",
            )

    # Divider lines between structures
    if show_dividers:
        for col in range(1, n_cols):
            x_div = (col - 0.5) * col_step
            ax.axvline(
                x_div,
                color=divider_color,
                linestyle=divider_style,
                linewidth=0.8,
                alpha=0.5,
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
    renderer="xyzrender",
    perspective_tilt=0.0,
) -> Any:
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
    img_data = _render_atoms(
        atoms, renderer, zoom, rotation, perspective_tilt=perspective_tilt
    )
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
    ax, rc, energy, f_para, color, alpha, zorder, method="hermite", smoothing=None,
    label=None,
) -> Any:
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

        ax.plot(x_fine, y_fine, color=color, alpha=alpha, zorder=zorder, label=label)
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
        ax.plot(rc, energy, color=color, alpha=alpha, zorder=zorder, label=label)


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
) -> Any:
    """Plot the 2D landscape surface using reaction valley projection.

    If project_path evaluates to True, the plot maps into
    reaction valley coordinates
    (Progress $s$ vs Orthogonal Distance $d$).

    Implements 2D reaction valley projection method from cite:[goswami2026valley].
    The method rotates the RMSD plane into reaction progress and orthogonal
    deviation coordinates.

    ```{versionadded} 0.1.0
    ```

    ```{versionchanged} 1.1.0
    Added the *project_path* parameter for reaction-valley coordinate projection.
    ```
    """
    log.info(f"Generating 2D surface using {method} (Projected: {project_path})...")

    basis = compute_projection_basis(rmsd_r, rmsd_p) if project_path else None

    # --- 1. Grid Setup (Handles both Projection and Standard RMSD) ---
    if project_path:
        s_data, d_data = project_to_sd(rmsd_r, rmsd_p, basis)
        s_min, s_max = s_data.min(), s_data.max()
        d_min, d_max = d_data.min(), d_data.max()

        if extra_points is not None and len(extra_points) > 0:
            extra_s, extra_d = project_to_sd(
                extra_points[:, 0], extra_points[:, 1], basis
            )
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
    if method == "grad_imq" and len(rmsd_r) > NYSTROM_THRESHOLD:
        log.warning(
            "More than %d points, switching to Nystrom",
            NYSTROM_THRESHOLD,
        )
        method = method + "_ny"
    model_class = get_surface_model(method)
    is_gradient_model = method.startswith("grad_")
    _min_rbf_smooth = 1e-4
    h_ls = rbf_smooth if rbf_smooth and rbf_smooth > _min_rbf_smooth else 0.5
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
                smoothing=opt_kwargs["smoothing"],
                optimize=opt_kwargs["optimize"],
            )
        )
        best_ls = getattr(learner, "ls", getattr(learner, "epsilon", h_ls))
        best_noise = getattr(learner, "noise", getattr(learner, "sm", h_noise))
    except Exception as e:
        log.warning(f"Optimization failed: {e}")
        best_ls, best_noise = h_ls, h_noise

    # --- 3. Prediction and Variance ---
    if project_path:
        eval_a, eval_b = inverse_sd_to_ab(xg.ravel(), yg.ravel(), basis)
        grid_pts_eval = np.column_stack([eval_a, eval_b])
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

        # Calculate levels as 5%, threshold%, and 95% of the variance range.
        # Skip contours entirely if variance is flat (avoids matplotlib error
        # "Contour levels must be increasing").
        if v_range < 1e-10:
            v_con = None
        else:
            v_levs = sorted(set([
                v_min + 0.05 * v_range,
                v_min + variance_threshold * v_range,
                v_min + 0.95 * v_range,
            ]))

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
    all_r=None,
    all_p=None,
    all_z=None,
) -> Any:
    """Overlay the colored path line on the landscape.

    Mapped to the chosen coordinate basis. When ``all_r/all_p/all_z`` arrays
    are provided (all NEB iterations), a triangulated filled contour is drawn
    as the background so the landscape is never empty.

    ```{versionadded} 0.1.0
    ```

    ```{versionchanged} 1.1.0
    Added the *project_path* parameter for reaction-valley coordinate projection.
    ```

    ```{versionchanged} 1.6.0
    Added *all_r*, *all_p*, *all_z* for triangulated background contours.
    ```
    """
    if project_path:
        basis = compute_projection_basis(r, p)
        plot_x, plot_y = project_to_sd(r, p, basis)
    else:
        plot_x = r
        plot_y = p

    try:
        colormap = plt.get_cmap(cmap)
    except ValueError:
        colormap = plt.get_cmap("viridis")

    norm = plt.Normalize(z.min(), z.max())

    # --- Triangulated background contours from all iterations ---
    if all_r is not None and all_p is not None and all_z is not None:
        if project_path:
            bg_x, bg_y = project_to_sd(all_r, all_p, basis)
        else:
            bg_x, bg_y = all_r, all_p
        try:
            triang = tri.Triangulation(bg_x, bg_y)
            bg_norm = plt.Normalize(all_z.min(), all_z.max())
            ax.tricontourf(
                triang, all_z, levels=20, cmap=colormap, alpha=0.6,
                norm=bg_norm, zorder=5,
            )
            ax.tricontour(
                triang, all_z, levels=10, colors="k", alpha=0.15,
                linewidths=0.4, zorder=6,
            )
        except Exception:
            log.debug("Triangulation failed, skipping background contours.")

    points = np.array([plot_x, plot_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

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


def plot_mmf_peaks_overlay(
    ax,
    peak_rmsd_r,
    peak_rmsd_p,
    peak_energies,
    project_path=True,  # noqa: FBT002
    path_rmsd_r=None,
    path_rmsd_p=None,
) -> None:
    """Overlay MMF (mode-following) refinement peak positions on the landscape.

    Used for OCI-NEB/RONEB visualization to show where dimer refinement
    was applied along the band.

    Parameters
    ----------
    ax
        Matplotlib axes (same as the landscape plot).
    peak_rmsd_r, peak_rmsd_p
        RMSD coordinates of MMF peak structures.
    peak_energies
        Energy values at the peak positions.
    project_path
        Whether to project into (s, d) coordinates.
    path_rmsd_r, path_rmsd_p
        RMSD arrays of the main NEB path, used to define the projection
        basis. Required when ``project_path=True``. If None, falls back
        to computing basis from the peaks themselves (less accurate).

    ```{versionadded} 1.5.0
    ```
    """
    if len(peak_rmsd_r) == 0:
        return

    peak_r = np.asarray(peak_rmsd_r)
    peak_p = np.asarray(peak_rmsd_p)
    peak_e = np.asarray(peak_energies)

    if project_path:
        if path_rmsd_r is not None and path_rmsd_p is not None:
            basis = compute_projection_basis(
                np.asarray(path_rmsd_r), np.asarray(path_rmsd_p)
            )
        else:
            basis = compute_projection_basis(peak_r, peak_p)
        plot_x, plot_y = project_to_sd(peak_r, peak_p, basis)
    else:
        plot_x, plot_y = peak_r, peak_p

    ax.scatter(
        plot_x,
        plot_y,
        c=peak_e,
        cmap="coolwarm",
        marker="*",
        s=200,
        edgecolors="black",
        linewidths=1.0,
        zorder=50,
        label="MMF peaks",
    )


def plot_neb_evolution(
    ax,
    step_rmsd_r_list: list[np.ndarray],
    step_rmsd_p_list: list[np.ndarray],
    project_path=True,  # noqa: FBT002
    cmap="Blues",
) -> None:
    """Show NEB band evolution across optimization iterations.

    Older bands are drawn with lower opacity; the final band is most visible.
    All bands are projected using the *final* band's basis so they share
    one consistent coordinate frame.

    Parameters
    ----------
    ax
        Matplotlib axes.
    step_rmsd_r_list
        List of RMSD-R arrays, one per NEB iteration.
    step_rmsd_p_list
        List of RMSD-P arrays, one per NEB iteration.
    project_path
        Whether to project into (s, d) coordinates.
    cmap
        Colormap for fading bands (older = lighter).

    ```{versionadded} 1.5.0
    ```
    """
    n_steps = len(step_rmsd_r_list)
    if n_steps == 0:
        return

    colormap = plt.get_cmap(cmap)

    # Use the final band to define the projection basis for all bands
    final_basis = None
    if project_path:
        final_r = np.asarray(step_rmsd_r_list[-1])
        final_p = np.asarray(step_rmsd_p_list[-1])
        final_basis = compute_projection_basis(final_r, final_p)

    for i, (rr, rp) in enumerate(zip(step_rmsd_r_list, step_rmsd_p_list)):
        rr = np.asarray(rr)
        rp = np.asarray(rp)

        if project_path:
            px, py = project_to_sd(rr, rp, final_basis)
        else:
            px, py = rr, rp

        alpha = 0.15 + 0.85 * (i / max(n_steps - 1, 1))
        color = colormap(0.3 + 0.7 * (i / max(n_steps - 1, 1)))
        ax.plot(px, py, "o-", color=color, alpha=alpha, markersize=2, linewidth=1)


def plot_orca_neb_profile(
    neb_data: dict[str, Any],
    output: Path,
    *,
    width: float = 7.0,
    height: float = 5.0,
    dpi: int = 200,
) -> None:
    """Plot ORCA NEB energy profile from OPI-parsed data.

    Parameters
    ----------
    neb_data
        Dictionary from parse_orca_neb() containing:
        - energies: list of energies in eV
        - n_images: number of images
        - barrier_forward, barrier_reverse: barrier heights
    output
        Output file path
    width, height
        Figure dimensions in inches
    dpi
        Output resolution

    Example
    -------
    >>> from chemparseplot.parse.orca.neb import parse_orca_neb
    >>> from chemparseplot.plot.neb import plot_orca_neb_profile
    >>> data = parse_orca_neb("job", Path("calc"))
    >>> plot_orca_neb_profile(data, "neb_profile.pdf")
    """
    import matplotlib.pyplot as plt

    energies = neb_data.get("energies", [])
    n_images = neb_data.get("n_images", len(energies))

    if not energies:
        msg = "No energy data in neb_data"
        raise ValueError(msg)

    # Create figure
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    # Plot energy profile
    image_indices = list(range(n_images))
    ax.plot(image_indices, energies, "o-", linewidth=2, markersize=8)

    # Label reactant, product, and saddle
    if len(energies) >= 3:
        ax.plot(0, energies[0], "go", markersize=12, label="Reactant")
        ax.plot(-1, energies[-1], "ro", markersize=12, label="Product")

        # Find saddle point
        saddle_idx = energies.index(max(energies))
        if saddle_idx != 0 and saddle_idx != len(energies) - 1:
            ax.plot(saddle_idx, energies[saddle_idx], "ys", markersize=12, label="Saddle")

    # Add barrier annotations
    barrier_fwd = neb_data.get("barrier_forward")
    neb_data.get("barrier_reverse")

    if barrier_fwd is not None and barrier_fwd > 0:
        ax.annotate(
            f"ΔE‡ = {barrier_fwd:.2f} eV",
            xy=(saddle_idx, energies[saddle_idx]),
            xytext=(saddle_idx + 1, energies[saddle_idx] + 0.5),
            arrowprops={"arrowstyle": "->", "color": "black"},
            fontsize=10,
        )

    # Labels and formatting
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("ORCA NEB Energy Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.minorticks_on()

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_orca_neb_energy_profile(
    neb_data: dict[str, Any],
    output: Path,
    *,
    width: float = 5.37,
    height: float = 5.37,
    dpi: int = 200,
    method: str = "hermite",
    smoothing: Any = None,
) -> None:
    """Plot ORCA NEB energy profile using existing eOn-style plotting.

    Creates publication-quality energy profile similar to eOn NEB plots.
    Uses the same plotting functions as eOn NEB for consistency.

    Parameters
    ----------
    neb_data
        Dictionary from parse_orca_neb() containing:
        - energies: array of energies in eV
        - rmsd_r, rmsd_p: RMSD coordinates (optional)
        - grad_r, grad_p: gradients (optional)
        - n_images: number of images
    output
        Output file path
    width, height
        Figure dimensions in inches
    dpi
        Output resolution
    method
        Interpolation method: 'hermite' or 'spline'
    smoothing
        Smoothing parameters

    Example
    -------
    >>> from chemparseplot.parse.orca.neb import parse_orca_neb
    >>> from chemparseplot.plot.neb import plot_orca_neb_energy_profile
    >>> data = parse_orca_neb("job", Path("calc"))
    >>> plot_orca_neb_energy_profile(data, "orca_neb_profile.pdf")
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    energies = neb_data.get("energies", [])
    rmsd_r = neb_data.get("rmsd_r")
    rmsd_p = neb_data.get("rmsd_p")
    grad_r = neb_data.get("grad_r")
    neb_data.get("grad_p")
    n_images = neb_data.get("n_images", len(energies))
    barrier_fwd = neb_data.get("barrier_forward")
    neb_data.get("barrier_reverse")

    if len(energies) == 0:
        msg = "No energy data in neb_data"
        raise ValueError(msg)

    # Use RMSD as reaction coordinate if available, otherwise use image index
    if rmsd_r is not None and rmsd_p is not None:
        # Use progress coordinate (similar to eOn)
        rc = rmsd_r
        f_para = grad_r if grad_r is not None else np.zeros_like(rc)
        xlabel = r"RMSD from Reactant ($\AA$)"
    else:
        rc = np.arange(n_images)
        f_para = np.zeros(n_images)
        xlabel = "Image Index"

    # Create figure
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0])

    # Plot energy profile using same function as eOn
    from chemparseplot.plot.theme import get_theme

    get_theme("ruhi")

    color = "#1f77b4"  # Default blue color
    plot_energy_path(
        ax,
        rc,
        energies,
        f_para,
        color=color,
        alpha=1.0,
        zorder=10,
        method=method,
        smoothing=smoothing,
    )

    # Label key points
    if n_images >= 2:
        ax.plot(rc[0], energies[0], "go", markersize=10, label="Reactant", zorder=20)
        ax.plot(rc[-1], energies[-1], "ro", markersize=10, label="Product", zorder=20)

        # Find and label saddle
        saddle_idx = int(np.argmax(energies))
        if saddle_idx != 0 and saddle_idx != n_images - 1:
            ax.plot(
                rc[saddle_idx],
                energies[saddle_idx],
                "ys",
                markersize=12,
                label="Saddle",
                zorder=20,
            )

            # Add barrier annotation
            if barrier_fwd is not None and barrier_fwd > 0:
                ax.annotate(
                    f"$\\Delta E^\\ddagger = {barrier_fwd:.2f}$ eV",
                    xy=(rc[saddle_idx], energies[saddle_idx]),
                    xytext=(rc[saddle_idx] + 0.5, energies[saddle_idx] + 0.5),
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.5},
                    fontsize=9,
                    zorder=30,
                )

    # Labels and formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy (eV)")
    ax.set_title("ORCA NEB Energy Profile")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.minorticks_on()

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_orca_neb_landscape(
    neb_data: dict[str, Any],
    output: Path,
    *,
    width: float = 5.37,
    height: float = 5.37,
    dpi: int = 200,
    method: str = "grad_matern",
    project_path: bool = True,
) -> None:
    """Plot ORCA NEB 2D landscape using existing eOn-style plotting.

    Creates publication-quality landscape plot similar to eOn NEB plots.
    Uses the same plotting functions as eOn NEB for consistency.

    Parameters
    ----------
    neb_data
        Dictionary from parse_orca_neb() containing:
        - energies: array of energies in eV
        - rmsd_r, rmsd_p: RMSD coordinates
        - grad_r, grad_p: gradients
        - n_images: number of images
    output
        Output file path
    width, height
        Figure dimensions in inches
    dpi
        Output resolution
    method
        Surface interpolation method
    project_path
        Whether to project into reaction valley coordinates

    Example
    -------
    >>> from chemparseplot.parse.orca.neb import parse_orca_neb
    >>> from chemparseplot.plot.neb import plot_orca_neb_landscape
    >>> data = parse_orca_neb("job", Path("calc"))
    >>> plot_orca_neb_landscape(data, "orca_neb_landscape.pdf")
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    energies = neb_data.get("energies", [])
    rmsd_r = neb_data.get("rmsd_r")
    rmsd_p = neb_data.get("rmsd_p")
    grad_r = neb_data.get("grad_r")
    grad_p = neb_data.get("grad_p")

    # Convert to numpy arrays if lists
    rmsd_r = np.asarray(rmsd_r)
    rmsd_p = np.asarray(rmsd_p)
    energies = np.asarray(energies)

    if rmsd_r is None or rmsd_p is None:
        msg = (
            "RMSD coordinates required for landscape plot. "
            "Re-run ORCA calculation with geometry output enabled."
        )
        raise ValueError(msg)

    # Create figure
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0])

    # Get theme
    from chemparseplot.plot.theme import get_theme

    theme = get_theme("ruhi")
    cmap = theme.cmap_landscape

    # Plot 2D landscape surface using same function as eOn
    plot_landscape_surface(
        ax,
        rmsd_r,
        rmsd_p,
        grad_r if grad_r is not None else np.zeros_like(rmsd_r),
        grad_p if grad_p is not None else np.zeros_like(rmsd_p),
        energies,
        method=method,
        cmap=cmap,
        show_pts=True,
        project_path=project_path,
    )

    # Overlay path using same function as eOn
    plot_landscape_path_overlay(
        ax,
        rmsd_r,
        rmsd_p,
        energies,
        cmap=cmap,
        z_label="Energy (eV)",
        project_path=project_path,
    )

    # Labels and formatting
    if project_path:
        ax.set_xlabel(r"Reaction progress $s$ ($\AA$)")
        ax.set_ylabel(r"Orthogonal deviation $d$ ($\AA$)")
        ax.set_title("ORCA NEB Reaction Landscape")
    else:
        ax.set_xlabel(r"RMSD from Reactant ($\AA$)")
        ax.set_ylabel(r"RMSD from Product ($\AA$)")
        ax.set_title("ORCA NEB RMSD Landscape")

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.minorticks_on()

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
