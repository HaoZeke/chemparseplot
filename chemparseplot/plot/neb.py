import io
import logging
from collections import namedtuple
from dataclasses import dataclass

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from ase.io import write as ase_write
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ArrowStyle
from scipy.interpolate import (
    CubicHermiteSpline,
    griddata,
    splev,
    splrep,
)
from scipy.signal import savgol_filter

from rgpycrumbs.surfaces import get_surface_model

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
    method="grad_matern",
    rbf_smooth=None,
    cmap="viridis",
    show_pts=True,
    variance_threshold=0.05,
    project_path=True,
    extra_points=None,
):
    """
    Plots the 2D landscape surface. If project_path evaluates to True,
    the plot maps into a reaction valley coordinates (Progress $s$ vs Orthogonal Distance $d$).
    """
    log.info(f"Generating 2D surface using {method} (Projected: {project_path})...")

    r_start, p_start = rmsd_r[0], rmsd_p[0]
    r_end, p_end = rmsd_r[-1], rmsd_p[-1]

    vec_r = r_end - r_start
    vec_p = p_end - p_start
    path_norm = np.hypot(vec_r, vec_p)

    u_r = vec_r / path_norm
    u_p = vec_p / path_norm
    v_r = -u_p
    v_p = u_r

    if project_path:
        s_data = (rmsd_r - r_start) * u_r + (rmsd_p - p_start) * u_p
        d_data = (rmsd_r - r_start) * v_r + (rmsd_p - p_start) * v_p

        s_min, s_max = s_data.min(), s_data.max()
        d_min, d_max = d_data.min(), d_data.max()
        
        if extra_points is not None and len(extra_points) > 0:
            extra_r, extra_p = extra_points[:, 0], extra_points[:, 1]
            extra_s = (extra_r - r_start) * u_r + (extra_p - p_start) * u_p
            extra_d = (extra_r - r_start) * v_r + (extra_p - p_start) * v_p
            s_min, s_max = np.min([s_min, extra_s.min()]), np.max([s_max, extra_s.max()])
            d_min, d_max = np.min([d_min, extra_d.min()]), np.max([d_max, extra_d.max()])

        nx, ny = 150, 150
        x_margin = (s_max - s_min) * 0.1
        y_margin = (d_max - d_min) * 0.2
        if y_margin < 0.05:
            y_margin = 0.1

        xg_1d = np.linspace(s_min - x_margin, s_max + x_margin, nx)
        yg_1d = np.linspace(d_min - y_margin, d_max + y_margin, ny)
        xg, yg = np.meshgrid(xg_1d, yg_1d)

        plot_x_data = s_data
        plot_y_data = d_data

    else:
        r_min, r_max = rmsd_r.min(), rmsd_r.max()
        p_min, p_max = rmsd_p.min(), rmsd_p.max()

        if extra_points is not None and len(extra_points) > 0:
            r_min, r_max = np.min([r_min, extra_points[:, 0].min()]), np.max([r_max, extra_points[:, 0].max()])
            p_min, p_max = np.min([p_min, extra_points[:, 1].min()]), np.max([p_max, extra_points[:, 1].max()])

        nx, ny = 150, 150
        x_margin = (r_max - r_min) * 0.1
        y_margin = (p_max - p_min) * 0.1

        xg_1d = np.linspace(r_min - x_margin, r_max + x_margin, nx)
        yg_1d = np.linspace(p_min - y_margin, p_max + y_margin, ny)
        xg, yg = np.meshgrid(xg_1d, yg_1d)

    if method == "grid":
        zg = griddata((plot_x_data, plot_y_data), z_data, (xg, yg), method="cubic")
    else:
        ModelClass = get_surface_model(method)
        is_gradient_model = method.startswith("grad_")

        # --- 2. Hyperparameter Optimization (Learning Phase) ---
        # We learn physics (length_scale) from the *real* data (final path),
        # without synthetic augmentation artifacts.
        # 'rbf_smooth' passed from estimate_rbf_smoothing is actually a distance metric.
        # It is a good GUESS for length_scale, but terrible for noise (too high).

        # Initial Guess for Length Scale (Correlation distance)
        heuristic_ls = rbf_smooth if rbf_smooth is not None and rbf_smooth > 1e-4 else 0.5

        # Initial Guess for Noise (Data fidelity)
        heuristic_noise = 1e-2
        best_ls, best_noise = heuristic_ls, heuristic_noise

        pts_clean = np.column_stack([rmsd_r, rmsd_p])
        vals_clean = z_data

        if grad_r is not None:
            grads_clean = np.column_stack([grad_r, grad_p])
        else:
            grads_clean = np.zeros_like(pts_clean)

        # Optimization Subset Selection
        if step_data is not None:
            max_step = step_data.max()
            is_final = step_data == max_step
            mask_opt = (
                is_final if np.sum(is_final) > 3 else np.ones(len(z_data), dtype=bool)
            )
        else:
            mask_opt = np.ones(len(z_data), dtype=bool)

        log.info(f"Optimizing hyperparameters on subset of {np.sum(mask_opt)} points...")

        # Jitter to prevent singular matrix during opt
        rng = np.random.default_rng(42)
        jitter_opt = rng.normal(0, 1e-6, size=pts_clean[mask_opt].shape)

        opt_kwargs = {
            "x": pts_clean[mask_opt] + jitter_opt,
            "y": vals_clean[mask_opt],
            "smoothing": heuristic_noise,
            "length_scale": heuristic_ls,
            "ls": heuristic_ls,
            "optimize": True,
        }

        if is_gradient_model:
            opt_kwargs["gradients"] = grads_clean[mask_opt]

        try:
            if is_gradient_model:
                learner = ModelClass(**opt_kwargs)
            else:
                learner = ModelClass(
                    x_obs=opt_kwargs["x"], y_obs=opt_kwargs["y"], **opt_kwargs
                )

            if hasattr(learner, "ls"):
                best_ls = learner.ls
            if hasattr(learner, "epsilon"):
                best_ls = learner.epsilon  # IMQ uses epsilon
            if hasattr(learner, "sm"):
                best_noise = learner.sm  # TPS uses sm
            if hasattr(learner, "noise"):
                best_noise = learner.noise

            log.info(f"Learned Params :: LS/Eps: {best_ls:.4f}, Noise: {best_noise:.4f}")

        except Exception as e:
            log.warning(f"Optimization failed ({e}). Using heuristics.")
            best_ls, best_noise = heuristic_ls, heuristic_noise

        # --- 3. Data Augmentation (Visualization Phase) ---
        # Now we build the full dataset for the final high-res surface.
        # We add "helper" points to force the surface to look good (bowls at minima).

        # A. Minima Augmentation
        # Increase radius/dE slightly so the GP actually "sees" the well features
        # Radius 0.001 is often smaller than the grid resolution
        aug_radius = 0.02
        aug_dE = 0.05

        if not is_gradient_model:
            r_aug, p_aug, z_aug = _augment_minima_points(
                rmsd_r, rmsd_p, z_data, radius=aug_radius, dE=aug_dE, num_pts=12
            )
        else:
            # For gradient models, we usually don't need ring augmentation if gradients are correct.
            # But if we do, we must be careful. Let's trust the gradients at the minima
            # (which should be zero) instead of adding fake rings.
            r_aug, p_aug, z_aug = rmsd_r, rmsd_p, z_data

        if is_gradient_model:
            final_pts = np.column_stack([r_aug, p_aug])
            final_vals = z_aug
            # Ensure gradients match shape (no augmentation logic needed if we skip rings)
            final_grads = (
                np.column_stack([grad_r, grad_p])
                if grad_r is not None
                else np.zeros_like(final_pts)
            )
        else:
            # Standard Models: Bake gradients into geometry
            if grad_r is not None:
                # Use the helper function on the ORIGINAL data first
                # (Augmenting gradients on synthetic minima rings is overkill/messy)
                r_grad_aug, p_grad_aug, z_grad_aug = _augment_with_gradients(
                    rmsd_r, rmsd_p, z_data, grad_r, grad_p
                )

                # Combine: Minima Rings + Gradient Helpers
                # Note: r_aug contains [original + rings]. r_grad_aug contains [original + gradient_helpers].
                # We need [original + rings + gradient_helpers].

                # Extract just the rings (tail of aug)
                ring_start_idx = len(rmsd_r)
                r_rings = r_aug[ring_start_idx:]
                p_rings = p_aug[ring_start_idx:]
                z_rings = z_aug[ring_start_idx:]

                final_r = np.concatenate([r_grad_aug, r_rings])
                final_p = np.concatenate([p_grad_aug, p_rings])
                final_vals = np.concatenate([z_grad_aug, z_rings])
                final_pts = np.column_stack([final_r, final_p])
                final_grads = None
            else:
                # No gradients available
                final_pts = np.column_stack([r_aug, p_aug])
                final_vals = z_aug
                final_grads = None

        # --- 4. Final Fit & Prediction ---
        log.info(f"Fitting final surface on {len(final_vals)} points...")

        # # Jitter final points
        # final_pts += rng.normal(0, 1e-6, size=final_pts.shape)

        # Final instantiation kwargs
        fit_kwargs = {
            "smoothing": best_noise,
            "length_scale": np.min([best_ls, 1.0]),
            "optimize": False,
        }

        if is_gradient_model:
            rbf = ModelClass(
                x=final_pts, y=final_vals, gradients=final_grads, **fit_kwargs
            )
        else:
            rbf = ModelClass(x_obs=final_pts, y_obs=final_vals, **fit_kwargs)

        if project_path:
            grid_r_eval = r_start + xg.ravel() * u_r + yg.ravel() * v_r
            grid_p_eval = p_start + xg.ravel() * u_p + yg.ravel() * v_p
            grid_pts_eval = np.column_stack([grid_r_eval, grid_p_eval])
        else:
            grid_pts_eval = np.column_stack([xg.ravel(), yg.ravel()])

        zg_flat = rbf(grid_pts_eval)
        zg_jax = jnp.asarray(zg_flat).reshape(xg.shape)

        if variance_threshold is not None and hasattr(rbf, "predict_var"):
            var_flat = rbf.predict_var(grid_pts_eval)
            var_jax = jnp.asarray(var_flat).reshape(xg.shape)
            effective_threshold = best_noise + variance_threshold
            zg_jax = zg_jax.at[var_jax > effective_threshold].set(jnp.nan)

        zg = np.asarray(zg_jax)

    # --- Plotting ---
    try:
        colormap = plt.get_cmap(cmap)
    except ValueError:
        colormap = plt.get_cmap("viridis")

    # High-res contourf
    ax.contourf(xg, yg, zg, levels=20, cmap=colormap, alpha=0.75, zorder=10)
    # ax.contour(
    #     xg, yg, zg, levels=15, colors="white", alpha=0.3, linewidths=0.5, zorder=11
    # )

    if show_pts:
        # TODO(rg): will a user every want to control this?
        # Filter: Only show points that are NOT augmented (step != -1)
        if step_data is not None:
            mask = step_data != -1
            px, py = plot_x_data[mask], plot_y_data[mask]
        else:
            px, py = plot_x_data, plot_y_data

        ax.scatter(px, py, c="k", s=12, marker=".", alpha=0.6, zorder=40)


def plot_landscape_path_overlay(ax, r, p, z, cmap, z_label, project_path=True):
    """Overlays the colored path line on the landscape, mapped to the chosen coordinate basis."""
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
