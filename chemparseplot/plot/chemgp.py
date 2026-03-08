"""ChemGP visualization functions.

Mixed plotnine (for 1D line charts) and matplotlib (for 2D
contour surfaces) plotting functions for GP-based optimization
convergence, surfaces, and diagnostics.

Surface plots use matplotlib contourf with the RUHI colormap,
matching the Julia CairoMakie originals. Line charts use plotnine
for ggplot2 grammar.

```{versionadded} 1.4.0
```
"""

import logging

import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_hline,
    geom_line,
    geom_point,
    geom_ribbon,
    ggplot,
    labs,
    scale_color_manual,
    scale_y_log10,
    theme,
    theme_minimal,
)

from chemparseplot.plot.theme import (
    RUHI_COLORS,
    setup_publication_theme,
    get_theme,
)

log = logging.getLogger(__name__)

# Ordered palette for method comparisons
_METHOD_PALETTE = [
    RUHI_COLORS["teal"],
    RUHI_COLORS["coral"],
    RUHI_COLORS["sky"],
    RUHI_COLORS["magenta"],
    RUHI_COLORS["sunshine"],
]

_JOST_THEME = theme_minimal() + theme(
    text=element_text(family="Jost"),
    plot_title=element_text(family="Jost", size=12),
    axis_title=element_text(family="Jost", size=11),
    strip_text=element_text(family="Jost", size=10),
)

_STROKE = [mpe.withStroke(linewidth=1.5, foreground="black")]


def _ensure_ruhi_cmap():
    """Ensure the ruhi_diverging colormap is registered."""
    try:
        plt.colormaps["ruhi_diverging"]
    except KeyError:
        from chemparseplot.plot.theme import build_cmap
        build_cmap(
            [
                RUHI_COLORS["teal"],
                RUHI_COLORS["sky"],
                RUHI_COLORS["magenta"],
                RUHI_COLORS["coral"],
                RUHI_COLORS["sunshine"],
            ],
            name="ruhi_diverging",
        )
    # Also register the reversed version
    try:
        plt.colormaps["ruhi_diverging_r"]
    except KeyError:
        cmap = plt.colormaps["ruhi_diverging"]
        plt.colormaps.register(cmap.reversed(), name="ruhi_diverging_r")


def _setup():
    """Common setup for matplotlib plots."""
    _ensure_ruhi_cmap()
    setup_publication_theme(get_theme("ruhi"))


# ---- Plotnine-based 1D line charts ----


def plot_convergence_curve(
    df: pd.DataFrame,
    x: str = "oracle_calls",
    y: str = "max_fatom",
    color: str = "method",
    log_y: bool = True,
    conv_tol: float | dict | None = None,
    width: float = 3.2,
    height: float = 2.5,
) -> ggplot:
    """Log-scale convergence: oracle calls vs force, colored by method.

    Parameters
    ----------
    conv_tol
        Single float draws one horizontal line. Dict mapping method name
        to threshold draws per-method dashed lines in matching colors.
    """
    methods = df[color].unique()
    palette = dict(zip(methods, _METHOD_PALETTE))

    p = (
        ggplot(df, aes(x=x, y=y, color=color))
        + geom_line(size=0.9)
        + geom_point(size=1.5, alpha=0.7)
        + scale_color_manual(values=palette)
        + labs(
            x="Oracle calls",
            y="Max per-atom force (eV/A)",
            color="Method",
        )
        + _JOST_THEME
        + theme(figure_size=(width, height))
    )
    if log_y:
        p = p + scale_y_log10()
    if conv_tol is not None:
        if isinstance(conv_tol, dict):
            for method_name, tol_val in conv_tol.items():
                method_color = palette.get(method_name, "grey")
                p = p + geom_hline(
                    yintercept=tol_val,
                    linetype="dashed",
                    color=method_color,
                    size=0.4,
                    alpha=0.6,
                )
        else:
            p = p + geom_hline(
                yintercept=conv_tol,
                linetype="dashed",
                color="grey",
                size=0.5,
            )
    return p


def plot_rff_quality(
    df: pd.DataFrame,
    exact_e_mae: float,
    exact_g_mae: float,
    width: float = 3.2,
    height: float = 2.5,
) -> ggplot:
    """Two-panel plot of energy and gradient MAE vs D_rff."""
    e_df = df[["d_rff", "energy_mae"]].rename(
        columns={"energy_mae": "mae"}
    ).assign(metric="Energy MAE")
    g_df = df[["d_rff", "gradient_mae"]].rename(
        columns={"gradient_mae": "mae"}
    ).assign(metric="Gradient MAE")
    long = pd.concat([e_df, g_df], ignore_index=True)

    baselines = pd.DataFrame({
        "metric": ["Energy MAE", "Gradient MAE"],
        "exact": [exact_e_mae, exact_g_mae],
    })

    p = (
        ggplot(long, aes(x="d_rff", y="mae"))
        + geom_line(color=RUHI_COLORS["teal"], size=0.9)
        + geom_point(color=RUHI_COLORS["teal"], size=1.5)
        + geom_hline(
            data=baselines,
            mapping=aes(yintercept="exact"),
            linetype="dashed",
            color=RUHI_COLORS["coral"],
            size=0.5,
        )
        + facet_wrap("metric", ncol=2, scales="free_y")
        + labs(x="D_rff", y="MAE")
        + _JOST_THEME
        + theme(figure_size=(width, height))
    )
    return p


def plot_hyperparameter_sensitivity(
    x_slice: np.ndarray,
    y_true: np.ndarray,
    panels: dict[str, dict],
    width: float = 10.5,
    height: float = 9.0,
) -> plt.Figure:
    """3x3 grid of 1D GP slices with confidence bands.

    Parameters
    ----------
    x_slice : array
        Shared x coordinates for all panels.
    y_true : array
        True surface values.
    panels : dict
        ``{"gp_ls1_sv1": {"E_pred": array, "E_std": array}, ...}``
        for each of the 9 panels.
    width, height : float
        Figure size in inches.
    """
    _setup()

    ls_labels = [r"$\ell = 0.05$", r"$\ell = 0.3$", r"$\ell = 2.0$"]
    sv_labels = [r"$\sigma_f = 0.1$", r"$\sigma_f = 1.0$", r"$\sigma_f = 100$"]

    fig, axes = plt.subplots(3, 3, figsize=(width, height), layout="constrained")

    for j in range(3):  # lengthscale (columns)
        for i in range(3):  # signal variance (rows)
            ax = axes[i][j]
            name = f"gp_ls{j+1}_sv{i+1}"
            if name not in panels:
                ax.set_visible(False)
                continue

            e_pred = np.asarray(panels[name]["E_pred"])
            e_std = np.asarray(panels[name]["E_std"])

            # Confidence band
            ax.fill_between(
                x_slice,
                e_pred - 2 * e_std,
                e_pred + 2 * e_std,
                color=RUHI_COLORS["sky"], alpha=0.3,
            )
            # True surface
            ax.plot(x_slice, y_true,
                    color="black", linewidth=1.0, linestyle="--")
            # GP mean
            ax.plot(x_slice, e_pred,
                    color=RUHI_COLORS["teal"], linewidth=1.5)

            ax.set_ylim(-250, 100)

            # Column title (top row only)
            if i == 0:
                ax.set_title(ls_labels[j], fontsize=11)
            # Row label (left column only)
            if j == 0:
                ax.set_ylabel(sv_labels[i], fontsize=11)
            else:
                ax.set_yticklabels([])
            # X label (bottom row only)
            if i == 2:
                ax.set_xlabel(r"$x$", fontsize=11)
            else:
                ax.set_xticklabels([])

    # Legend below the grid
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="black", linestyle="--", label="True surface"),
        Line2D([0], [0], color=RUHI_COLORS["teal"], linewidth=1.5, label="GP mean"),
        Patch(facecolor=RUHI_COLORS["sky"], alpha=0.3, label=r"$\pm 2\sigma$"),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        fontsize=10,
    )

    return fig


def plot_trust_region(
    x_slice: np.ndarray,
    e_true: np.ndarray,
    e_pred: np.ndarray,
    e_std: np.ndarray,
    in_trust: np.ndarray,
    train_x: np.ndarray | None = None,
    width: float = 7.0,
    height: float = 5.0,
) -> plt.Figure:
    """Trust region illustration with confidence band, boundary markers,
    and hypothetical bad step + oracle fallback.

    Parameters
    ----------
    x_slice : array
        X coordinates along the 1D slice.
    e_true : array
        True energy values.
    e_pred : array
        GP predicted energy values.
    e_std : array
        GP standard deviation.
    in_trust : array
        Boolean mask (1.0 for in trust, 0.0 for outside).
    train_x : array or None
        X coordinates of training points (projected to true surface).
    width, height : float
        Figure size in inches.
    """
    _setup()

    fig, ax = plt.subplots(figsize=(width, height))

    # Confidence band
    ax.fill_between(
        x_slice,
        e_pred - 2 * e_std,
        e_pred + 2 * e_std,
        color=RUHI_COLORS["sky"], alpha=0.25,
        label=r"$\pm 2\sigma$",
    )

    # True surface
    ax.plot(x_slice, e_true,
            color="black", linewidth=1.0, linestyle="--",
            label="True surface")

    # GP mean
    ax.plot(x_slice, e_pred,
            color=RUHI_COLORS["teal"], linewidth=1.5,
            label="GP mean")

    # Trust boundary vertical lines
    trust_bool = np.asarray(in_trust) > 0.5
    boundary_idx = np.where(np.diff(trust_bool.astype(int)) != 0)[0]
    for bi in boundary_idx:
        ax.axvline(x_slice[bi], color=RUHI_COLORS["magenta"],
                   linewidth=1.0, linestyle=":")

    # Training points projected to true surface
    if train_x is not None and len(train_x) > 0:
        # Find closest slice index for each training x
        train_e = []
        for tx in train_x:
            idx = np.argmin(np.abs(x_slice - tx))
            train_e.append(e_true[idx])
        ax.scatter(train_x, train_e, c="black", s=36, zorder=30)

    # Hypothetical bad step outside trust region
    x_bad = 1.0
    idx_bad = np.argmin(np.abs(x_slice - x_bad))
    e_bad_pred = e_pred[idx_bad]
    e_bad_true = e_true[idx_bad]

    ax.scatter([x_bad], [e_bad_pred], marker="X", s=100,
               c=RUHI_COLORS["coral"], zorder=31)
    ax.scatter([x_bad], [e_bad_true], marker="*", s=100,
               c=RUHI_COLORS["teal"], zorder=31)

    ax.annotate("GP step", (x_bad, e_bad_pred),
                textcoords="offset points", xytext=(5, 8),
                fontsize=9, color=RUHI_COLORS["coral"])
    ax.annotate("Oracle fallback", (x_bad, e_bad_true),
                textcoords="offset points", xytext=(5, 8),
                fontsize=9, color=RUHI_COLORS["teal"])

    # Trust boundary label
    if len(boundary_idx) > 0:
        bx = x_slice[boundary_idx[-1]]
        ax.annotate("trust\nboundary", (bx, -50),
                    textcoords="offset points", xytext=(5, 0),
                    fontsize=8, color=RUHI_COLORS["magenta"])

    ax.set_ylim(-250, 100)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$E$ (a.u.)")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    return fig


def plot_fps_projection(
    selected_pc1,
    selected_pc2,
    pruned_pc1,
    pruned_pc2,
    width: float = 3.2,
    height: float = 2.5,
) -> ggplot:
    """PCA scatter: selected (teal) vs pruned (grey) points."""
    sel_df = pd.DataFrame({
        "pc1": np.asarray(selected_pc1),
        "pc2": np.asarray(selected_pc2),
        "group": "Selected",
    })
    prn_df = pd.DataFrame({
        "pc1": np.asarray(pruned_pc1),
        "pc2": np.asarray(pruned_pc2),
        "group": "Pruned",
    })
    all_df = pd.concat([prn_df, sel_df], ignore_index=True)

    palette = {
        "Selected": RUHI_COLORS["teal"],
        "Pruned": "#AAAAAA",
    }

    p = (
        ggplot(all_df, aes(x="pc1", y="pc2", color="group"))
        + geom_point(size=1.5, alpha=0.7)
        + scale_color_manual(values=palette)
        + labs(x="PC1", y="PC2", color="")
        + _JOST_THEME
        + theme(figure_size=(width, height))
    )
    return p


def plot_energy_profile(
    df: pd.DataFrame,
    x: str = "image",
    y: str = "energy",
    color: str = "method",
    width: float = 3.2,
    height: float = 2.5,
) -> ggplot:
    """NEB energy profile across images."""
    methods = df[color].unique()
    palette = dict(zip(methods, _METHOD_PALETTE))

    p = (
        ggplot(df, aes(x=x, y=y, color=color))
        + geom_line(size=0.9)
        + geom_point(size=2.0)
        + scale_color_manual(values=palette)
        + labs(
            x="Image index",
            y=r"$\Delta E$ (eV)",
            color="Method",
        )
        + _JOST_THEME
        + theme(figure_size=(width, height))
    )
    return p


# ---- Matplotlib-based 2D surface plots ----


def plot_surface_contour(
    grid_x,
    grid_y,
    grid_z,
    paths: dict[str, tuple] | None = None,
    points: dict[str, tuple] | None = None,
    clamp_lo: float | None = None,
    clamp_hi: float | None = None,
    levels: int | np.ndarray | None = None,
    contour_step: float | None = None,
    point_style: str = "star",
    width: float = 7.0,
    height: float = 5.0,
) -> plt.Figure:
    """2D filled contour with optional path and point overlays.

    Matches the Julia CairoMakie style: RUHI colormap, black contour
    lines, white NEB path, coral image circles, numbered images,
    sunshine star endpoints (or circle+label for minima/saddles).

    Parameters
    ----------
    grid_x, grid_y, grid_z : array-like
        2D meshgrid arrays for the surface.
    paths : dict or None
        ``{label: (xs, ys)}`` paths to overlay as lines.
    points : dict or None
        ``{label: (xs, ys)}`` scatter points to overlay.
        For "minima"/"saddles" keys, uses circle/x markers with labels.
        For "endpoints", uses star markers.
    clamp_lo, clamp_hi : float or None
        Value clamping for z data.
    levels : int or array or None
        Explicit contour levels. If None, uses 25 levels over clamped range.
    contour_step : float or None
        Step size for black contour lines. If None, auto-computed.
    point_style : str
        Default point style: "star" or "labeled".
    width, height : float
        Figure size in inches.
    """
    _setup()

    gz = np.asarray(grid_z).copy()
    if clamp_lo is not None:
        gz = np.clip(gz, clamp_lo, None)
    if clamp_hi is not None:
        gz = np.clip(gz, None, clamp_hi)

    lo = clamp_lo if clamp_lo is not None else float(gz.min())
    hi = clamp_hi if clamp_hi is not None else float(gz.max())

    if levels is None:
        levels = np.linspace(lo, hi, 25)

    if contour_step is None:
        contour_step = (hi - lo) / 10.0
    contour_levels = np.arange(lo, hi + contour_step, contour_step)

    fig, ax = plt.subplots(figsize=(width, height))
    gx = np.asarray(grid_x)
    gy = np.asarray(grid_y)

    cf = ax.contourf(gx, gy, gz, levels=levels, cmap="ruhi_diverging")
    ax.contour(gx, gy, gz, levels=contour_levels,
               colors="black", linewidths=0.3)
    fig.colorbar(cf, ax=ax, label=r"$E$ (eV)", shrink=0.8)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    # NEB paths: white line + coral circles + numbered images
    if paths is not None:
        for label, (xs, ys) in paths.items():
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            ax.plot(xs, ys, color="white", linewidth=2.0, zorder=30)
            ax.scatter(xs, ys, c=RUHI_COLORS["coral"], s=50,
                       marker="o", edgecolors="white", linewidths=1.0,
                       zorder=31)
            for j in range(len(xs)):
                ax.annotate(
                    str(j + 1), (xs[j], ys[j]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=8, color="white", fontweight="bold",
                    zorder=32,
                )

    # Points: context-dependent markers
    if points is not None:
        _min_labels = iter(["A", "B", "C", "D", "E"])
        _sad_labels = iter(["S1", "S2", "S3", "S4"])

        for pname, (xs, ys) in points.items():
            xs = np.asarray(xs)
            ys = np.asarray(ys)

            if "minim" in pname.lower():
                # White circles with black stroke + letter labels
                ax.scatter(xs, ys, c="white", s=80, marker="o",
                           edgecolors="black", linewidths=1.5, zorder=33)
                for k in range(len(xs)):
                    lbl = next(_min_labels, f"M{k}")
                    ax.annotate(
                        lbl, (xs[k], ys[k]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=12, fontweight="bold", color="white",
                        path_effects=_STROKE, zorder=34,
                    )
            elif "saddle" in pname.lower():
                # White x markers with black stroke + S labels
                ax.scatter(xs, ys, c="white", s=100, marker="X",
                           edgecolors="black", linewidths=1.5, zorder=33)
                for k in range(len(xs)):
                    lbl = next(_sad_labels, f"S{k}")
                    ax.annotate(
                        lbl, (xs[k], ys[k]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=12, fontweight="bold", color="white",
                        path_effects=_STROKE, zorder=34,
                    )
            else:
                # Endpoints or generic: sunshine stars
                ax.scatter(xs, ys, c=RUHI_COLORS["sunshine"], s=120,
                           marker="*", edgecolors="white", linewidths=1.0,
                           zorder=33)

    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_gp_progression(
    grids: dict[int, dict],
    true_energy,
    x_range,
    y_range,
    clamp_lo: float = -200.0,
    clamp_hi: float = 50.0,
    n_cols: int = 2,
    width: float = 10.0,
    height: float = 8.0,
) -> plt.Figure:
    """Faceted contour panels showing GP mean at different training sizes.

    Matches Julia: clamped energy, RUHI colormap, black contour lines,
    training points as black dots.

    Parameters
    ----------
    grids : dict
        ``{n_train: {"gp_mean": 2d_array, "train_x": array, "train_y": array}}``.
    true_energy : array-like
        2D array of the true energy surface.
    x_range, y_range : array-like
        1D coordinate arrays for the grid.
    clamp_lo, clamp_hi : float
        Energy clamping range.
    n_cols : int
        Number of columns in the panel layout.
    width, height : float
        Figure size in inches.
    """
    _setup()

    sorted_grids = sorted(grids.items())
    n_panels = len(sorted_grids)
    n_rows = (n_panels + n_cols - 1) // n_cols

    xv, yv = np.meshgrid(x_range, y_range)
    levels = np.linspace(clamp_lo, clamp_hi, 25)
    contour_levels = np.arange(clamp_lo, clamp_hi + 25, 25)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(width, height),
        squeeze=False, layout="constrained",
    )

    cf = None
    for idx, (n_train, data) in enumerate(sorted_grids):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        gp_mean = np.clip(np.asarray(data["gp_mean"]), clamp_lo, clamp_hi)

        cf = ax.contourf(xv, yv, gp_mean, levels=levels,
                         cmap="ruhi_diverging")
        ax.contour(xv, yv, gp_mean, levels=contour_levels,
                   colors="black", linewidths=0.3)

        # Training points
        if "train_x" in data and "train_y" in data:
            ax.scatter(data["train_x"], data["train_y"],
                       c="black", s=15, marker="o",
                       edgecolors="white", linewidths=0.5, zorder=30)

        ax.set_title(f"$N = {n_train}$", fontsize=11)
        ax.set_aspect("equal")
        if row == n_rows - 1:
            ax.set_xlabel(r"$x$")
        if col == 0:
            ax.set_ylabel(r"$y$")

    # Hide unused axes
    for idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    if cf is not None:
        fig.colorbar(cf, ax=axes.ravel().tolist(),
                     label=r"$E$ (a.u.)", shrink=0.8)

    return fig


def plot_nll_landscape(
    grid_x,
    grid_y,
    grid_nll,
    grid_grad_norm=None,
    optimum: tuple[float, float] | None = None,
    width: float = 7.0,
    height: float = 5.0,
) -> plt.Figure:
    """NLL contour in (log sigma^2, log theta) space.

    Matches Julia: reversed RUHI colormap (low NLL = warm),
    quantile-clipped, gradient norm dashed overlay.

    Parameters
    ----------
    grid_x, grid_y : array-like
        2D meshgrid of log sigma^2 and log theta values.
    grid_nll : array-like
        2D array of NLL values.
    grid_grad_norm : array-like or None
        2D array of gradient norm values (dashed contour overlay).
    optimum : tuple or None
        (log_sigma2, log_theta) of the MAP optimum to mark.
    width, height : float
        Figure size in inches.
    """
    _setup()

    fig, ax = plt.subplots(figsize=(width, height))
    gx = np.asarray(grid_x)
    gy = np.asarray(grid_y)
    gz = np.asarray(grid_nll)

    # Log-scale shifted NLL to reveal basin structure.
    # NLL ranges from ~-300 to ~40000; linear colormap hides the basin.
    finite = gz[np.isfinite(gz)]
    nll_min = float(np.min(finite))
    # Shift so minimum = 1, then log10
    gz_shifted = np.where(np.isfinite(gz), gz - nll_min + 1.0, np.nan)
    gz_log = np.log10(gz_shifted)

    # Clip top 2% of log values for clean colorbar
    log_finite = gz_log[np.isfinite(gz_log)]
    hi = float(np.quantile(log_finite, 0.98))
    gz_clipped = np.clip(gz_log, 0, hi)

    # Reversed RUHI colormap (Julia: Reverse(ENERGY_COLORMAP))
    cf = ax.contourf(gx, gy, gz_clipped, levels=20,
                     cmap="ruhi_diverging_r")
    fig.colorbar(
        cf, ax=ax,
        label=r"$\log_{10}(\mathcal{L}_{\mathrm{MAP}} - \mathcal{L}_{\min} + 1)$",
        shrink=0.8,
    )

    # Gradient norm overlay (dashed black contours)
    if grid_grad_norm is not None:
        gg = np.asarray(grid_grad_norm)
        g_finite = gg[np.isfinite(gg)]
        if len(g_finite) > 0:
            g_lo = np.quantile(g_finite, 0.05)
            g_hi = np.quantile(g_finite, 0.80)
            gg_clipped = np.clip(gg, g_lo, g_hi)
            ax.contour(gx, gy, gg_clipped, levels=8,
                       colors="black", linewidths=0.5, linestyles="--")

    ax.set_xlabel(r"$\ln\,\sigma^2$")
    ax.set_ylabel(r"$\ln\,\theta$")

    if optimum is not None:
        ax.plot(
            optimum[0], optimum[1],
            marker="*", color=RUHI_COLORS["coral"],
            markersize=15, markeredgecolor="white",
            markeredgewidth=0.8, zorder=30,
        )

    fig.tight_layout()
    return fig


def plot_variance_overlay(
    grid_x,
    grid_y,
    grid_energy,
    grid_variance,
    train_points: tuple | None = None,
    stationary: dict | None = None,
    clamp_lo: float = -200.0,
    clamp_hi: float = 50.0,
    width: float = 7.0,
    height: float = 5.0,
) -> plt.Figure:
    """Energy surface with hatched high-variance regions.

    Matches Julia: clamped energy contourf as base, diagonal hatching
    for high-variance regions, magenta boundary contour, labeled
    stationary points.

    Parameters
    ----------
    grid_x, grid_y : array-like
        2D meshgrid arrays.
    grid_energy : array-like
        2D array of energy values (shown as contourf).
    grid_variance : array-like
        2D array of variance values (hatching + boundary).
    train_points : tuple or None
        ``(xs, ys)`` of training data locations.
    stationary : dict or None
        ``{"min0": (x,y), "saddle0": (x,y), ...}`` labeled
        stationary points.
    clamp_lo, clamp_hi : float
        Energy clamping range for display.
    width, height : float
        Figure size in inches.
    """
    _setup()

    fig, ax = plt.subplots(figsize=(width, height))
    gx = np.asarray(grid_x)
    gy = np.asarray(grid_y)
    ge = np.clip(np.asarray(grid_energy), clamp_lo, clamp_hi)
    gv = np.asarray(grid_variance)

    levels = np.linspace(clamp_lo, clamp_hi, 25)
    contour_step = (clamp_hi - clamp_lo) / 10.0
    contour_levels = np.arange(clamp_lo, clamp_hi + contour_step, contour_step)

    # Energy surface as filled contour (shows basins clearly)
    cf = ax.contourf(gx, gy, ge, levels=levels, cmap="ruhi_diverging")
    ax.contour(gx, gy, ge, levels=contour_levels,
               colors="black", linewidths=0.3)
    fig.colorbar(cf, ax=ax, label="Energy", shrink=0.8)

    # Hatching for high-variance regions (75th percentile of positive values)
    positive_var = gv[gv > 0]
    if len(positive_var) > 0:
        high_thresh = np.quantile(positive_var, 0.75)

        # Use contourf with hatching for the high-variance region
        ax.contourf(
            gx, gy, gv,
            levels=[high_thresh, gv.max() * 1.1],
            colors="none",
            hatches=["//"],
            alpha=0.0,
        )

        # Magenta boundary contour at the threshold
        ax.contour(
            gx, gy, gv,
            levels=[high_thresh],
            colors=[RUHI_COLORS["magenta"]],
            linewidths=1.2,
        )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    # Training points
    if train_points is not None:
        xs, ys = train_points
        ax.scatter(xs, ys, c="black", s=20, marker="o",
                   edgecolors="white", linewidths=0.5, zorder=30)

    # Stationary points with labels matching Julia (A/B/C for minima, S1/S2 for saddles)
    if stationary is not None:
        min_labels = iter(["A", "B", "C", "D"])
        sad_labels = iter(["S1", "S2", "S3"])

        for pname, (sx, sy) in stationary.items():
            if "min" in pname:
                lbl = next(min_labels, pname)
                ax.plot(sx, sy, marker="o", color="white",
                        markersize=10, markeredgecolor="black",
                        markeredgewidth=1.5, zorder=31)
            else:
                lbl = next(sad_labels, pname)
                ax.plot(sx, sy, marker="x", color="white",
                        markersize=12, markeredgecolor="black",
                        markeredgewidth=1.5, zorder=31)
            ax.annotate(
                lbl, (sx, sy),
                textcoords="offset points", xytext=(6, 6),
                fontsize=12, fontweight="bold", color="white",
                path_effects=_STROKE, zorder=32,
            )

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markeredgecolor="white", markersize=5, label="Training"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=8, label="Minima"),
        Line2D([0], [0], marker="x", color="white",
               markeredgecolor="black", markersize=10,
               markeredgewidth=1.5, label="Saddles"),
        Line2D([0], [0], color=RUHI_COLORS["magenta"], linewidth=1.2,
               label=r"High $\sigma^2$ boundary"),
    ]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=9)

    fig.tight_layout()
    return fig
