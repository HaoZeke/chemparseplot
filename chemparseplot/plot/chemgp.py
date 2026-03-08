"""ChemGP visualization functions.

Mixed plotnine (for 1D line charts) and matplotlib (for 2D
contour surfaces) plotting functions for GP-based optimization
convergence, surfaces, and diagnostics.

Surface plots use matplotlib contourf with the RUHI colormap,
matching the plt-neb landscape style. Line charts use plotnine
for ggplot2 grammar.

```{versionadded} 1.4.0
```
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_grid,
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


def _ensure_ruhi_cmap():
    """Ensure the ruhi_diverging colormap is registered."""
    # theme.py registers it on import, but be safe
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


# ---- Plotnine-based 1D line charts ----


def plot_convergence_curve(
    df: pd.DataFrame,
    x: str = "oracle_calls",
    y: str = "max_fatom",
    color: str = "method",
    log_y: bool = True,
    conv_tol: float | None = None,
    width: float = 3.2,
    height: float = 2.5,
) -> ggplot:
    """Log-scale convergence: oracle calls vs force, colored by method.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns named by *x*, *y*, and *color*.
    x : str
        Column for the horizontal axis (default: oracle_calls).
    y : str
        Column for the vertical axis (default: max_fatom).
    color : str
        Column for method/series color grouping.
    log_y : bool
        If True, use log10 scale on the y axis.
    conv_tol : float or None
        If given, draw a horizontal dashed line at this value.
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.

    Returns
    -------
    ggplot
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
    """Two-panel plot of energy and gradient MAE vs D_rff.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: d_rff, energy_mae, gradient_mae.
    exact_e_mae : float
        Exact GP energy MAE (shown as horizontal baseline).
    exact_g_mae : float
        Exact GP gradient MAE (shown as horizontal baseline).
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.

    Returns
    -------
    ggplot
    """
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
    df: pd.DataFrame,
    width: float = 6.4,
    height: float = 5.0,
) -> ggplot:
    """3x3 facet grid of 1D GP slices for different ell/sigma_f combos.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: x, y_true, y_pred, y_lower, y_upper,
        ell, sigma_f.
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.

    Returns
    -------
    ggplot
    """
    p = (
        ggplot(df, aes(x="x"))
        + geom_line(
            aes(y="y_true"),
            color="grey",
            linetype="dashed",
            size=0.5,
        )
        + geom_line(
            aes(y="y_pred"),
            color=RUHI_COLORS["teal"],
            size=0.8,
        )
        + facet_grid("ell ~ sigma_f", labeller="label_both")
        + labs(x="x", y="y")
        + _JOST_THEME
        + theme(figure_size=(width, height))
    )
    return p


def plot_trust_region(
    df: pd.DataFrame,
    train_points: tuple | None = None,
    width: float = 3.2,
    height: float = 2.5,
) -> ggplot:
    """1D trust region illustration with confidence band.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: x, y_pred, y_lower, y_upper.
    train_points : tuple or None
        ``(xs, ys)`` of training observations to overlay.
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.

    Returns
    -------
    ggplot
    """
    p = (
        ggplot(df, aes(x="x"))
        + geom_ribbon(
            aes(ymin="y_lower", ymax="y_upper"),
            fill=RUHI_COLORS["sky"],
            alpha=0.25,
        )
        + geom_line(
            aes(y="y_pred"),
            color=RUHI_COLORS["teal"],
            size=0.9,
        )
        + labs(x="x", y="Predicted value")
        + _JOST_THEME
        + theme(figure_size=(width, height))
    )

    if train_points is not None:
        xs, ys = train_points
        tp_df = pd.DataFrame({
            "x": np.asarray(xs),
            "y": np.asarray(ys),
        })
        p = p + geom_point(
            data=tp_df,
            mapping=aes(x="x", y="y"),
            color=RUHI_COLORS["coral"],
            size=2.5,
        )
    return p


def plot_fps_projection(
    selected_pc1,
    selected_pc2,
    pruned_pc1,
    pruned_pc2,
    width: float = 3.2,
    height: float = 2.5,
) -> ggplot:
    """PCA scatter: selected (teal) vs pruned (grey) points.

    Parameters
    ----------
    selected_pc1, selected_pc2 : array-like
        PC1 and PC2 coordinates of FPS-selected points.
    pruned_pc1, pruned_pc2 : array-like
        PC1 and PC2 coordinates of pruned points.
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.

    Returns
    -------
    ggplot
    """
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
    """NEB energy profile across images.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns named by *x*, *y*, and *color*.
    x : str
        Column for the image index or reaction coordinate.
    y : str
        Column for the energy value.
    color : str
        Column for method/series grouping.
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.

    Returns
    -------
    ggplot
    """
    methods = df[color].unique()
    palette = dict(zip(methods, _METHOD_PALETTE))

    p = (
        ggplot(df, aes(x=x, y=y, color=color))
        + geom_line(size=0.9)
        + geom_point(size=2.0)
        + scale_color_manual(values=palette)
        + labs(
            x="Image index",
            y="Energy (eV)",
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
    width: float = 7.0,
    height: float = 5.0,
) -> plt.Figure:
    """2D filled contour with optional path and point overlays.

    Uses matplotlib contourf with the RUHI colormap, matching
    the plt-neb landscape style.

    Parameters
    ----------
    grid_x, grid_y, grid_z : array-like
        2D meshgrid arrays for the surface.
    paths : dict or None
        ``{label: (xs, ys)}`` paths to overlay as lines.
    points : dict or None
        ``{label: (xs, ys)}`` scatter points to overlay.
    clamp_lo, clamp_hi : float or None
        Optional value clamping for the z data.
    width, height : float
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_ruhi_cmap()
    setup_publication_theme(get_theme("ruhi"))

    gz = np.asarray(grid_z).copy()
    if clamp_lo is not None:
        gz = np.clip(gz, clamp_lo, None)
    if clamp_hi is not None:
        gz = np.clip(gz, None, clamp_hi)

    fig, ax = plt.subplots(figsize=(width, height))
    cf = ax.contourf(
        np.asarray(grid_x),
        np.asarray(grid_y),
        gz,
        levels=20,
        cmap="ruhi_diverging",
        alpha=0.85,
    )
    ax.contour(
        np.asarray(grid_x),
        np.asarray(grid_y),
        gz,
        levels=20,
        colors="white",
        linewidths=0.3,
        alpha=0.5,
    )
    fig.colorbar(cf, ax=ax, label="Energy", shrink=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if paths is not None:
        for i, (label, (xs, ys)) in enumerate(paths.items()):
            col = _METHOD_PALETTE[i % len(_METHOD_PALETTE)]
            ax.plot(
                xs, ys,
                color=col, linewidth=1.5, zorder=30,
                label=label,
            )
            # Start/end markers
            ax.plot(
                xs[0], ys[0],
                marker="o", color=col, markersize=8,
                markeredgecolor="white", markeredgewidth=0.8,
                zorder=31,
            )
            ax.plot(
                xs[-1], ys[-1],
                marker="*", color=col, markersize=12,
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=31,
            )

    if points is not None:
        for i, (label, (xs, ys)) in enumerate(points.items()):
            col = _METHOD_PALETTE[i % len(_METHOD_PALETTE)]
            ax.scatter(
                xs, ys,
                c=col, s=40, marker="D",
                edgecolors="white", linewidths=0.5,
                zorder=32, label=label,
            )

    if (paths and len(paths) > 1) or (points and len(points) > 1):
        ax.legend(fontsize=9, loc="best")

    fig.tight_layout()
    return fig


def plot_gp_progression(
    grids: dict[int, dict],
    true_energy,
    x_range,
    y_range,
    n_cols: int = 2,
    width: float = 10.0,
    height: float = 8.0,
) -> plt.Figure:
    """Faceted contour panels showing GP mean at different training sizes.

    Parameters
    ----------
    grids : dict
        ``{n_train: {"gp_mean": 2d_array, ...}}``.
    true_energy : array-like
        2D array of the true energy surface.
    x_range, y_range : array-like
        1D coordinate arrays for the grid.
    n_cols : int
        Number of columns in the panel layout.
    width, height : float
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_ruhi_cmap()
    setup_publication_theme(get_theme("ruhi"))

    sorted_grids = sorted(grids.items())
    n_panels = len(sorted_grids)
    n_rows = (n_panels + n_cols - 1) // n_cols

    xv, yv = np.meshgrid(x_range, y_range)

    # Shared color range across all panels
    vmin = min(np.asarray(d["gp_mean"]).min() for _, d in sorted_grids)
    vmax = max(np.asarray(d["gp_mean"]).max() for _, d in sorted_grids)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(width, height),
        squeeze=False, layout="constrained",
    )

    for idx, (n_train, data) in enumerate(sorted_grids):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        gp_mean = np.asarray(data["gp_mean"])

        cf = ax.contourf(
            xv, yv, gp_mean,
            levels=20,
            cmap="ruhi_diverging",
            alpha=0.85,
            vmin=vmin, vmax=vmax,
        )
        ax.contour(
            xv, yv, gp_mean,
            levels=20,
            colors="white",
            linewidths=0.3,
            alpha=0.5,
        )
        ax.set_title(f"N = {n_train}", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # Hide unused axes
    for idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.colorbar(cf, ax=axes.ravel().tolist(), label="GP mean", shrink=0.8)
    return fig


def plot_nll_landscape(
    grid_x,
    grid_y,
    grid_nll,
    optimum: tuple[float, float] | None = None,
    width: float = 7.0,
    height: float = 5.0,
) -> plt.Figure:
    """NLL contour in (log sigma^2, log theta) space.

    Parameters
    ----------
    grid_x, grid_y : array-like
        2D meshgrid of log sigma^2 and log theta values.
    grid_nll : array-like
        2D array of NLL values.
    optimum : tuple or None
        (log_sigma2, log_theta) of the MAP optimum to mark.
    width, height : float
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_ruhi_cmap()
    setup_publication_theme(get_theme("ruhi"))

    fig, ax = plt.subplots(figsize=(width, height))
    gx = np.asarray(grid_x)
    gy = np.asarray(grid_y)
    gz = np.asarray(grid_nll)

    cf = ax.contourf(
        gx, gy, gz,
        levels=20,
        cmap="ruhi_diverging",
        alpha=0.85,
    )
    ax.contour(
        gx, gy, gz,
        levels=20,
        colors="white",
        linewidths=0.3,
        alpha=0.5,
    )
    fig.colorbar(cf, ax=ax, label="NLL", shrink=0.8)
    ax.set_xlabel(r"$\log\,\sigma^2$")
    ax.set_ylabel(r"$\log\,\theta$")

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
    width: float = 7.0,
    height: float = 5.0,
) -> plt.Figure:
    """Variance heatmap overlaid on energy surface.

    Parameters
    ----------
    grid_x, grid_y : array-like
        2D meshgrid arrays.
    grid_energy : array-like
        2D array of energy values (shown as contour lines).
    grid_variance : array-like
        2D array of variance values (shown as fill).
    train_points : tuple or None
        ``(xs, ys)`` of training data locations.
    stationary : dict or None
        ``{"min": (x,y), "saddle": (x,y), ...}`` labeled
        stationary points.
    width, height : float
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    setup_publication_theme(get_theme("ruhi"))

    fig, ax = plt.subplots(figsize=(width, height))
    gx = np.asarray(grid_x)
    gy = np.asarray(grid_y)

    # Variance as filled contour
    cf = ax.contourf(
        gx, gy,
        np.asarray(grid_variance),
        levels=20,
        cmap="YlOrRd",
        alpha=0.8,
    )
    fig.colorbar(cf, ax=ax, label="Variance", shrink=0.8)

    # Energy as contour lines on top
    ax.contour(
        gx, gy,
        np.asarray(grid_energy),
        levels=15,
        colors=RUHI_COLORS["teal"],
        linewidths=0.6,
        alpha=0.7,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if train_points is not None:
        xs, ys = train_points
        ax.scatter(
            xs, ys,
            c=RUHI_COLORS["teal"], s=30,
            marker="x", linewidths=1.5,
            zorder=30, label="Training",
        )

    if stationary is not None:
        for label, (sx, sy) in stationary.items():
            marker = "v" if "min" in label else "^"
            ax.plot(
                sx, sy,
                marker=marker, color=RUHI_COLORS["coral"],
                markersize=10, markeredgecolor="white",
                markeredgewidth=0.8, zorder=31,
            )
            ax.annotate(
                label, (sx, sy),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color=RUHI_COLORS["teal"],
            )

    fig.tight_layout()
    return fig
