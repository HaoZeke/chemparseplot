"""ChemGP visualization functions using plotnine + polars.

Plotnine (ggplot2 grammar) plotting functions for GP-based
optimization convergence, surfaces, and diagnostics. All functions
accept polars DataFrames and return ggplot objects.

```{versionadded} 1.4.0
```
"""

import logging

import numpy as np
import polars as pl
from plotnine import (
    aes,
    annotate,
    element_text,
    facet_grid,
    facet_wrap,
    geom_contour,
    geom_hline,
    geom_line,
    geom_path,
    geom_point,
    geom_tile,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_gradient2,
    scale_fill_gradientn,
    scale_x_continuous,
    scale_y_continuous,
    scale_y_log10,
    theme,
    theme_minimal,
)

from chemparseplot.plot.theme import RUHI_COLORS

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


def _grid_to_long(
    grid_x,
    grid_y,
    grid_z,
    value_name: str = "value",
) -> pl.DataFrame:
    """Convert 2D meshgrid arrays to a long-form polars DataFrame.

    Parameters
    ----------
    grid_x : array-like
        2D array of x coordinates (from meshgrid).
    grid_y : array-like
        2D array of y coordinates (from meshgrid).
    grid_z : array-like
        2D array of values.
    value_name : str
        Column name for the z values.

    Returns
    -------
    pl.DataFrame
        Columns: x, y, <value_name>.
    """
    gx = np.asarray(grid_x).ravel()
    gy = np.asarray(grid_y).ravel()
    gz = np.asarray(grid_z).ravel()
    return pl.DataFrame({
        "x": gx,
        "y": gy,
        value_name: gz,
    })


def plot_convergence_curve(
    df: pl.DataFrame,
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
    df : pl.DataFrame
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
    pdf = df.to_pandas()
    methods = pdf[color].unique()
    palette = dict(zip(methods, _METHOD_PALETTE))

    p = (
        ggplot(pdf, aes(x=x, y=y, color=color))
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


def plot_surface_contour(
    grid_x,
    grid_y,
    grid_z,
    paths: dict[str, tuple] | None = None,
    points: dict[str, tuple] | None = None,
    width: float = 3.2,
    height: float = 2.5,
) -> ggplot:
    """2D filled contour with optional path and point overlays.

    Parameters
    ----------
    grid_x, grid_y, grid_z : array-like
        2D meshgrid arrays for the surface.
    paths : dict or None
        ``{label: (xs, ys)}`` paths to overlay as lines.
    points : dict or None
        ``{label: (xs, ys)}`` scatter points to overlay.
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.

    Returns
    -------
    ggplot
    """
    long_df = _grid_to_long(grid_x, grid_y, grid_z, "energy")
    pdf = long_df.to_pandas()

    p = (
        ggplot(pdf, aes(x="x", y="y"))
        + geom_tile(aes(fill="energy"))
        + geom_contour(
            aes(z="energy"),
            color="white",
            alpha=0.4,
            size=0.3,
        )
        + scale_fill_gradientn(
            colors=[
                RUHI_COLORS["teal"],
                RUHI_COLORS["sky"],
                RUHI_COLORS["sunshine"],
                RUHI_COLORS["coral"],
            ],
        )
        + labs(x="x", y="y", fill="Energy")
        + _JOST_THEME
        + theme(figure_size=(width, height))
    )

    if paths is not None:
        for i, (label, (xs, ys)) in enumerate(
            paths.items()
        ):
            path_df = pl.DataFrame({
                "x": np.asarray(xs),
                "y": np.asarray(ys),
            }).to_pandas()
            col = _METHOD_PALETTE[i % len(_METHOD_PALETTE)]
            p = p + geom_path(
                data=path_df,
                mapping=aes(x="x", y="y"),
                color=col,
                size=0.8,
            )

    if points is not None:
        for i, (label, (xs, ys)) in enumerate(
            points.items()
        ):
            pt_df = pl.DataFrame({
                "x": np.asarray(xs),
                "y": np.asarray(ys),
            }).to_pandas()
            col = _METHOD_PALETTE[i % len(_METHOD_PALETTE)]
            p = p + geom_point(
                data=pt_df,
                mapping=aes(x="x", y="y"),
                color=col,
                size=2.0,
            )

    return p


def plot_gp_progression(
    grids: dict[int, dict],
    true_energy,
    x_range,
    y_range,
    n_cols: int = 2,
    width: float = 6.4,
    height: float = 5.0,
) -> ggplot:
    """Faceted contour panels showing GP mean at different training sizes.

    Parameters
    ----------
    grids : dict
        ``{n_train: {"gp_mean": 2d_array, "train_x": 1d, ...}}``.
        Each entry produces one facet panel.
    true_energy : array-like
        2D array of the true energy surface (same shape as
        gp_mean arrays).
    x_range : array-like
        1D array of x coordinates for the grid.
    y_range : array-like
        1D array of y coordinates for the grid.
    n_cols : int
        Number of columns in the facet layout.
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.

    Returns
    -------
    ggplot
    """
    xv, yv = np.meshgrid(x_range, y_range)
    frames = []
    for n_train, data in sorted(grids.items()):
        gp_mean = np.asarray(data["gp_mean"])
        panel_df = _grid_to_long(xv, yv, gp_mean, "energy")
        panel_df = panel_df.with_columns(
            pl.lit(f"N={n_train}").alias("panel"),
        )
        frames.append(panel_df)

    all_df = pl.concat(frames).to_pandas()

    p = (
        ggplot(all_df, aes(x="x", y="y"))
        + geom_tile(aes(fill="energy"))
        + geom_contour(
            aes(z="energy"),
            color="white",
            alpha=0.4,
            size=0.3,
        )
        + facet_wrap("panel", ncol=n_cols)
        + scale_fill_gradientn(
            colors=[
                RUHI_COLORS["teal"],
                RUHI_COLORS["sky"],
                RUHI_COLORS["sunshine"],
                RUHI_COLORS["coral"],
            ],
        )
        + labs(x="x", y="y", fill="GP mean")
        + _JOST_THEME
        + theme(figure_size=(width, height))
    )
    return p


def plot_rff_quality(
    df: pl.DataFrame,
    exact_e_mae: float,
    exact_g_mae: float,
    width: float = 3.2,
    height: float = 2.5,
) -> ggplot:
    """Two-panel plot of energy and gradient MAE vs D_rff.

    Parameters
    ----------
    df : pl.DataFrame
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
    e_df = df.select(
        pl.col("d_rff"),
        pl.col("energy_mae").alias("mae"),
    ).with_columns(
        pl.lit("Energy MAE").alias("metric"),
    )
    g_df = df.select(
        pl.col("d_rff"),
        pl.col("gradient_mae").alias("mae"),
    ).with_columns(
        pl.lit("Gradient MAE").alias("metric"),
    )
    long = pl.concat([e_df, g_df]).to_pandas()

    # Baselines for each facet
    baselines = pl.DataFrame({
        "metric": ["Energy MAE", "Gradient MAE"],
        "exact": [exact_e_mae, exact_g_mae],
    }).to_pandas()

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


def plot_nll_landscape(
    grid_x,
    grid_y,
    grid_nll,
    optimum: tuple[float, float] | None = None,
    width: float = 3.2,
    height: float = 2.5,
) -> ggplot:
    """NLL contour in (log sigma^2, log theta) space.

    Parameters
    ----------
    grid_x, grid_y : array-like
        2D meshgrid of log sigma^2 and log theta values.
    grid_nll : array-like
        2D array of NLL values.
    optimum : tuple or None
        (log_sigma2, log_theta) of the MAP optimum to mark.
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.

    Returns
    -------
    ggplot
    """
    long_df = _grid_to_long(
        grid_x, grid_y, grid_nll, "nll"
    ).to_pandas()

    p = (
        ggplot(long_df, aes(x="x", y="y"))
        + geom_tile(aes(fill="nll"))
        + geom_contour(
            aes(z="nll"),
            color="white",
            alpha=0.5,
            size=0.3,
            bins=20,
        )
        + scale_fill_gradientn(
            colors=[
                RUHI_COLORS["teal"],
                RUHI_COLORS["sky"],
                RUHI_COLORS["sunshine"],
                RUHI_COLORS["coral"],
            ],
        )
        + labs(
            x="log sigma^2",
            y="log theta",
            fill="NLL",
        )
        + _JOST_THEME
        + theme(figure_size=(width, height))
    )
    if optimum is not None:
        p = p + annotate(
            "point",
            x=optimum[0],
            y=optimum[1],
            color=RUHI_COLORS["coral"],
            size=4,
            shape="*",
        )
    return p


def plot_hyperparameter_sensitivity(
    df: pl.DataFrame,
    width: float = 6.4,
    height: float = 5.0,
) -> ggplot:
    """3x3 facet grid of 1D GP slices for different ell/sigma_f combos.

    Parameters
    ----------
    df : pl.DataFrame
        Must have columns: x, y_true, y_pred, y_lower, y_upper,
        ell, sigma_f. The ell and sigma_f columns are used for
        faceting.
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.

    Returns
    -------
    ggplot
    """
    pdf = df.to_pandas()

    p = (
        ggplot(pdf, aes(x="x"))
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
    df: pl.DataFrame,
    train_points: tuple | None = None,
    width: float = 3.2,
    height: float = 2.5,
) -> ggplot:
    """1D trust region illustration with confidence band.

    Parameters
    ----------
    df : pl.DataFrame
        Must have columns: x, y_pred, y_lower, y_upper,
        in_trust (bool).
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
    from plotnine import geom_ribbon

    pdf = df.to_pandas()

    p = (
        ggplot(pdf, aes(x="x"))
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
        tp_df = pl.DataFrame({
            "x": np.asarray(xs),
            "y": np.asarray(ys),
        }).to_pandas()
        p = p + geom_point(
            data=tp_df,
            mapping=aes(x="x", y="y"),
            color=RUHI_COLORS["coral"],
            size=2.5,
        )
    return p


def plot_variance_overlay(
    grid_x,
    grid_y,
    grid_energy,
    grid_variance,
    train_points: tuple | None = None,
    stationary: dict | None = None,
    width: float = 3.2,
    height: float = 2.5,
) -> ggplot:
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
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.

    Returns
    -------
    ggplot
    """
    var_df = _grid_to_long(
        grid_x, grid_y, grid_variance, "variance"
    )
    e_df = _grid_to_long(
        grid_x, grid_y, grid_energy, "energy"
    )
    # Merge energy into the variance frame for contour overlay
    combined = var_df.with_columns(
        e_df.get_column("energy"),
    ).to_pandas()

    p = (
        ggplot(combined, aes(x="x", y="y"))
        + geom_tile(aes(fill="variance"))
        + geom_contour(
            aes(z="energy"),
            color=RUHI_COLORS["teal"],
            alpha=0.6,
            size=0.4,
        )
        + scale_fill_gradient2(
            low="white",
            mid=RUHI_COLORS["sky"],
            high=RUHI_COLORS["coral"],
            midpoint=float(
                np.median(np.asarray(grid_variance))
            ),
        )
        + labs(x="x", y="y", fill="Variance")
        + _JOST_THEME
        + theme(figure_size=(width, height))
    )

    if train_points is not None:
        xs, ys = train_points
        tp_df = pl.DataFrame({
            "x": np.asarray(xs),
            "y": np.asarray(ys),
        }).to_pandas()
        p = p + geom_point(
            data=tp_df,
            mapping=aes(x="x", y="y"),
            color=RUHI_COLORS["teal"],
            size=1.5,
            shape="x",
        )

    if stationary is not None:
        for label, (sx, sy) in stationary.items():
            p = p + annotate(
                "point",
                x=sx,
                y=sy,
                color=RUHI_COLORS["coral"],
                size=3,
            ) + annotate(
                "text",
                x=sx,
                y=sy,
                label=label,
                color=RUHI_COLORS["teal"],
                size=8,
                va="bottom",
                ha="left",
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
    sel_df = pl.DataFrame({
        "pc1": np.asarray(selected_pc1),
        "pc2": np.asarray(selected_pc2),
        "group": ["Selected"] * len(selected_pc1),
    })
    prn_df = pl.DataFrame({
        "pc1": np.asarray(pruned_pc1),
        "pc2": np.asarray(pruned_pc2),
        "group": ["Pruned"] * len(pruned_pc1),
    })
    all_df = pl.concat([prn_df, sel_df]).to_pandas()

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
    df: pl.DataFrame,
    x: str = "image",
    y: str = "energy",
    color: str = "method",
    width: float = 3.2,
    height: float = 2.5,
) -> ggplot:
    """NEB energy profile across images.

    Parameters
    ----------
    df : pl.DataFrame
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
    pdf = df.to_pandas()
    methods = pdf[color].unique()
    palette = dict(zip(methods, _METHOD_PALETTE))

    p = (
        ggplot(pdf, aes(x=x, y=y, color=color))
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
