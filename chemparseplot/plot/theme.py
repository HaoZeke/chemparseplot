"""Plotting themes and colormap utilities.

```{versionadded} 0.1.0
```
"""

import logging
from dataclasses import dataclass, replace

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

from chemparseplot.plot.palette import RUHI_COLORS, RUHI_CYCLE

log = logging.getLogger(__name__)

# Accessibility: chemparseplot.plot.palette.palette_report() gives the WCAG
# contrast of each Ruhi colour on white and its CIEDE2000 separation under
# simulated colour-vision deficiency. Sunshine and coral are below the 3:1
# contrast for graphics, and teal/magenta (protan) and coral/magenta (tritan)
# are the closest pairs, so figures should not rely on colour alone: label
# lines directly or vary markers and dashes as well.


@dataclass(frozen=True)
class PlotTheme:
    """Holds all aesthetic parameters for a matplotlib theme.

    ```{versionadded} 0.1.0
    ```
    """

    name: str
    font_family: str
    font_size: int
    facecolor: str
    textcolor: str
    edgecolor: str
    gridcolor: str
    cmap_profile: str
    cmap_landscape: str
    highlight_color: str
    # Structure captions. Every theme takes this default, so it is the one
    # switch. Atkinson Hyperlegible is the ruhi face. Liberation Sans, with
    # the Arial metrics Nature prefers, is pending. A default also keeps
    # existing PlotTheme(...) calls working.
    caption_font: str = "Atkinson Hyperlegible"


# --- Theme Definitions ---
BATLOW_THEME = PlotTheme(
    name="cmc.batlow",
    font_family="Atkinson Hyperlegible",
    font_size=12,
    facecolor="white",
    textcolor="black",
    edgecolor="black",
    gridcolor="#FFFFFF",
    cmap_profile="cmc.batlow",
    cmap_landscape="cmc.batlow",
    highlight_color="#FF0000",
)

RUHI_THEME = PlotTheme(
    name="ruhi",
    font_family="Atkinson Hyperlegible",
    font_size=12,
    facecolor="white",
    textcolor="black",
    edgecolor="black",
    gridcolor="floralwhite",
    cmap_profile="ruhi_diverging",
    cmap_landscape="ruhi_diverging",
    highlight_color="black",
)

THEMES = {
    "cmc.batlow": BATLOW_THEME,
    "ruhi": RUHI_THEME,
}


def build_cmap(hex_list, name):
    """Build and register a LinearSegmentedColormap from a list of hex colors.

    ```{versionadded} 0.1.0
    ```
    """
    cols = [c.strip() for c in hex_list]
    cmap = LinearSegmentedColormap.from_list(name, cols, N=256)
    try:
        mpl.colormaps.register(cmap)
    except ValueError:
        pass  # Already registered
    return cmap


# Linear-RGB blends. Profiles and landscapes on the Ruhi theme use
# ruhi_diverging (teal, sky, magenta, coral, sunshine). Its lightness is
# not monotone. ruhi_seq and ruhi_div stay registered for callers that ask
# for a perceptually ordered map.
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

build_cmap(
    [
        RUHI_COLORS["coral"],
        RUHI_COLORS["sunshine"],
        RUHI_COLORS["teal"],
        RUHI_COLORS["sky"],
        RUHI_COLORS["magenta"],
    ],
    name="ruhi_full",
)


def setup_global_theme(theme: PlotTheme):
    """Sets global plt.rcParams based on the theme.

    ```{versionadded} 0.1.0
    ```
    """
    log.info(f"Setting global rcParams for {theme.name} theme")

    font_family = theme.font_family
    try:
        mpl.font_manager.findfont(font_family, fallback_to_default=False)
    except Exception:
        log.warning(f"Font '{font_family}' not found. Falling back to 'sans-serif'.")
        font_family = "sans-serif"

    plt.rcParams.update(
        {
            "font.size": theme.font_size,
            "font.family": font_family,
            "text.color": theme.textcolor,
            "axes.labelcolor": theme.textcolor,
            "xtick.color": theme.textcolor,
            "ytick.color": theme.textcolor,
            "axes.edgecolor": theme.edgecolor,
            "axes.titlecolor": theme.textcolor,
            "figure.facecolor": theme.facecolor,
            "axes.titlesize": theme.font_size * 1.1,
            "axes.labelsize": theme.font_size,
            "xtick.labelsize": theme.font_size,
            "ytick.labelsize": theme.font_size,
            "legend.fontsize": theme.font_size,
            "savefig.facecolor": theme.facecolor,
            "savefig.transparent": False,
            "axes.prop_cycle": cycler(color=list(RUHI_CYCLE)),
        }
    )
    if theme.cmap_landscape in mpl.colormaps:
        plt.rcParams["image.cmap"] = theme.cmap_landscape


def apply_axis_theme(ax: plt.Axes, theme: PlotTheme):
    """Applies theme properties specific to an axis instance.

    ```{versionadded} 0.1.0
    ```
    """
    ax.set_facecolor(theme.facecolor)
    for spine in ax.spines.values():
        spine.set_edgecolor(theme.edgecolor)
    ax.tick_params(axis="x", colors=theme.textcolor)
    ax.tick_params(axis="y", colors=theme.textcolor)
    ax.yaxis.label.set_color(theme.textcolor)
    ax.xaxis.label.set_color(theme.textcolor)
    ax.title.set_color(theme.textcolor)


def setup_publication_theme(theme: PlotTheme, journal: str | None = None):
    """Extended theme setup for publication-quality figures.

    Calls ``setup_global_theme`` and adds line widths, spine removal,
    DPI, and tight bounding box defaults. With ``journal`` (``"jctc"``,
    ``"jcp"``, ``"cpc"``, ``"pccp"`` or a preset key), applies the journal
    style of :mod:`chemparseplot.plot.journal` instead, whose sizes all
    derive from the journal's column widths and LaTeX font sizes.

    ```{versionadded} 1.3.0
    ```

    ```{versionchanged} 1.14.0
    Added *journal*.
    ```
    """
    setup_global_theme(theme)
    if journal is not None:
        # Deferred because journal imports this module.
        from chemparseplot.plot.journal import apply_style  # noqa: PLC0415

        apply_style(journal, theme=theme)
        return
    plt.rcParams.update(
        {
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def get_theme(name: str, **overrides) -> PlotTheme:
    """Retrieves a theme by name and applies optional property overrides.

    ```{versionadded} 0.1.0
    ```
    """
    base = THEMES.get(name, RUHI_THEME)
    return replace(base, **{k: v for k, v in overrides.items() if v is not None})
