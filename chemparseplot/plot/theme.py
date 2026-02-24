"""Plotting themes and colormap utilities.

```{versionadded} 0.1.0
```
"""

import logging
from dataclasses import dataclass, replace

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

log = logging.getLogger(__name__)

# --- Color Definitions ---
RUHI_COLORS = {
    "coral": "#FF655D",
    "sunshine": "#F1DB4B",
    "teal": "#004D40",
    "sky": "#1E88E5",
    "magenta": "#D81B60",
}


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


# Register default colormaps
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
        }
    )


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


def get_theme(name: str, **overrides) -> PlotTheme:
    """Retrieves a theme by name and applies optional property overrides.

    ```{versionadded} 0.1.0
    ```
    """
    base = THEMES.get(name, RUHI_THEME)
    return replace(base, **{k: v for k, v in overrides.items() if v is not None})
