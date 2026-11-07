# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Journal figure presets and the publication style derived from them.

A :class:`JournalPreset` holds what a journal fixes for its figures: column
widths, the font sizes of its LaTeX class, the smallest lettering and line
weight it accepts, the raster resolution it asks for and the faces it
recommends. :func:`figure_style` derives every other length from those
numbers, so a figure needs no hand-tuned sizes:

- Text uses the class sizes (``\\normalsize``, ``\\small`` and
  ``\\footnotesize``), converted from TeX points (72.27 per inch) to the
  PostScript points matplotlib uses (72 per inch), and never goes below the
  journal minimum. Axis labels and panel letters are at the body size, tick
  labels and legends at ``\\small``.
- Line widths follow the narrow : wide : extra-wide = 1 : 2 : 4 series of
  ISO 128-2:2020. The narrow width is one step of the series (a factor
  sqrt(2)) below the ISO 3098-1:2015 type B letter stroke d = h/10, where h
  is the capital height of the face at the body size, and never goes below
  the journal minimum.
- Tick lengths, gaps and marker sizes are ISO 3098-1 type B proportions of
  the same h: x-height 0.7 h for major ticks, descender 0.3 h for minor
  ticks, word space 0.6 h between ticks and labels, character space 0.2 h
  around panels, the mixed-case baseline pitch 1.5 h between stacked
  labels, and markers as tall as a capital.

```{versionadded} 1.14.0
```
"""

from __future__ import annotations

import math
import string
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib import font_manager, patheffects
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.ticker import (
    LogFormatterSciNotation,
    LogLocator,
    NullFormatter,
    ScalarFormatter,
)
from matplotlib.transforms import ScaledTranslation

from chemparseplot.plot.palette import (
    RUHI_CYCLE,
    WCAG_GRAPHICS_CONTRAST,
    WCAG_TEXT_CONTRAST,
    contrast_ratio,
    readable_color,
)
from chemparseplot.plot.theme import get_theme

if TYPE_CHECKING:
    from os import PathLike

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D
    from matplotlib.text import Annotation, Text
    from matplotlib.typing import ColorType
    from numpy.typing import ArrayLike

    from chemparseplot.plot.theme import PlotTheme

# --- Units -------------------------------------------------------------------

MM_PER_INCH = 25.4
MM_PER_CM = 10.0
BP_PER_INCH = 72.0
TEX_PT_PER_INCH = 72.27


def mm(value: float) -> float:
    """Millimetres to inches."""
    return value / MM_PER_INCH


def cm(value: float) -> float:
    """Centimetres to inches."""
    return value * MM_PER_CM / MM_PER_INCH


def bp(value: float) -> float:
    """PostScript (big) points to inches."""
    return value / BP_PER_INCH


def tex_pt(value: float) -> float:
    """TeX points to inches."""
    return value / TEX_PT_PER_INCH


def tex_pt_to_bp(value: float) -> float:
    """TeX points to PostScript points, the unit of matplotlib font sizes."""
    return value * BP_PER_INCH / TEX_PT_PER_INCH


def bp_to_tex_pt(value: float) -> float:
    """PostScript points to TeX points."""
    return value * TEX_PT_PER_INCH / BP_PER_INCH


# --- Standards ---------------------------------------------------------------

# ISO 3098-1:2015, Table 2 (lettering type B), as fractions of the capital
# height h. The baseline pitch is b2, for mixed case without diacritics.
_STROKE = 1 / 10
_X_HEIGHT = 7 / 10
_DESCENDER = 3 / 10
_CHAR_SPACE = 2 / 10
_WORD_SPACE = 6 / 10
_BASELINE_PITCH = 15 / 10

# ISO 128-2:2020: widths step by sqrt(2); narrow : wide : extra-wide = 1 : 2 : 4.
_LINE_STEP = math.sqrt(2.0)
_WIDE = 2.0
_EXTRA_WIDE = 4.0

# ISO 128-2:2020 line elements, in multiples of the line width d: dots 0.5 d,
# gaps 3 d, dashes 12 d, long dashes 24 d. Matplotlib scales dash patterns by
# the line width (lines.scale_dashes), so these are the patterns themselves.
_DOT = 0.5
_GAP = 3.0
_DASH = 12.0
_LONG_DASH = 24.0
# The types built from one dash kind and dots. Those with fewer kinds of mark
# come first, and types with as many follow ISO 128-2's numbering.
LINE_TYPES = (
    ("01", "continuous", "solid"),
    ("02", "dashed", (0, (_DASH, _GAP))),
    ("07", "dotted", (0, (_DOT, _GAP))),
    ("04", "long-dashed dotted", (0, (_LONG_DASH, _GAP, _DOT, _GAP))),
    ("10", "dashed dotted", (0, (_DASH, _GAP, _DOT, _GAP))),
)

# NIST SP 811 (2008), section 7.9: prefixes are chosen so that numerical
# values lie between 0.1 and 1000. Matplotlib switches to powers of ten when
# the exponent is <= -2 or >= 3, which keeps that range in plain numbers.
SI_POWERLIMITS = (-2, 3)

# Panel aspect (height / width) when the data give none: "about 50 percent
# wider than tall", Tufte, The Visual Display of Quantitative Information
# (1983), p. 190.
PANEL_ASPECT = 2.0 / 3.0

# Faces that share the advance widths of the faces journals name, so text
# set in them takes the same room.
_METRIC_CLONES = {
    "Helvetica": ("Nimbus Sans", "Liberation Sans", "Arimo"),
    "Arial": ("Liberation Sans", "Arimo"),
}
# pdf.fonttype 42 embeds TrueType outlines; CFF-flavoured OpenType cannot be.
_TRUETYPE_SUFFIXES = {".ttf", ".ttc"}
_FALLBACK_FACE = "DejaVu Sans"


# --- Presets -----------------------------------------------------------------


@dataclass(frozen=True)
class JournalPreset:
    """What a journal fixes for its figures.

    Widths and heights are in inches, ``class_sizes`` in TeX points
    (``\\normalsize``, ``\\small``, ``\\footnotesize`` of the LaTeX class
    in its two-column layout, or of the published page where no class
    reproduces it), ``min_text`` and ``min_line`` in PostScript points.
    ``None`` means the journal states no value. Where a guideline gives a
    length in two units, the two are rounded conversions of each other and
    the smaller is used, so the figure fits both.

    ```{versionadded} 1.14.0
    ```
    """

    key: str
    name: str
    widths: Mapping[str, float]
    max_height: float | None
    class_sizes: tuple[float, float, float]
    min_text: float
    min_line: float | None
    raster_dpi: int
    fonts: tuple[str, ...]
    panel_label: str
    sources: tuple[str, ...]

    def width(self, columns: str = "single") -> float:
        """Figure width in inches for ``columns``."""
        try:
            return self.widths[columns]
        except KeyError:
            msg = (
                f"{self.name} has no {columns!r} width; choose from {sorted(self.widths)}"
            )
            raise ValueError(msg) from None


ACS = JournalPreset(
    key="acs",
    name="ACS (J. Chem. Theory Comput., J. Chem. Inf. Model.)",
    widths={"single": min(bp(240), 3.33), "double": min(bp(504), 7.0)},
    max_height=min(bp(660), 9.167),
    # achemso sets these journals only in its 12 pt submission layout. The
    # published page (J. Chem. Theory Comput., doi:10.1021/acs.jctc.5c00353)
    # sets text at 10, 9 and 8 PostScript points.
    class_sizes=tuple(bp_to_tex_pt(size) for size in (10.0, 9.0, 8.0)),
    min_text=4.5,
    min_line=0.5,
    raster_dpi=1200,
    fonts=("Helvetica", "Arial"),
    panel_label="({})",
    sources=(
        "https://researcher-resources.acs.org/publish/author_guidelines?coden=jctcce",
        "https://researcher-resources.acs.org/publish/author_guidelines?coden=jcisd8",
    ),
)

AIP = JournalPreset(
    key="aip",
    name="AIP (J. Chem. Phys.)",
    widths={"single": min(3.37, cm(8.5)), "double": min(6.69, cm(17.0))},
    max_height=min(8.25, cm(21.1)),
    class_sizes=(10.0, 9.0, 8.0),
    min_text=8.0,
    min_line=0.5,
    raster_dpi=1200,
    fonts=(),
    panel_label="({})",
    sources=("https://publishing.aip.org/resources/researchers/author-instructions/",),
)

ELSEVIER = JournalPreset(
    key="elsevier",
    name="Elsevier (Comput. Phys. Commun.)",  # codespell:ignore commun
    widths={
        "single": min(mm(90), bp(255)),
        "onehalf": min(mm(140), bp(397)),
        "double": min(mm(190), bp(539)),
    },
    max_height=None,
    class_sizes=(10.0, 9.0, 8.0),
    min_text=7.0,
    min_line=0.25,
    raster_dpi=1000,
    fonts=("Arial",),
    panel_label="({})",
    sources=(
        "https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing",
        "https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-faq",
    ),
)

RSC = JournalPreset(
    key="rsc",
    name="RSC (Phys. Chem. Chem. Phys.)",
    widths={"single": cm(8.3), "double": cm(17.1)},
    max_height=cm(23.3),
    class_sizes=(9.0, 8.0, 7.0),
    # The only lettering size RSC gives: captions and atom labels in its
    # structure-drawing settings.
    min_text=7.0,
    min_line=None,
    raster_dpi=600,
    fonts=("Arial", "Helvetica"),
    panel_label="({})",
    sources=(
        "https://www.rsc.org/publishing/publish-with-us/publish-a-journal-article/pccp",
        "https://www.rsc.org/publishing/publish-with-us/publish-a-journal-article/article-templates",
    ),
)

PRESETS: dict[str, JournalPreset] = {p.key: p for p in (ACS, AIP, ELSEVIER, RSC)}
ALIASES = {
    "jctc": "acs",
    "jcim": "acs",
    "jcp": "aip",
    "cpc": "elsevier",
    "pccp": "rsc",
}
DEFAULT_JOURNAL = "acs"


def get_preset(journal: str | JournalPreset = DEFAULT_JOURNAL) -> JournalPreset:
    """Look up a preset by key (``"acs"``) or journal alias (``"jctc"``).

    ```{versionadded} 1.14.0
    ```
    """
    if isinstance(journal, JournalPreset):
        return journal
    key = journal.lower()
    key = ALIASES.get(key, key)
    try:
        return PRESETS[key]
    except KeyError:
        names = sorted({*PRESETS, *ALIASES})
        msg = f"Unknown journal {journal!r}; expected one of {names}"
        raise ValueError(msg) from None


# --- Fonts -------------------------------------------------------------------


def resolve_font(candidates: Iterable[str]) -> str:
    """First installed TrueType face among ``candidates`` or their metric clones.

    Falls back to DejaVu Sans, which matplotlib ships.

    ```{versionadded} 1.14.0
    ```
    """
    for name in candidates:
        for face in (name, *_METRIC_CLONES.get(name, ())):
            try:
                path = font_manager.findfont(
                    FontProperties(family=face), fallback_to_default=False
                )
            except ValueError:
                continue
            if Path(path).suffix.lower() in _TRUETYPE_SUFFIXES:
                return face
    return _FALLBACK_FACE


def cap_height(face: str | None = None) -> float:
    """Capital height of ``face`` (default: the current rcParams face) per point of size.

    Measured from the outline of "H".

    ```{versionadded} 1.14.0
    ```
    """
    prop = FontProperties(family=face) if face else FontProperties()
    return float(TextPath((0, 0), "H", size=1.0, prop=prop).vertices[:, 1].max())


# --- Style -------------------------------------------------------------------


@dataclass(frozen=True)
class JournalStyle:
    """Lengths derived from a preset. Sizes are in PostScript points.

    ```{versionadded} 1.14.0
    ```
    """

    preset: JournalPreset
    font: str
    body: float
    small: float
    footnote: float
    cap: float
    narrow: float
    wide: float
    extra: float
    tick: float
    minor_tick: float
    pad: float
    gap: float
    marker: float
    textcolor: str
    facecolor: str

    @property
    def pitch(self) -> float:
        """Smallest baseline spacing for ``\\small`` labels stacked in a column."""
        return _BASELINE_PITCH * self.cap * self.small / self.body

    @property
    def marker_large(self) -> float:
        """Marker one step of the sqrt(2) series above :attr:`marker`, for
        points that must stand out from the data (minima, saddles)."""
        return _LINE_STEP * self.marker

    def width(self, columns: str = "single") -> float:
        """Figure width in inches for ``columns``, as :meth:`JournalPreset.width`."""
        return self.preset.width(columns)

    def halo(self, color: str | None = None) -> list:
        """Path effects that outline text or lines in ``color`` (default the
        text colour), one narrow width wide, so they read on any fill."""
        return [
            patheffects.withStroke(
                linewidth=self.wide, foreground=color or self.textcolor
            )
        ]

    def rc(self) -> dict:
        """rcParams for this style."""
        face = self.font
        lines = {
            "axes.linewidth": self.narrow,
            "grid.linewidth": self.narrow,
            "patch.linewidth": self.narrow,
            "hatch.linewidth": self.narrow,
            "lines.linewidth": self.wide,
            "lines.scale_dashes": True,
            "lines.markeredgewidth": self.narrow,
            "lines.markersize": self.marker,
            "contour.linewidth": self.narrow,
            # Caps span 2 * capsize, as wide as a marker.
            "errorbar.capsize": self.marker / 2.0,
        }
        ticks = {}
        for axis in ("xtick", "ytick"):
            ticks |= {
                f"{axis}.major.width": self.narrow,
                f"{axis}.minor.width": self.narrow,
                f"{axis}.major.size": self.tick,
                f"{axis}.minor.size": self.minor_tick,
                f"{axis}.major.pad": self.pad,
                f"{axis}.minor.pad": self.pad,
                f"{axis}.labelsize": self.small,
                f"{axis}.color": self.textcolor,
                f"{axis}.labelcolor": self.textcolor,
            }
        return {
            "font.family": "sans-serif",
            "font.sans-serif": [face, _FALLBACK_FACE],
            "font.size": self.body,
            "axes.labelsize": self.body,
            "axes.titlesize": self.body,
            "figure.titlesize": self.body,
            "figure.labelsize": self.body,
            "legend.fontsize": self.small,
            "legend.title_fontsize": self.small,
            "legend.frameon": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": face,
            "mathtext.sf": face,
            "mathtext.it": f"{face}:italic",
            "mathtext.bf": f"{face}:bold",
            "mathtext.cal": f"{face}:italic",
            "axes.labelpad": self.pad,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=list(RUHI_CYCLE)),
            "axes.formatter.use_mathtext": True,
            "axes.formatter.limits": list(SI_POWERLIMITS),
            "axes.formatter.useoffset": False,
            "axes.unicode_minus": True,
            "axes.edgecolor": self.textcolor,
            "axes.labelcolor": self.textcolor,
            "axes.titlecolor": self.textcolor,
            "axes.facecolor": self.facecolor,
            "text.color": self.textcolor,
            "image.cmap": "ruhi_diverging",
            "figure.facecolor": self.facecolor,
            "figure.constrained_layout.w_pad": bp(self.gap),
            "figure.constrained_layout.h_pad": bp(self.gap),
            "figure.constrained_layout.wspace": 0.0,
            "figure.constrained_layout.hspace": 0.0,
            "savefig.dpi": self.preset.raster_dpi,
            "savefig.bbox": "standard",
            "savefig.facecolor": self.facecolor,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "chemparseplot",
            **lines,
            **ticks,
        }


def figure_style(
    journal: str | JournalPreset = DEFAULT_JOURNAL,
    *,
    font: str | None = None,
    theme: str | PlotTheme = "ruhi",
) -> JournalStyle:
    """Derive the figure style for ``journal``.

    Text is set in ``font``, by default the face of ``theme`` (a theme name
    or :class:`~chemparseplot.plot.theme.PlotTheme`), which also sets the
    text and background colours. Where the theme's face is not installed,
    the faces the journal names stand in.

    ```{versionadded} 1.14.0
    ```
    """
    preset = get_preset(journal)
    plot_theme = get_theme(theme) if isinstance(theme, str) else theme
    candidates = (font,) if font else (plot_theme.font_family, *preset.fonts)
    face = resolve_font(candidates)
    body, small, footnote = (
        max(tex_pt_to_bp(size), preset.min_text) for size in preset.class_sizes
    )
    h = body * cap_height(face)
    narrow = _STROKE * h / _LINE_STEP
    if preset.min_line is not None:
        narrow = max(narrow, preset.min_line)
    return JournalStyle(
        preset=preset,
        font=face,
        body=body,
        small=small,
        footnote=footnote,
        cap=h,
        narrow=narrow,
        wide=_WIDE * narrow,
        extra=_EXTRA_WIDE * narrow,
        tick=_X_HEIGHT * h,
        minor_tick=_DESCENDER * h,
        pad=_WORD_SPACE * h,
        gap=_CHAR_SPACE * h,
        marker=h,
        textcolor=plot_theme.textcolor,
        facecolor=plot_theme.facecolor,
    )


def apply_style(
    journal: str | JournalPreset = DEFAULT_JOURNAL, **kwargs: Any
) -> JournalStyle:
    """Derive the style for ``journal`` and set it in the global rcParams.

    Keyword arguments go to :func:`figure_style`.

    ```{versionadded} 1.14.0
    ```
    """
    style = figure_style(journal, **kwargs)
    mpl.rcParams.update(style.rc())
    return style


@contextmanager
def journal_style(
    journal: str | JournalPreset = DEFAULT_JOURNAL, **kwargs: Any
) -> Iterator[JournalStyle]:
    """Context manager form of :func:`apply_style`.

    ```{versionadded} 1.14.0
    ```
    """
    style = figure_style(journal, **kwargs)
    with mpl.rc_context(style.rc()):
        yield style


# --- Figures -----------------------------------------------------------------


def journal_figure(
    journal: str | JournalPreset = DEFAULT_JOURNAL,
    columns: str = "single",
    *,
    nrows: int = 1,
    ncols: int = 1,
    aspect: float | None = None,
    height: float | None = None,
    style: JournalStyle | None = None,
    **subplots_kw: Any,
) -> tuple[Figure, Any]:
    """Figure at the exact column width of ``journal``, with its style applied.

    The style goes into the global rcParams (as
    :func:`chemparseplot.plot.theme.setup_publication_theme` does) unless
    ``style`` is given, in which case the caller manages rcParams. The
    height is ``height`` inches, or the width times ``aspect`` (panel
    height over width, default :data:`PANEL_ASPECT`) times ``nrows / ncols``.

    Returns ``(fig, axes)`` as :func:`matplotlib.pyplot.subplots` does.

    ```{versionadded} 1.14.0
    ```
    """
    style = style or apply_style(journal)
    width = style.width(columns)
    if height is None:
        aspect = PANEL_ASPECT if aspect is None else aspect
        height = width * aspect * nrows / ncols
    limit = style.preset.max_height
    if limit is not None and height > limit:
        msg = (
            f"Height {height:.3f} in exceeds the {style.preset.name} maximum of "
            f"{limit:.3f} in"
        )
        raise ValueError(msg)
    return plt.subplots(
        nrows, ncols, figsize=(width, height), layout="constrained", **subplots_kw
    )


def fit_height(fig: Figure, *, dpi: float | None = None) -> float:
    """Shrink the figure height to its content, keeping the width.

    Axes with a fixed aspect (energy landscapes) leave empty bands under
    constrained layout; this removes them. Stops when the height changes by
    less than one output pixel at ``dpi`` (default ``savefig.dpi``), or
    stops shrinking. Returns the new height in inches.

    ```{versionadded} 1.14.0
    ```
    """
    dpi = _output_dpi(fig, dpi)
    engine = fig.get_layout_engine()
    pad = engine.get()["h_pad"] if engine is not None else 0.0
    width, height = fig.get_size_inches()
    while True:
        fig.draw_without_rendering()
        renderer = fig.canvas.get_renderer()
        content = fig.get_tightbbox(renderer).height
        new = content + 2.0 * pad
        if new >= height or height - new < 1.0 / dpi:
            return float(height)
        height = new
        fig.set_size_inches(width, height)


def _output_dpi(fig, dpi: float | None = None) -> float:
    dpi = dpi or mpl.rcParams["savefig.dpi"]
    return fig.dpi if dpi == "figure" else dpi


def save_figure(
    fig: Figure,
    path: str | PathLike[str],
    *,
    dpi: float | None = None,
    **savefig_kw: Any,
) -> Path:
    """Save ``fig`` at its exact size, with fonts embedded and fixed metadata.

    No tight bounding box is applied, so the page keeps the column width.
    PDF and SVG carry no creation date and SVG ids come from a fixed salt,
    so a script writes the same bytes on every run. Raster formats use
    ``savefig.dpi``, which :func:`apply_style` sets to the journal
    resolution.

    ```{versionadded} 1.14.0
    ```
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix.lstrip(".").lower() or mpl.rcParams["savefig.format"]
    creator = f"chemparseplot {_version()}"
    meta = {
        "pdf": {"Creator": creator, "CreationDate": None},
        "svg": {"Creator": creator, "Date": None},
        "png": {"Software": creator},
    }.get(fmt)
    kwargs = {"metadata": meta} if meta is not None else {}
    with mpl.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(path, dpi=dpi, **kwargs, **savefig_kw)
    return path


def _version() -> str:
    try:
        return metadata.version("chemparseplot")
    except metadata.PackageNotFoundError:
        return "unknown"


# --- Panel letters -----------------------------------------------------------


def _panel_axes(fig) -> list:
    return [ax for ax in fig.axes if ax.get_label() != "<colorbar>" and ax.get_visible()]


def label_panels(
    target: Figure | Sequence[Axes] | np.ndarray,
    labels: Sequence[str] | None = None,
    *,
    fmt: str | None = None,
    journal: str | JournalPreset | None = None,
    fontweight: str = "normal",
) -> list[Text]:
    """Letter the panels of a figure, top left of each, in one style.

    ``target`` is a figure or a sequence of axes (default order: as added
    to the figure). Each letter sits above the axes, its left edge on the
    left edge of the panel's tick and axis labels; panels in one grid
    column share that edge. ``fmt`` defaults to the journal's panel label
    format, "(a)".

    ```{versionadded} 1.14.0
    ```
    """
    if isinstance(target, mpl.figure.Figure):
        fig, axes = target, _panel_axes(target)
    else:
        axes = list(np.ravel(target))
        fig = axes[0].figure
    fmt = fmt or get_preset(journal or DEFAULT_JOURNAL).panel_label
    labels = labels or string.ascii_lowercase[: len(axes)]
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    left = {}
    for ax in axes:
        x0 = ax.get_tightbbox(renderer).x0
        col = _column(ax)
        left[col] = min(left.get(col, x0), x0)
    gap = mpl.rcParams["figure.constrained_layout.h_pad"]
    texts = []
    for ax, label in zip(axes, labels, strict=False):
        dx = (left[_column(ax)] - ax.get_window_extent(renderer).x0) / fig.dpi
        offset = ScaledTranslation(dx, gap, fig.dpi_scale_trans)
        texts.append(
            ax.text(
                0.0,
                1.0,
                fmt.format(label),
                transform=ax.transAxes + offset,
                ha="left",
                va="bottom",
                fontsize=mpl.rcParams["axes.labelsize"],
                fontweight=fontweight,
            )
        )
    return texts


def _column(ax):
    spec = ax.get_subplotspec()
    return id(ax) if spec is None else (id(spec.get_gridspec()), spec.colspan.start)


# --- Direct labels and legends -----------------------------------------------


def _spread(y: np.ndarray, pitch: float) -> np.ndarray:
    """Closest positions to sorted ``y`` that are at least ``pitch`` apart.

    Least squares under the spacing constraints: subtract ``k * pitch``,
    take the isotonic (non-decreasing) regression by pool-adjacent-violators,
    add the offsets back.
    """
    z = y - pitch * np.arange(len(y))
    blocks: list[list[float]] = []
    for value in z:
        blocks.append([value, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, n2 = blocks.pop()
            v1, n1 = blocks.pop()
            blocks.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    fitted = np.concatenate([np.full(int(n), v) for v, n in blocks])
    return fitted + pitch * np.arange(len(y))


def direct_labels(
    ax: Axes, lines: Sequence[Line2D] | None = None, *, side: str = "right"
) -> list[Annotation]:
    """Label lines at their ends instead of in a legend.

    Labels use each line's label and a shade of its colour that reaches the
    WCAG text contrast on the axes background. Labels that would collide are
    moved apart to at least one baseline pitch (ISO 3098-1 b2, 1.5 capital
    heights), as little as possible in the least-squares sense. Constrained
    layout makes room for the labels, which moves the line ends; the labels
    are spread again until the ends move by less than one output pixel at
    ``savefig.dpi``, or stop settling.

    ```{versionadded} 1.14.0
    ```
    """
    lines = [
        ln
        for ln in (lines if lines is not None else ax.get_lines())
        if ln.get_label() and not ln.get_label().startswith("_") and ln.get_visible()
    ]
    if not lines:
        return []
    fig = ax.figure
    fig.draw_without_rendering()
    size = mpl.rcParams["legend.fontsize"]
    size = FontProperties(size=size).get_size_in_points()
    h = cap_height() * size
    pitch = _BASELINE_PITCH * h * fig.dpi / BP_PER_INCH
    background = ax.get_facecolor()
    anchors = []
    for line in lines:
        x, y = (np.asarray(v, dtype=float) for v in line.get_data())
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            continue
        i = np.flatnonzero(ok)[-1 if side == "right" else 0]
        disp = ax.transData.transform((x[i], y[i]))
        anchors.append((disp[1], (x[i], y[i]), line))
    anchors.sort(key=lambda item: item[0])
    gap = _CHAR_SPACE * h + mpl.rcParams["lines.markersize"] / 2.0
    dx = gap if side == "right" else -gap
    texts = [
        ax.annotate(
            line.get_label(),
            xy=xy,
            xytext=(dx, 0.0),
            textcoords="offset points",
            ha="left" if side == "right" else "right",
            va="center",
            color=readable_color(line.get_color(), background, WCAG_TEXT_CONTRAST),
            fontsize=size,
            annotation_clip=False,
        )
        for _, xy, line in anchors
    ]
    tol = fig.dpi / _output_dpi(fig)
    placed, moved = None, np.inf
    while True:
        fig.draw_without_rendering()
        ends = np.array([ax.transData.transform(t.xy)[1] for t in texts])
        if placed is not None:
            step = float(np.max(np.abs(ends - placed)))
            if step < tol or step >= moved:
                break
            moved = step
        for text, end, target in zip(texts, ends, _spread(ends, pitch), strict=True):
            text.xyann = (dx, (target - end) * BP_PER_INCH / fig.dpi)
        placed = ends
    return texts


# Matplotlib's inside legend locations in the order its loc="best" search tries
# them (Legend.codes 1 to 10). "right" and "center right" are one place, tried
# at the code of "right".
_INSIDE = (
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "center right",
    "center left",
    "lower center",
    "upper center",
    "center",
)


def _no_marker(marker) -> bool:
    return marker in (None, "None", "none", "", " ")


def _overlaps_data(ax, bbox, renderer, exclude=()) -> bool:
    px_per_pt = ax.figure.dpi / BP_PER_INCH
    for line in ax.get_lines():
        if not line.get_visible() or line in exclude:
            continue
        path = line.get_transform().transform_path(line.get_path())
        drawn = line.get_linestyle() not in ("None", "", " ")
        if drawn and path.intersects_bbox(bbox, filled=False):
            return True
        if not _no_marker(line.get_marker()):
            radius = line.get_markersize() * px_per_pt / 2.0
            if bbox.padded(radius).count_contains(path.vertices):
                return True
    for coll in ax.collections:
        if not coll.get_visible():
            continue
        if isinstance(coll, QuadMesh):
            hit = coll.get_window_extent(renderer).overlaps(bbox)
        elif isinstance(coll, PathCollection):
            sizes = coll.get_sizes()
            radius = (np.sqrt(sizes.max()) if len(sizes) else 0.0) * px_per_pt / 2.0
            pts = coll.get_offset_transform().transform(coll.get_offsets())
            hit = bool(bbox.padded(radius).count_contains(pts))
        else:
            faces = coll.get_facecolor()
            filled = bool(len(faces)) and bool(np.any(faces[:, -1] > 0.0))
            transform = coll.get_transform()
            hit = any(
                transform.transform_path(p).intersects_bbox(bbox, filled=filled)
                for p in coll.get_paths()
            )
        if hit:
            return True
    artists = [a for a in (*ax.images, *ax.patches, *ax.texts) if a not in exclude]
    return any(
        a.get_visible() and a.get_window_extent(renderer).overlaps(bbox) for a in artists
    )


def label_lines(
    ax: Axes,
    lines: Sequence[Line2D] | None = None,
    *,
    fallback: bool = True,
    **legend_kw: Any,
) -> list[Annotation] | list[Legend]:
    """Direct labels where they touch no data, otherwise :func:`smart_legend`.

    Returns the label texts, or a one-element list holding the legend. With
    ``fallback=False`` labels that would touch data are removed and an empty
    list is returned, so a figure can try its panels in turn.

    ```{versionadded} 1.14.0
    ```
    """
    texts = direct_labels(ax, lines)
    if not texts:
        return []
    fig = ax.figure
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    figbox = fig.bbox
    clash = False
    for text in texts:
        box = text.get_window_extent(renderer)
        others = [t for t in texts if t is not text]
        outside = box.x1 > figbox.x1 or box.x0 < figbox.x0
        if outside or _overlaps_data(ax, box, renderer, exclude=texts):
            clash = True
            break
        if any(box.overlaps(o.get_window_extent(renderer)) for o in others):
            clash = True
            break
    if not clash:
        return texts
    for text in texts:
        text.remove()
    return [smart_legend(ax, **legend_kw)] if fallback else []


def label_hline(
    ax: Axes,
    line: Line2D,
    label: str,
    *,
    color: ColorType | None = None,
    ignore: Sequence[Artist] = (),
) -> Annotation | Legend:
    """Label a horizontal reference line at one of its ends.

    Tries the right end above the line, then below it, then the left end
    above and below, and keeps the first label that touches no data.
    Artists in ``ignore``, such as faint context lines, may be crossed.
    Labels are ``\\small`` and one ISO 3098-1 character space from the
    line. When every end is taken the line goes into a :func:`smart_legend`
    instead. Returns the label text or the legend.

    ```{versionadded} 1.14.0
    ```
    """
    fig = ax.figure
    size = FontProperties(size=mpl.rcParams["legend.fontsize"]).get_size_in_points()
    gap = _CHAR_SPACE * cap_height() * size
    y = float(np.asarray(line.get_ydata(), dtype=float)[0])
    color = color or mpl.rcParams["text.color"]
    candidates = (
        (1.0, "right", "bottom"),
        (1.0, "right", "top"),
        (0.0, "left", "bottom"),
        (0.0, "left", "top"),
    )

    def place(x, ha, va):
        return ax.annotate(
            label,
            xy=(x, y),
            xycoords=ax.get_yaxis_transform(),
            xytext=(0.0, gap if va == "bottom" else -gap),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=size,
            color=color,
        )

    for candidate in candidates:
        text = place(*candidate)
        fig.draw_without_rendering()
        renderer = fig.canvas.get_renderer()
        box = text.get_window_extent(renderer)
        if not _overlaps_data(ax, box, renderer, exclude=(line, text, *ignore)):
            return text
        text.remove()
    return smart_legend(ax, [line], [label])


def series_style(index: int, background: ColorType = "white") -> dict:
    """Colour and ISO 128-2 line type of data series ``index``.

    Colours are the Ruhi line colours that reach the WCAG 1.4.11 contrast
    for graphics on ``background``; line types cycle independently, so
    series differ in both, stay apart in greyscale and for readers with
    colour-vision deficiencies, and a pair repeats only after the least
    common multiple of the two cycle lengths.

    ```{versionadded} 1.14.0
    ```
    """
    colors = [
        c for c in RUHI_CYCLE if contrast_ratio(c, background) >= WCAG_GRAPHICS_CONTRAST
    ]
    return {
        "color": colors[index % len(colors)],
        "linestyle": LINE_TYPES[index % len(LINE_TYPES)][2],
    }


def smart_legend(
    ax: Axes,
    handles: Sequence[Artist] | None = None,
    labels: Sequence[str] | None = None,
    **kwargs: Any,
) -> Legend:
    """Legend inside the axes where it covers no data, otherwise above them.

    ``handles`` default to the labelled artists of ``ax`` and ``labels`` to
    the handles' labels. Matplotlib draws one legend per axes, so the
    entries of a legend already there are kept. Tries matplotlib's inside
    locations in the order its ``loc="best"`` search does and keeps the
    first whose box touches no line, marker, filled area, image or text.
    When every location covers data (filled contours always do), the legend
    goes in one row above the axes, where constrained layout makes room for
    it.

    ```{versionadded} 1.14.0
    ```
    """
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    else:
        handles = list(handles)
        labels = [h.get_label() for h in handles] if labels is None else list(labels)
    old = ax.get_legend()
    if old is not None:
        for handle, text in zip(old.legend_handles, old.get_texts(), strict=True):
            if text.get_text() not in labels:
                handles.append(handle)
                labels.append(text.get_text())
    fig = ax.figure
    for loc in _INSIDE:
        legend = ax.legend(handles, labels, loc=loc, **kwargs)
        fig.draw_without_rendering()
        renderer = fig.canvas.get_renderer()
        if not _overlaps_data(ax, legend.get_window_extent(renderer), renderer):
            return legend
        legend.remove()
    kwargs.setdefault("ncols", max(len(labels), 1))
    return ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0.0,
        **kwargs,
    )


# --- Axes --------------------------------------------------------------------


def _axes(ax, axis: str):
    return {"x": [ax.xaxis], "y": [ax.yaxis], "both": [ax.xaxis, ax.yaxis]}[axis]


def sci_axis(ax: Axes, axis: str = "y") -> None:
    """Plain numbers from 0.1 to 1000, powers of ten outside, never an offset.

    Follows NIST SP 811's advice on numerical values (see
    :data:`SI_POWERLIMITS`); the power is written with mathtext.

    ```{versionadded} 1.14.0
    ```
    """
    for a in _axes(ax, axis):
        fmt = ScalarFormatter(useMathText=True, useOffset=False)
        fmt.set_powerlimits(SI_POWERLIMITS)
        a.set_major_formatter(fmt)


def relative_error(value: ArrayLike, reference: ArrayLike) -> np.ndarray:
    """``|value - reference| / |reference|``, floored at machine epsilon.

    Errors below the working precision are not resolvable and would vanish
    from a log axis; they are shown at epsilon of the result dtype instead.

    ```{versionadded} 1.14.0
    ```
    """
    value = np.asarray(value, dtype=float)
    reference = np.asarray(reference, dtype=float)
    err = np.abs(value - reference) / np.abs(reference)
    return np.maximum(err, np.finfo(err.dtype).eps)


def log_error_axis(ax: Axes, axis: str = "y") -> None:
    """Log axis for errors: labelled decades as powers of ten, unlabelled minors.

    Non-positive values are clipped; use :func:`relative_error` to keep
    exact zeros visible at machine epsilon.

    ```{versionadded} 1.14.0
    ```
    """
    decade = 10
    for a in _axes(ax, axis):
        (ax.set_xscale if a is ax.xaxis else ax.set_yscale)("log", nonpositive="clip")
        a.set_major_locator(LogLocator(base=decade))
        a.set_major_formatter(LogFormatterSciNotation(base=decade))
        a.set_minor_locator(LogLocator(base=decade, subs=np.arange(2, decade)))
        a.set_minor_formatter(NullFormatter())
