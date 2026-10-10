# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Figures for surrogate-assisted saddle searches.

Every function takes the records of :mod:`chemparseplot.parse.surrogate.model`
and returns a :class:`matplotlib.figure.Figure`; none reads a file or knows the
code that produced the data. Colour is never the only channel: each series
also differs in marker or line style.

Encodings shared by the figures:

- teal, circles, solid: the surrogate (posterior mean, predicted force)
- magenta, squares: the true potential (oracle evaluations, true force)
- sky, triangles, dashed: a reference (true profile, climbing-image force)
- coral ticks: decisions to call the oracle
- dashed black: the force tolerance

.. versionadded:: 1.10.0
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator

from chemparseplot.parse.surrogate.cases import CaseBoard
from chemparseplot.parse.surrogate.model import (
    BandHistory,
    BandSnapshot,
    CampaignTable,
    ComponentTable,
    PopTable,
    ScalingTable,
    SearchHistory,
    SingleEndedHistory,
    SurrogateSearch,
)
from chemparseplot.plot.structs import (
    convert_energy,
    eigenvalue_axis_label,
    energy_axis_label,
)
from chemparseplot.plot.theme import RUHI_COLORS, get_theme, setup_publication_theme

SURROGATE = RUHI_COLORS["teal"]
ORACLE = RUHI_COLORS["magenta"]
REFERENCE = RUHI_COLORS["sky"]
ACQUISITION = RUHI_COLORS["coral"]
HIGHLIGHT = RUHI_COLORS["sunshine"]
NEUTRAL = "#6b6b6b"

_FORCE_LABEL = r"max atomic force (eV/$\mathrm{\AA}$)"
_MAX_EVOLUTION_PANELS = 8
_MIN_SNAPSHOTS = 2
_WALL_INT_MIN = 10
_TWO = 2
# Bars at least this many times the axis start hold their value inside.
_WALL_INSIDE_MIN_RATIO = 3
# Below this deviation/progress span ratio the landscape would be a sliver.
_EQUAL_ASPECT_MIN = 0.35
_EFFICIENCY_DARK = 0.45
_FORCE_LABEL_MIN_ROWS = 8
_CI_ARMS = {"current_ci", "ci"}


_FONT: list[str | None] = [None]
_LEGEND_PT: list[float | None] = [None]
FONT_DIRS_ENV = "CHEMPARSEPLOT_FONT_DIRS"


def _font_search_dirs(extra) -> list[Path]:
    env = [Path(d) for d in os.environ.get(FONT_DIRS_ENV, "").split(os.pathsep) if d]
    return [
        *(Path(d) for d in extra),
        *env,
        Path.home() / ".local/share/fonts",
        Path("/usr/share/fonts"),
    ]


def _fontconfig_faces(family: str) -> list[str]:
    """Files fontconfig lists for ``family``; empty when fc-list is absent."""
    exe = shutil.which("fc-list")
    if exe is None:
        return []
    try:
        out = subprocess.run(  # noqa: S603
            [exe, f":family={family}", "file"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [
        ln.strip().rstrip(":").strip()
        for ln in out.splitlines()
        if ln.strip().lower().rstrip(":").endswith((".ttf", ".otf"))
    ]


def set_font(family: str | None, font_dirs: Sequence[str | Path] = ()) -> None:
    """Draw every figure of this module in ``family`` (None: the theme font).

    Faces named ``<family>*.ttf`` or ``.otf`` (spaces ignored, any case) are
    searched, after fontconfig (``fc-list``), recursively in ``font_dirs``,
    ``$CHEMPARSEPLOT_FONT_DIRS``,
    ``~/.local/share/fonts`` and ``/usr/share/fonts`` and registered with
    matplotlib. Text and math text use the family. A family that still cannot
    be resolved raises ``ValueError`` naming the directories searched; there is
    no fallback.
    """
    if not family:
        _FONT[0] = None
        return
    dirs = _font_search_dirs(font_dirs)
    stem = family.replace(" ", "").lower()
    for face in _fontconfig_faces(family):
        font_manager.fontManager.addfont(face)
    for d in dirs:
        if d.is_dir():
            for face in sorted(d.rglob("*")):
                if face.suffix.lower() in {".ttf", ".otf"} and face.stem.lower().replace(
                    " ", ""
                ).startswith(stem):
                    font_manager.fontManager.addfont(str(face))
    try:
        font_manager.findfont(family, fallback_to_default=False)
    except ValueError as exc:
        msg = (
            f"font family {family!r} is not available to matplotlib; searched "
            f"fontconfig and {[str(d) for d in dirs]}. Install it or add its "
            "directory with "
            f"--font-dir or ${FONT_DIRS_ENV}."
        )
        raise ValueError(msg) from exc
    _FONT[0] = family


def set_legend_fontsize(pt: float | None) -> None:
    """Set the legend font size (points) of every figure of this module.

    ``None`` keeps each figure's own size (8 or 9 pt). Layouts that size
    themselves around the legend (the landscape and the profile with a strip)
    make room for the larger text.
    """
    _LEGEND_PT[0] = pt


def _lfs(default: float) -> float:
    """Legend font size: the module setting, else the figure's own ``default``."""
    return _LEGEND_PT[0] or default


def _measured_legend_height(handles, labels, fontsize, ncols, fig_w):
    """Height (inches) matplotlib gives a figure legend with these entries."""
    tmp = plt.figure(figsize=(fig_w, 2.0))
    try:
        leg = tmp.legend(
            handles,
            labels,
            frameon=False,
            fontsize=fontsize,
            ncols=ncols,
            loc="lower center",
        )
        tmp.canvas.draw()
        return leg.get_window_extent().height / tmp.dpi
    finally:
        plt.close(tmp)


def _wrap_handle_labels(handles, fontsize, fig_w, handle_in=0.55):
    """Re-wrap legend labels so two columns fit the figure width."""
    import textwrap  # noqa: PLC0415

    chars = max(12, int((fig_w / 2 - handle_in - 0.1) * 72 / (fontsize * 0.52)))
    for h in handles:
        text = " ".join(h.get_label().split())
        h.set_label(textwrap.fill(text, chars))


def _legend_layout(labels, fontsize, fig_w, handle_in=0.55, max_cols=2):
    """Columns and height (inches) for a figure legend of ``labels``.

    Two columns when two columns of the longest label fit the figure width,
    else one; the height follows the number of text lines at ``fontsize``.
    """
    width = max(max(len(ln) for ln in lab.split("\n")) for lab in labels)
    col_w = handle_in + width * fontsize * 0.52 / 72
    ncols = _TWO if (max_cols >= _TWO and 2 * col_w <= fig_w) else 1
    rows = -(-len(labels) // ncols)
    per_row = []
    for r in range(rows):
        # matplotlib fills columns first: row r holds labels[r], labels[rows + r]
        chunk = [labels[r]] + (
            [labels[rows + r]] if ncols == _TWO and rows + r < len(labels) else []
        )
        per_row.append(max(lab.count("\n") + 1 for lab in chunk))
    line_in = fontsize * 1.2 / 72  # one text line
    gap_in = fontsize * 0.5 / 72  # matplotlib's default label spacing
    height = sum(per_row) * line_in + rows * gap_in + 0.15
    return ncols, height


def _style() -> None:
    setup_publication_theme(get_theme("ruhi"))
    if _FONT[0]:
        fam = _FONT[0]
        plt.rcParams.update(
            {
                "font.family": fam,
                "mathtext.fontset": "custom",
                "mathtext.rm": fam,
                "mathtext.it": f"{fam}:italic",
                "mathtext.bf": f"{fam}:bold",
                "mathtext.default": "regular",
            }
        )
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path"})


def _rel_energy(snap: BandSnapshot, values: np.ndarray, unit: str) -> np.ndarray:
    e0 = snap.energy[0]
    return convert_energy(np.asarray(values, dtype=float) - e0, unit)


def _plain_log_axis(axis) -> None:
    """Label a log axis with plain numbers at 1, 2 and 5 per decade."""
    axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(FuncFormatter(lambda _v, _p: ""))


def _event_group(kind: str) -> str:
    if kind in _CI_ARMS:
        return "climbing-image acquisition"
    if kind == "measured-step":
        return "measured step"
    if kind == "ceiling":
        return "observation dropped at cap"
    if kind == "spectrum":
        return "spectrum batch"
    return "band acquisition"


_GROUP_MARKER = {
    "climbing-image acquisition": ("|", ACQUISITION),
    "band acquisition": ("|", NEUTRAL),
    "measured step": ("x", ORACLE),
    "observation dropped at cap": ("v", HIGHLIGHT),
    "spectrum batch": ("|", ACQUISITION),
}


def _rug(ax, events, xkey: str, y: float, *, transform=None) -> list[Line2D]:
    """Draw decision marks below ``ax``, one row per kind; return legend handles."""
    handles = []
    groups: dict[str, list[float]] = {}
    for e in events:
        x = e.oracle_calls if xkey == "oracle_calls" else e.outer
        if x is None or x < 0:
            continue
        groups.setdefault(_event_group(e.kind), []).append(x)
    for k, g in enumerate(g for g in _GROUP_MARKER if g in groups):
        xs = groups[g]
        marker, color = _GROUP_MARKER[g]
        ax.plot(
            xs,
            [y - 0.045 * k] * len(xs),
            ls="none",
            marker=marker,
            ms=7,
            mew=1.2,
            color=color,
            transform=transform or ax.get_xaxis_transform(),
            clip_on=False,
        )
        handles.append(
            Line2D([], [], ls="none", marker=marker, ms=7, mew=1.2, color=color, label=g)
        )
    return handles


# ---------------------------------------------------------------- (a) profile


def plot_band_profile(
    band: BandHistory,
    *,
    snapshot: BandSnapshot | None = None,
    energy_unit: str = "eV",
    sigma_scale: float = 2.0,
    show_points: bool = True,
    observations: str = "fade",
    observation_distance: float = 0.1,
    structures: str | None = None,
    label_critical_points: bool | None = None,
    n_structures: int | None = None,
    strip_renderer: str = "xyzrender",
    rotation: str = "auto",
    xyzrender_config: str = "paton",
    ax=None,
    title: str | None = None,
) -> Figure:
    """Energy along the path: surrogate mean, uncertainty and true evaluations.

    The curve is the posterior mean at the images; the ribbon is
    ``+/- sigma_scale`` predictive standard deviations when the producer
    recorded them. Magenta squares are images whose true energy is known,
    diamonds the oracle evaluations projected on the path, the dashed blue
    curve the true profile when one was recomputed on the final path.

    An evaluation off the final band is not an error of the mean, so
    ``observations`` says how distance is shown: ``"fade"`` colours each
    diamond by its Cartesian distance from the final path (colourbar, dark is
    near); ``"near"`` draws only those within ``observation_distance`` (A) and
    says in the legend how many farther ones were omitted; ``"none"`` (or
    ``show_points=False``) draws none.

    ``label_critical_points`` puts the letters R, SP (or CI) and P at their points
    on the curve; it is on whenever a strip is drawn and selectable without one
    (the structures can then be shown elsewhere, e.g. under the landscape).

    ``structures`` (``"crit_points"`` or ``"all"``, ``n_structures`` for evenly
    spaced images) adds the strip of structures that ``rgpycrumbs eon plt-neb``
    draws under its profile, captioned R, SP (or CI) and P and tied to their
    points on the curve; it needs ``band.numbers`` (atom types) and raises
    ``ValueError`` naming the files looked at otherwise.
    """
    _style()
    snap = snapshot or band.final
    own_axes = ax is None
    entries = []
    if structures not in (None, "none"):
        if not own_axes:
            msg = "a structure strip needs the function to create its own figure"
            raise ValueError(msg)

        def locate(_pos, image):
            return snap.coordinate[image], _rel_energy(
                snap, snap.energy[image : image + 1], energy_unit
            )[0]

        entries = _strip_entries(band, structures, n_structures, locate)
    if own_axes and entries:
        n_rows_strip = -(-len(entries) // _STRIP_MAX_COLS)
        strip_h = _STRIP_ROW_IN * n_rows_strip + (0.3 if n_rows_strip > 1 else 0.0)
        plot_w, plot_h = 5.2, 3.4
        fig_w = _Y_LABEL_IN + plot_w + _CBAR_IN * 0.7
        fig_h = _TOP_IN + plot_h + 0.7 + strip_h + _GAP_IN + _LEGEND_IN
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes((0.1, 0.5, 0.5, 0.4))
        ax_strip = fig.add_axes((0.1, 0.1, 0.5, 0.1))
    elif own_axes:
        fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    else:
        fig = ax.figure
    x = snap.coordinate
    y = _rel_energy(snap, snap.energy, energy_unit)
    if snap.sigma is not None and np.isfinite(snap.sigma).any():
        lo = _rel_energy(snap, snap.energy - sigma_scale * snap.sigma, energy_unit)
        hi = _rel_energy(snap, snap.energy + sigma_scale * snap.sigma, energy_unit)
        ax.fill_between(
            x,
            lo,
            hi,
            color=SURROGATE,
            alpha=0.18,
            lw=0,
            label=rf"$\pm${sigma_scale:g}$\sigma$",
        )
    ax.plot(x, y, "-o", color=SURROGATE, ms=4, label="surrogate mean")
    ymin, ymax = y.min(), y.max()
    if band.reference is not None:
        r = band.reference
        ax.plot(
            r.coordinate,
            _rel_energy(snap, r.energy, energy_unit),
            "--^",
            color=REFERENCE,
            ms=4,
            lw=1.4,
            label="true potential on path",
        )
    ev = snap.evaluated
    if ev.any():
        ye = _rel_energy(snap, snap.true_energy[ev], energy_unit)
        ax.plot(
            x[ev],
            ye,
            ls="none",
            marker="s",
            ms=7,
            mfc="none",
            mec=ORACLE,
            mew=1.6,
            label="true energy at image",
        )
        ymin, ymax = min(ymin, ye.min()), max(ymax, ye.max())
    if snap.climbing is not None and 0 <= snap.climbing < len(x):
        ax.plot(
            x[snap.climbing],
            y[snap.climbing],
            ls="none",
            marker="o",
            ms=13,
            mfc="none",
            mec="black",
            mew=1.2,
            label="climbing image",
        )
    pad = 0.35 * (ymax - ymin or 1.0)
    lo_v, hi_v = ymin - pad, ymax + pad
    if not show_points:
        observations = "none"
    pts = band.points
    if (
        observations != "none"
        and pts is not None
        and pts.coordinate is not None
        and pts.distance is not None
    ):
        yp = convert_energy(pts.energy - snap.energy[0], energy_unit)
        keep = (yp >= lo_v) & (yp <= hi_v)
        omitted = 0
        if observations == "near":
            far = pts.distance > observation_distance
            omitted = int((far & keep).sum())
            keep &= ~far
        label = f"oracle evaluations ({int(keep.sum())}), projected onto the final path"
        if omitted:
            label += f", {omitted} farther than {observation_distance:g} \u00c5 omitted"
        if observations == "fade":
            cmap = LinearSegmentedColormap.from_list("obsfade", ["#1a1a1a", "#dcdcdc"])
            sc = ax.scatter(
                pts.coordinate[keep],
                yp[keep],
                c=pts.distance[keep],
                cmap=cmap,
                vmin=0,
                vmax=max(float(pts.distance[keep].max()), 1e-9) if keep.any() else 1,
                marker="D",
                s=14,
                label=label,
                zorder=2,
            )
            fig.colorbar(
                sc,
                ax=ax,
                pad=0.02,
                label=r"distance from the final path ($\mathrm{\AA}$)",
            )
        else:
            ax.plot(
                pts.coordinate[keep],
                yp[keep],
                ls="none",
                marker="D",
                ms=3.5,
                color=NEUTRAL,
                alpha=0.8,
                label=label,
            )
    ax.set_ylim(lo_v, hi_v)
    ax.set_xlabel(r"path coordinate ($\mathrm{\AA}$)")
    ax.set_ylabel(energy_axis_label(energy_unit, label="energy relative to reactant"))
    if title:
        ax.set_title(title)
    if label_critical_points and not entries:
        crit = _critical_labels(band, snap)
        for lab, x, y in crit:
            ax.annotate(
                lab,
                (x, _rel_energy(snap, np.array([y]), energy_unit)[0]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                zorder=102,
                annotation_clip=False,
                path_effects=[mpe.withStroke(linewidth=2.5, foreground="white")],
            )
        ax.plot([], [], ls="none", label=_critical_legend(crit, "")[0])
    if entries:
        for ent in entries:
            if ent.label in {"R", "SP", "CI", "P"}:
                ax.annotate(
                    ent.label,
                    (ent.x, ent.y),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    zorder=102,
                    annotation_clip=False,
                    path_effects=[mpe.withStroke(linewidth=2.5, foreground="white")],
                )
        shown = [e.label for e in entries if e.label in {"R", "SP", "CI", "P"}]
        meaning = {
            "R": "reactant",
            "SP": "saddle",
            "CI": "climbing image",
            "P": "product",
        }
        h, lb = ax.get_legend_handles_labels()
        h.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="",
                label=(
                    ", ".join(shown)
                    + ": "
                    + ", ".join(meaning[x] for x in shown)
                    + " (structures below)"
                ),
            )
        )
        lb.append(h[-1].get_label())
        legend_in = 0.1 + _measured_legend_height(h, lb, _lfs(9), 1, fig_w)
        fig_h = _TOP_IN + plot_h + 0.7 + strip_h + _GAP_IN + legend_in
        fig.set_size_inches(fig_w, fig_h, forward=True)
        left = _Y_LABEL_IN / fig_w
        plot_bottom = (_GAP_IN + legend_in + strip_h + 0.7) / fig_h
        ax.set_position([left, plot_bottom, plot_w / fig_w, plot_h / fig_h])
        for other in list(fig.axes):
            if other is ax or other is ax_strip:
                continue
            other.set_position(
                [
                    left + plot_w / fig_w + 0.012,
                    plot_bottom,
                    max(0.015, 0.2 / fig_w),
                    plot_h / fig_h,
                ]
            )
        strip_pos = [
            left,
            (_GAP_IN + legend_in) / fig_h,
            plot_w / fig_w,
            strip_h / fig_h,
        ]
        ax_strip.set_position(strip_pos)
        ax_strip.axis("off")
        fig.canvas.draw()
        _neb_plot().plot_structure_strip(
            ax_strip,
            entries,
            zoom=_STRIP_ZOOM,
            rotation=rotation,
            theme_color="black",
            renderer=strip_renderer,
            xyzrender_config=xyzrender_config,
            col_spacing=_STRIP_SPACING,
            width_fill_fraction=_STRIP_FILL,
            max_cols=_STRIP_MAX_COLS,
            prefer_single_row=False,
        )
        ax_strip.set_position(strip_pos)
        fig.legend(
            h,
            lb,
            frameon=False,
            fontsize=_lfs(9),
            ncols=1,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
        )
    elif own_axes:
        # Below the axes: the observation label is long and must not cover data.
        fig.legend(frameon=False, fontsize=_lfs(9), ncols=2, loc="outside lower center")
    else:
        ax.legend(frameon=False, fontsize=_lfs(9), loc="best")
    return fig


# ------------------------------------------------------------- (b) evolution


def plot_band_evolution(
    band: BandHistory,
    *,
    energy_unit: str = "eV",
    max_panels: int = _MAX_EVOLUTION_PANELS,
) -> Figure:
    """Small multiples of the band over outer iterations.

    With two or more snapshots, one profile panel per snapshot (evenly spaced
    in oracle calls, at most ``max_panels``) on shared axes. A producer that
    stores only the final band gets the acquisition record instead: outer
    iteration against the image the oracle was called on, one marker shape per
    kind of decision.
    """
    _style()
    snaps = band.snapshots
    if len(snaps) >= _MIN_SNAPSHOTS:
        pick = np.unique(
            np.linspace(0, len(snaps) - 1, min(max_panels, len(snaps)))
            .round()
            .astype(int)
        )
        n = len(pick)
        cols = min(4, n)
        rows = -(-n // cols)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(2.6 * cols, 2.2 * rows),
            sharex=True,
            sharey=True,
            layout="constrained",
            squeeze=False,
        )
        for ax, k in zip(axes.ravel(), pick, strict=False):
            s = snaps[k]
            plot_band_profile(
                BandHistory(final=s, reference=band.reference),
                energy_unit=energy_unit,
                show_points=False,
                ax=ax,
            )
            leg = ax.get_legend()
            if leg:
                leg.remove()
            ax.set_title(f"outer {s.outer}, {s.oracle_calls} calls", fontsize=9)
            ax.set_xlabel("")
            ax.set_ylabel("")
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        fig.supxlabel(r"path coordinate ($\mathrm{\AA}$)")
        fig.supylabel(energy_axis_label(energy_unit, label="energy relative to reactant"))
        return fig
    fig, ax = plt.subplots(figsize=(7.0, 3.6), layout="constrained")
    markers = {
        "climbing-image acquisition": ("D", ACQUISITION),
        "band acquisition": ("o", NEUTRAL),
        "measured step": ("x", ORACLE),
        "observation dropped at cap": ("v", HIGHLIGHT),
    }
    for g, (mk, col) in markers.items():
        pts = [
            (e.outer, e.image)
            for e in band.events
            if _event_group(e.kind) == g and e.image >= 0
        ]
        if not pts:
            continue
        xs, ys = zip(*pts, strict=True)
        ax.plot(
            xs,
            ys,
            ls="none",
            marker=mk,
            ms=5,
            color=col,
            mfc="none" if mk == "o" else col,
            label=g,
        )
    n_img = band.final.n_images
    ax.set_ylim(-0.5, n_img - 0.5)
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("image the oracle was called on")
    if band.final.climbing is not None:
        ax.axhline(band.final.climbing, color=ACQUISITION, lw=0.8, ls=":")
    fig.legend(frameon=False, fontsize=_lfs(8), ncols=2, loc="outside lower center")
    return fig


# ----------------------------------------------------------- (d) reduced landscape


# Layout and rendering constants of the structure strip, as in rgpycrumbs eon plt-neb.
_STRIP_MAX_COLS = 6
_STRIP_ZOOM = 0.5 * 3.15
_STRIP_SPACING = 1.5
_STRIP_FILL = 0.92
_STRIP_ROW_IN = 1.6
_MAP_IN, _Y_LABEL_IN, _CBAR_IN = 5.2, 0.95, 1.2
_TOP_IN, _GAP_IN, _LEGEND_IN = 0.4, 0.1, 1.25
_XLABEL_GAP_IN = 0.5  # room for the x label between the map and the strip
_SAME_GEOMETRY = 1e-3


def _neb_plot():
    from chemparseplot.plot import neb as neb_plot  # noqa: PLC0415

    return neb_plot


def _strip_indices(n_images, ci, n_structures):
    """Band images for the strip: R, the climbing image, P and, with
    ``n_structures``, evenly spaced images filling up to that many."""
    chosen = {0, n_images - 1, ci}
    if n_structures:
        for cand in np.unique(
            np.linspace(0, n_images - 1, n_structures).round().astype(int)
        ):
            if len(chosen) >= n_structures:
                break
            chosen.add(int(cand))
    return sorted(chosen)


def _strip_entries(band, structures, n_structures, locate):
    """StructurePlacement entries for the strip.

    ``locate(positions, image_index_or_None)`` returns the (x, y) of a structure
    on the panel the strip belongs to.
    """
    from ase import Atoms  # noqa: PLC0415

    from chemparseplot.plot.structs import StructurePlacement  # noqa: PLC0415

    final = band.final
    if band.numbers is None:
        msg = (
            "no atom types for the structure strip: looked for "
            f"{band.numbers_looked_for or 'a types file'}; pass a geometry file with "
            "the same atom order (--types-from FILE) or draw without the strip "
            "(--plot-structures none)"
        )
        raise ValueError(msg)
    n_images = final.n_images
    ci = final.climbing if final.climbing is not None else int(np.argmax(final.energy))
    if structures == "all":
        idx = list(range(n_images))
    elif structures == "crit_points":
        idx = _strip_indices(n_images, ci, n_structures)
    else:
        msg = f"unknown structures {structures!r}; use 'crit_points', 'all' or None"
        raise ValueError(msg)

    def place(pos, label, image=None):
        x, y = locate(pos, image)
        atoms = Atoms(numbers=band.numbers, positions=np.asarray(pos).reshape(-1, 3))
        return StructurePlacement(atoms=atoms, x=float(x), y=float(y), label=label)

    out = []
    saddle_is_ci = (
        band.saddle is not None
        and np.linalg.norm(band.saddle - final.positions[ci]) < _SAME_GEOMETRY
    )
    for i in idx:
        if i == 0:
            label = "R"
        elif i == n_images - 1:
            label = "P"
        elif i == ci:
            label = "SP" if (band.saddle is None or saddle_is_ci) else "CI"
        else:
            label = str(i)
        out.append(place(final.positions[i], label, i))
    if band.saddle is not None and not saddle_is_ci:
        out.append(place(band.saddle, "SP", ci))
    out.sort(key=lambda e: e.x)
    return out


def _critical_labels(band, snap):
    """(label, path coordinate, energy) of R, the saddle (or climbing image) and P."""
    n = snap.n_images
    ci = snap.climbing if snap.climbing is not None else int(np.argmax(snap.energy))
    saddle_is_ci = (
        band.saddle is None
        or snap.positions is None
        or np.linalg.norm(band.saddle - snap.positions[ci]) < _SAME_GEOMETRY
    )
    out = [("R", snap.coordinate[0], snap.energy[0])]
    out.append(
        (
            "SP" if (band.saddle is None or saddle_is_ci) else "CI",
            snap.coordinate[ci],
            snap.energy[ci],
        )
    )
    if not saddle_is_ci:
        out.append(("SP", snap.coordinate[ci], snap.energy[ci]))
    out.append(("P", snap.coordinate[n - 1], snap.energy[n - 1]))
    return out


def _critical_legend(crit, tail):
    shown = [lab for lab, _x, _y in crit]
    meaning = {"R": "reactant", "SP": "saddle", "CI": "climbing image", "P": "product"}
    return [", ".join(shown) + ": " + ", ".join(meaning[x] for x in shown) + tail]


def _fit_window(surf, fade_variance, xs, ys, margin=0.06):
    """Window around the unfaded surface and all the plotted points.

    The box is widened by ``margin`` of its larger side; every given point is
    inside it.
    """
    x = np.concatenate([np.asarray(v, dtype=float).ravel() for v in xs])
    y = np.concatenate([np.asarray(v, dtype=float).ravel() for v in ys])
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    if surf is not None and surf.relative_variance is not None:
        keep = (
            np.ones_like(surf.relative_variance, dtype=bool)
            if fade_variance is None
            else surf.relative_variance <= fade_variance
        )
        if keep.any():
            x0, x1 = min(x0, surf.x[keep].min()), max(x1, surf.x[keep].max())
            y0, y1 = min(y0, surf.y[keep].min()), max(y1, surf.y[keep].max())
    pad = margin * max(x1 - x0, y1 - y0)
    lo_x, hi_x, lo_y, hi_y = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    if surf is not None:
        # Do not open a blank margin beyond the evaluated surface.
        lo_x, hi_x = (
            max(lo_x, min(surf.x.min(), x.min())),
            min(hi_x, max(surf.x.max(), x.max())),
        )
        lo_y, hi_y = (
            max(lo_y, min(surf.y.min(), y.min())),
            min(hi_y, max(surf.y.max(), y.max())),
        )
    return (lo_x, hi_x), (lo_y, hi_y)


def reduced_coordinates(band: BandHistory):
    """(s, d) coordinates of the band and of the oracle evaluations.

    ``a`` and ``b`` are the RMSD (A) of a geometry to the reactant and to the
    product image of the final band; ``s`` and ``d`` are the progress along and
    the deviation from the straight reactant-to-product line in the (a, b)
    plane, the reaction-valley projection of
    :mod:`chemparseplot.parse.projection` that ``rgpycrumbs eon plt-neb`` uses.
    """
    from chemparseplot.parse.projection import (  # noqa: PLC0415
        compute_projection_basis,
        project_to_sd,
    )

    final = band.final
    if final.positions is None:
        msg = "the band holds no geometries"
        raise ValueError(msg)
    n_atoms = final.positions.shape[1] // 3

    def rmsd(x, ref):
        return np.linalg.norm(x - ref, axis=-1) / np.sqrt(n_atoms)

    ra = rmsd(final.positions, final.positions[0])
    rb = rmsd(final.positions, final.positions[-1])
    basis = compute_projection_basis(ra, rb)
    path = project_to_sd(ra, rb, basis)
    pts = None
    if band.points is not None and band.points.positions is not None:
        p = band.points.positions
        pts = project_to_sd(
            rmsd(p, final.positions[0]), rmsd(p, final.positions[-1]), basis
        )
    return path, pts


def rmsd_gradients(positions, grads, ref_a, ref_b):
    """Energy gradients in the (RMSD to a, RMSD to b) plane.

    Solves ``g = E_a * grad(a) + E_b * grad(b)`` in least squares for each
    Cartesian gradient ``g``, with ``a = |x - x_a| / sqrt(N)`` and the same for
    ``b``: the part of the true gradient that lies in the plane the landscape
    is drawn in. Returns ``(E_a, E_b)`` in energy per angstrom of RMSD.
    """
    n = positions.shape[1] // 3
    root_n = np.sqrt(n)
    out = np.zeros((len(positions), 2))
    for i, (x, g) in enumerate(zip(positions, grads, strict=True)):
        cols = []
        for ref in (ref_a, ref_b):
            d = x - ref
            norm = np.linalg.norm(d)
            cols.append(d / (root_n * norm) if norm > 0 else np.zeros_like(d))
        out[i] = np.linalg.lstsq(np.column_stack(cols), g, rcond=1e-6)[0]
    return out[:, 0], out[:, 1]


def _drop_cut_labels(fig, ax) -> None:
    """Remove contour labels whose text is cut by the axes: they explain nothing."""
    fig.canvas.draw()
    box = ax.get_window_extent()
    for txt in list(ax.texts):
        if not txt.get_text().startswith("relative variance"):
            continue
        ext = txt.get_window_extent()
        if (
            ext.x0 < box.x0 - 1
            or ext.x1 > box.x1 + 1
            or ext.y0 < box.y0 - 1
            or ext.y1 > box.y1 + 1
        ):
            txt.remove()


def plot_reduced_landscape(
    band: BandHistory,
    *,
    energy_unit: str = "eV",
    surface: str | None = "grad_matern",
    color_by: str = "energy",
    fade_variance: float | None = 0.95,
    label_every: int | None = None,
    window: str = "square",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    structures: str | None = None,
    label_critical_points: bool | None = None,
    n_structures: int | None = None,
    strip_renderer: str = "xyzrender",
    rotation: str = "auto",
    xyzrender_config: str = "paton",
    ax=None,
    title: str | None = None,
) -> Figure:
    """Reaction-valley landscape of a surrogate-assisted NEB.

    The same plot ``rgpycrumbs eon plt-neb`` draws, through the same functions
    (:func:`chemparseplot.plot.neb.plot_landscape_surface`,
    :func:`~chemparseplot.plot.neb.plot_landscape_path_overlay`,
    :func:`~chemparseplot.plot.neb.mark_saddle_point`): axes are the progress
    ``s`` and the orthogonal deviation ``d`` (RMSD, angstrom) relative to the
    straight reactant-to-product line, with the theme's landscape colour map.

    What this version adds, each with a legend entry:

    - the GP energy surface is a fresh fit to the oracle energies and the
      in-plane gradient components in the reduced (s, d) plane, with dashed
      relative variance contours; it is not the search's own model.
      Where the relative variance exceeds ``fade_variance`` (default 0.95, the
      outermost labelled contour; None keeps it all) the surface is faded to
      the background, so no confident surface is drawn where nothing was
      evaluated. ``surface=None`` omits it;
    - the oracle evaluations as dots, coloured by true energy
      (``color_by="energy"``) or by order of evaluation
      (``"iteration"``); ``label_every=k`` also numbers every k-th one, and the
      legend then says the numbers are the order of evaluation;
    - the final path (its colour is the surrogate mean energy);
    - the climbing image (ringed) and the saddle the search reports (gold
      star), certified when the producer says the cell passed.

    ``label_critical_points`` puts R, SP (or CI) and P on the path when no strip
    is drawn (the structures can be shown elsewhere); with a strip they are
    always labelled.

    ``window`` sets the extent: ``"square"`` (default) is the equal-metric square
    window of ``plt-neb``; ``"fit"`` crops to the bounding box of the part of the
    surface that is not faded (relative variance at most ``fade_variance``) united
    with every oracle evaluation, the final path and the saddle, plus a margin,
    keeping the same angstrom scale on both axes, so the map becomes a rectangle.
    ``xlim`` and ``ylim`` set the window explicitly and win over ``window``.

    ``structures`` adds the strip of structures ``rgpycrumbs eon plt-neb``
    draws under its landscapes (same renderer and layout constants):
    ``"crit_points"`` draws the reactant, the saddle (or the climbing image) and
    the product, ``"all"`` every band image, and ``n_structures=N`` that many
    images evenly spaced along the band, always including those three. Each
    structure is captioned on the strip and, for R, SP/CI and P, at its
    position on the path. The atom types come from ``band.numbers``; without
    them the call raises ``ValueError`` naming the files looked at.

    Needs ``rgpycrumbs`` (surface models, via JAX).
    """
    from chemparseplot.parse.projection import (  # noqa: PLC0415
        compute_projection_basis,
        project_to_sd,
    )
    from chemparseplot.plot import neb as neb_plot  # noqa: PLC0415

    _style()
    if band.points is None or band.points.positions is None:
        msg = "the band holds no oracle-evaluation geometries"
        raise ValueError(msg)
    final = band.final
    (s_p, d_p), (s_o, d_o) = reduced_coordinates(band)
    n_atoms = final.positions.shape[1] // 3
    ref_a, ref_b = final.positions[0], final.positions[-1]
    root_n = np.sqrt(n_atoms)

    def rmsd(x, ref):
        return np.linalg.norm(x - ref, axis=-1) / root_n

    pts = band.points
    z_obs = convert_energy(pts.energy - final.energy[0], energy_unit)
    z_path = convert_energy(final.energy - final.energy[0], energy_unit)
    r_obs, p_obs = rmsd(pts.positions, ref_a), rmsd(pts.positions, ref_b)
    r_path, p_path = rmsd(final.positions, ref_a), rmsd(final.positions, ref_b)
    basis = compute_projection_basis(r_path, p_path)
    theme = get_theme("ruhi")
    own_axes = ax is None
    entries = []
    if structures not in (None, "none"):
        if own_axes is False:
            msg = "a structure strip needs the function to create its own figure"
            raise ValueError(msg)

        def locate(pos, _image):
            sd = project_to_sd(
                np.array([rmsd(pos, ref_a)]), np.array([rmsd(pos, ref_b)]), basis
            )
            return sd[0][0], sd[1][0]

        entries = _strip_entries(band, structures, n_structures, locate)
    ax_strip = None
    if own_axes:
        n_rows_strip = -(-len(entries) // _STRIP_MAX_COLS)
        strip_h = (
            _STRIP_ROW_IN * n_rows_strip + (0.3 if n_rows_strip > 1 else 0.0)
            if entries
            else 0.0
        )
        fig_w = _Y_LABEL_IN + _MAP_IN + _CBAR_IN
        fig_h = _TOP_IN + _MAP_IN + _XLABEL_GAP_IN + strip_h + _GAP_IN + _LEGEND_IN
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes((0.1, 0.5, 0.5, 0.4))
        if entries:
            ax_strip = fig.add_axes((0.1, 0.1, 0.5, 0.1))
    else:
        fig = ax.figure
    half = neb_plot.landscape_half_span(
        (min(s_p.min(), s_o.min()), max(s_p.max(), s_o.max())),
        r_path,
        p_path,
        [],
        basis,
    )
    s_mid = 0.5 * (min(s_p.min(), s_o.min()) + max(s_p.max(), s_o.max()))
    handles = []
    saddle_xy = None
    surf = None
    if surface and pts.gradients is not None:
        gr, gp = rmsd_gradients(pts.positions, pts.gradients, ref_a, ref_b)
        surf = neb_plot.plot_landscape_surface(
            ax,
            r_obs,
            p_obs,
            convert_energy(gr, energy_unit),
            convert_energy(gp, energy_unit),
            z_obs,
            method=surface,
            cmap=theme.cmap_landscape,
            show_pts=False,
            project_path=True,
            xlim=(s_mid - half, s_mid + half),
            ylim=(-half, half),
            basis=basis,
            variance_label=lambda x: f"relative variance {x:.2g}",
            fade_variance_above=fade_variance,
        )
        handles.append(
            Patch(
                facecolor="#9aa7b0",
                alpha=0.6,
                label=(
                    "energy surface: GP fitted afresh to the oracle energies and "
                    "in-plane gradients, not the search's model"
                ),
            )
        )
        handles.append(
            Line2D(
                [],
                [],
                color="black",
                ls="--",
                lw=1.0,
                label=("relative variance contours\n(0 at the data, 1 far from it)"),
            )
        )
        if fade_variance is not None:
            handles.append(
                Patch(
                    facecolor="white",
                    edgecolor="#999999",
                    label=(
                        f"faded: relative variance above {fade_variance:g},\n"
                        "no oracle evaluation nearby"
                    ),
                )
            )
    neb_plot.plot_landscape_path_overlay(
        ax,
        r_path,
        p_path,
        z_path,
        theme.cmap_landscape,
        energy_axis_label(energy_unit, label="energy relative to reactant"),
        project_path=True,
        basis=basis,
    )
    handles.append(
        Line2D(
            [],
            [],
            color="black",
            lw=3,
            marker="o",
            ms=5,
            mfc="white",
            label="final path (coloured by surrogate energy)",
        )
    )
    if color_by == "iteration":
        c, cmap, cbar_label = (
            np.arange(1, len(z_obs) + 1),
            plt.get_cmap("Greys"),
            "order of evaluation",
        )
        sc = ax.scatter(
            s_o, d_o, c=c, cmap=cmap, s=16, edgecolor="black", lw=0.4, zorder=45
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label=cbar_label)
        handles.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=5,
                mfc="#888888",
                mec="black",
                label="oracle evaluations (shade: order of evaluation)",
            )
        )
    else:
        norm = plt.Normalize(z_obs.min(), z_obs.max())
        ax.scatter(
            s_o,
            d_o,
            c=z_obs,
            cmap=theme.cmap_landscape,
            norm=norm,
            s=16,
            edgecolor="black",
            lw=0.5,
            zorder=45,
        )
        handles.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=5,
                mfc="white",
                mec="black",
                label="oracle evaluations (fill: true energy, as the colourbar)",
            )
        )
    if label_every:
        for k in range(0, len(z_obs), label_every):
            ax.annotate(
                str(k + 1),
                (s_o[k], d_o[k]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                zorder=60,
            )
        handles[-1].set_label(handles[-1].get_label() + "; numbers: order of evaluation")
    ci = final.climbing
    if ci is not None:
        ax.plot(
            s_p[ci],
            d_p[ci],
            marker="o",
            ms=30,
            mfc="none",
            mec=ACQUISITION,
            mew=2.0,
            ls="none",
            zorder=70,
        )
        handles.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=11,
                mfc="none",
                mec=ACQUISITION,
                mew=2.0,
                label="climbing image",
            )
        )
    if band.saddle is not None:
        s_sd = project_to_sd(
            np.array([rmsd(band.saddle, ref_a)]),
            np.array([rmsd(band.saddle, ref_b)]),
            basis,
        )
        saddle_xy = (float(s_sd[0][0]), float(s_sd[1][0]))
        neb_plot.mark_saddle_point(ax, s_sd[0][0], s_sd[1][0], annotate=False)
        what = (
            "certified saddle"
            if band.saddle_certified
            else "reported saddle (not certified)"
        )
        handles.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="*",
                ms=13,
                mfc="#FFD700",
                mec="black",
                label=what,
            )
        )
    ax.set_xlabel(r"Reaction progress $s$ ($\AA$)")
    ax.set_ylabel(r"Orthogonal deviation $d$ ($\AA$)")
    win_x, win_y = (s_mid - half, s_mid + half), (-half, half)
    if xlim is not None or ylim is not None:
        win_x, win_y = xlim or win_x, ylim or win_y
    elif window == "fit":
        win_x, win_y = _fit_window(
            surf,
            fade_variance,
            [s_p, s_o, *([np.array([saddle_xy[0]])] if saddle_xy else [])],
            [d_p, d_o, *([np.array([saddle_xy[1]])] if saddle_xy else [])],
        )
    elif window != "square":
        msg = f"unknown window {window!r}; use 'square' or 'fit'"
        raise ValueError(msg)
    map_w_in = _MAP_IN
    map_h_in = max(2.5, _MAP_IN * (win_y[1] - win_y[0]) / (win_x[1] - win_x[0]))
    ax.set_xlim(*win_x)
    ax.set_ylim(*win_y)
    ax.set_aspect("equal", adjustable="box")
    ax.minorticks_on()
    if title:
        ax.set_title(title, loc="left")
    if label_critical_points and not entries:
        crit = _critical_labels(band, final)
        for lab, _x, _y in crit:
            idx = {"R": 0, "P": final.n_images - 1}.get(
                lab, final.climbing if final.climbing is not None else 0
            )
            ax.text(
                s_p[idx],
                d_p[idx],
                lab,
                fontsize=11,
                fontweight="bold",
                color="white",
                ha="center",
                va="bottom",
                zorder=102,
                clip_on=False,
                path_effects=[mpe.withStroke(linewidth=2.5, foreground="black")],
            )
        handles.append(
            Line2D([], [], ls="none", marker="", label=_critical_legend(crit, "")[0])
        )
    if entries:
        for ent in entries:
            if ent.label in {"R", "SP", "CI", "P"}:
                ax.text(
                    ent.x,
                    ent.y,
                    ent.label,
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                    ha="center",
                    va="bottom",
                    zorder=102,
                    clip_on=False,
                    path_effects=[mpe.withStroke(linewidth=2.5, foreground="black")],
                )
        shown = [e.label for e in entries if e.label in {"R", "SP", "CI", "P"}]
        meaning = {
            "R": "reactant",
            "SP": "saddle",
            "CI": "climbing image",
            "P": "product",
        }
        handles.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="",
                label=(
                    ", ".join(shown)
                    + ": "
                    + ", ".join(meaning[x] for x in shown)
                    + " (structures below)"
                ),
            )
        )
    if own_axes:
        _wrap_handle_labels(handles, _lfs(8), fig_w)
        ncols_leg, _ = _legend_layout([h.get_label() for h in handles], _lfs(8), fig_w)
        legend_in = 0.1 + _measured_legend_height(
            handles, [h.get_label() for h in handles], _lfs(8), ncols_leg, fig_w
        )
        fig_h = _TOP_IN + map_h_in + _XLABEL_GAP_IN + strip_h + _GAP_IN + legend_in
        fig.set_size_inches(fig_w, fig_h, forward=True)
        left = _Y_LABEL_IN / fig_w
        map_bottom = (_GAP_IN + legend_in + strip_h + _XLABEL_GAP_IN) / fig_h
        ax.set_position([left, map_bottom, map_w_in / fig_w, map_h_in / fig_h])
        ax.set_aspect("equal", adjustable="box", anchor="C")
        for other in list(fig.axes):
            if other is ax or other is ax_strip:
                continue
            other.set_position(
                [
                    left + map_w_in / fig_w + 0.012,
                    map_bottom,
                    max(0.015, 0.2 / fig_w),
                    map_h_in / fig_h,
                ]
            )
        if entries:
            ax_strip.set_position(
                [left, (_GAP_IN + legend_in) / fig_h, _MAP_IN / fig_w, strip_h / fig_h]
            )
            ax_strip.axis("off")
            fig.canvas.draw()
            neb_plot.plot_structure_strip(
                ax_strip,
                entries,
                zoom=_STRIP_ZOOM,
                rotation=rotation,
                theme_color=theme.textcolor,
                renderer=strip_renderer,
                xyzrender_config=xyzrender_config,
                col_spacing=_STRIP_SPACING,
                width_fill_fraction=_STRIP_FILL,
                max_cols=_STRIP_MAX_COLS,
                prefer_single_row=False,
            )
            ax_strip.set_position(
                [left, (_GAP_IN + legend_in) / fig_h, _MAP_IN / fig_w, strip_h / fig_h]
            )
        fig.legend(
            handles=handles,
            frameon=False,
            fontsize=_lfs(8),
            ncols=ncols_leg,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
        )
    else:
        ax.legend(handles=handles, frameon=False, fontsize=_lfs(8), loc="best")
    _drop_cut_labels(fig, ax)
    return fig


# -------------------------------------------------------------- (c) convergence


def plot_search_convergence(
    search: SearchHistory,
    *,
    x: str = "oracle_calls",
    show_training: bool = True,
    ax=None,
    title: str | None = None,
) -> Figure:
    """Max atomic force against oracle calls (or outer iterations).

    Magenta squares: true force over the evaluated images; teal circles:
    force the surrogate predicts; blue triangles: true force at the climbing
    image. The dashed line is the tolerance; marks below the axis are
    acquisition decisions. A lower panel gives the training-set size, with the
    retention cap when the producer dropped observations.
    """
    _style()
    two = show_training and "n_train" in search.series and ax is None
    if ax is None:
        if two:
            fig, (ax, ax2) = plt.subplots(
                2,
                1,
                figsize=(6.6, 5.2),
                sharex=True,
                height_ratios=(3, 1),
                layout="constrained",
            )
        else:
            fig, ax = plt.subplots(figsize=(6.6, 3.8), layout="constrained")
            ax2 = None
    else:
        fig, ax2 = ax.figure, None
    xs = search.series[x]
    spec = (
        ("surrogate_force", "surrogate prediction", SURROGATE, "-", "o"),
        ("true_force", "true force, evaluated images", ORACLE, "-", "s"),
        ("ci_true_force", "true force, climbing image", REFERENCE, "--", "^"),
    )
    for key, label, color, ls, mk in spec:
        v = search.get(key)
        if v is None or not np.isfinite(v).any():
            continue
        ax.plot(xs, v, ls=ls, marker=mk, ms=3.2, lw=1.3, color=color, label=label)
        if key == "surrogate_force":
            sg = search.get("surrogate_force_sigma")
            if sg is not None:
                ax.fill_between(
                    xs, np.maximum(v - sg, 1e-12), v + sg, color=color, alpha=0.18, lw=0
                )
    if search.force_tolerance:
        ax.axhline(
            search.force_tolerance, color="black", ls="--", lw=1.0, label="tolerance"
        )
    ax.set_yscale("log")
    ax.set_ylabel(_FORCE_LABEL)
    handles, labels = ax.get_legend_handles_labels()
    rug = _rug(ax, search.events, x, -0.04)
    fig.legend(
        handles + rug,
        labels + [h.get_label() for h in rug],
        frameon=False,
        fontsize=_lfs(8),
        ncols=2,
        loc="outside lower center",
    )
    total = search.meta.get("total_calls")
    last = np.nanmax(search.oracle_calls) if "oracle_calls" in search.series else None
    if total and last and total > last and x == "oracle_calls":
        ax.annotate(
            f"counter ends at {int(last)}; the run reports {int(total)} calls",
            xy=(1.0, 1.0),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=8,
            color=ORACLE,
        )
    if title:
        ax.set_title(title, loc="left")
    xlabel = "oracle calls" if x == "oracle_calls" else "outer iteration"
    if two and ax2 is not None:
        ax2.plot(xs, search["n_train"], "-", color=SURROGATE, lw=1.3)
        if search.retained_cap:
            ax2.axhline(search.retained_cap, color=NEUTRAL, ls=":", lw=1.0)
            ax2.annotate(
                "cap",
                xy=(0.01, search.retained_cap),
                xycoords=("axes fraction", "data"),
                fontsize=8,
                va="bottom",
                color=NEUTRAL,
            )
        ax2.set_ylabel("training rows")
        ax2.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)
    return fig


def plot_search_comparison(
    searches: Sequence[SurrogateSearch],
    labels: Sequence[str] | None = None,
    *,
    x: str = "outer",
) -> Figure:
    """Compare searches of the same reaction: forces per outer and a call ledger.

    Top: true force over the evaluated images; middle: true force at the
    climbing image (both log, with the tolerance); bottom: oracle calls by
    phase (endpoints, initial band, outer loop, transition, dimer) as stacked
    bars labelled with the total.
    """
    _style()
    labels = list(labels or [s.label for s in searches])
    styles = [
        ("-", "s", ORACLE),
        ("--", "o", SURROGATE),
        (":", "^", REFERENCE),
        ("-.", "D", ACQUISITION),
    ]
    fig, axes = plt.subplots(
        3, 1, figsize=(6.8, 7.2), height_ratios=(3, 3, 1.6), layout="constrained"
    )
    tol = None
    for s, lab, (ls, mk, col) in zip(searches, labels, styles, strict=False):
        h = s.search
        if h is None:
            continue
        tol = tol or h.force_tolerance
        for ax, key in zip(axes[:2], ("true_force", "ci_true_force"), strict=True):
            v = h.get(key)
            if v is not None:
                ax.plot(h[x], v, ls=ls, marker=mk, ms=2.8, lw=1.2, color=col, label=lab)
    for ax, ttl in zip(axes[:2], ("evaluated images", "climbing image"), strict=True):
        ax.set_yscale("log")
        ax.set_ylabel(_FORCE_LABEL)
        ax.set_title(f"true force, {ttl}", loc="left", fontsize=10)
        if tol:
            ax.axhline(tol, color="black", ls="--", lw=1.0)
    axes[0].legend(frameon=False, fontsize=_lfs(9))
    axes[1].set_xlabel("outer iteration" if x == "outer" else "oracle calls")
    phases = ["endpoints", "initial", "outer", "transition", "dimer", "certificate"]
    pal = [NEUTRAL, REFERENCE, SURROGATE, ACQUISITION, ORACLE, HIGHLIGHT]
    hatches = ["", "//", "", "xx", "..", "\\\\"]
    ax = axes[2]
    for i, s in enumerate(searches):
        led = s.cell.ledger if s.cell else {}
        left = 0
        for ph, col, ht in zip(phases, pal, hatches, strict=True):
            v = led.get(ph, 0)
            if v:
                ax.barh(
                    i,
                    v,
                    left=left,
                    color=col,
                    hatch=ht,
                    edgecolor="white",
                    lw=0.5,
                    label=ph
                    if i == 0 or ph not in ax.get_legend_handles_labels()[1]
                    else None,
                )
                left += v
        ax.text(left, i, f"  {int(led.get('total', left))}", va="center", fontsize=9)
    ax.set_yticks(range(len(searches)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("oracle calls to convergence, by phase")
    h, lb = ax.get_legend_handles_labels()
    ax.legend(
        dict(zip(lb, h, strict=True)).values(),
        dict(zip(lb, h, strict=True)).keys(),
        frameon=False,
        fontsize=_lfs(8),
        ncols=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),
    )
    ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
    return fig


# ------------------------------------------------------------- (e) diagnostics


def plot_model_diagnostics(search: SearchHistory, *, x: str = "outer") -> Figure:
    """Hyperparameters, training-set size and per-outer cost against ``x``.

    Panels appear only for the quantities the producer recorded: kernel
    amplitude and noise variance (log), length-scale extremes (log), training
    rows with the retention cap, and stacked seconds per outer iteration for
    refit, prediction and inner relaxation.
    """
    _style()
    s = search.series
    panels = []
    if "magnitude_sigma2" in s or "noise_sigma2" in s:
        panels.append("kernel")
    if "length_scale_min" in s:
        panels.append("length")
    if "n_train" in s:
        panels.append("train")
    if any(k in s for k in ("time_refit", "time_predict", "time_inner")):
        panels.append("time")
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(6.4, 1.9 * len(panels) + 0.8),
        sharex=True,
        layout="constrained",
        squeeze=False,
    )
    xs = s[x]
    for ax, kind in zip(axes.ravel(), panels, strict=True):
        if kind == "kernel":
            for key, lab, col, ls in (
                ("magnitude_sigma2", r"amplitude $\sigma_f^2$", SURROGATE, "-"),
                ("noise_sigma2", r"noise $\sigma_n^2$", ORACLE, "--"),
            ):
                if key in s and np.isfinite(s[key]).any():
                    ax.plot(
                        xs,
                        np.where(s[key] > 0, s[key], np.nan),
                        ls,
                        color=col,
                        lw=1.3,
                        label=lab,
                    )
            ax.set_yscale("log")
            ax.set_ylabel("kernel variance")
            ax.legend(frameon=False, fontsize=_lfs(8), ncols=2)
        elif kind == "length":
            ax.fill_between(
                xs,
                s["length_scale_min"],
                s["length_scale_max"],
                color=SURROGATE,
                alpha=0.2,
                lw=0,
            )
            ax.plot(
                xs, s["length_scale_min"], "-", color=SURROGATE, lw=1.2, label="shortest"
            )
            ax.plot(
                xs, s["length_scale_max"], "--", color=ORACLE, lw=1.2, label="longest"
            )
            ax.set_yscale("log")
            ax.set_ylabel("length scale")
            ax.legend(frameon=False, fontsize=_lfs(8), ncols=2)
        elif kind == "train":
            ax.plot(xs, s["n_train"], "-", color=SURROGATE, lw=1.3)
            if search.retained_cap:
                ax.axhline(search.retained_cap, color=NEUTRAL, ls=":", lw=1.0)
            ax.set_ylabel("training rows")
        else:
            bottom = np.zeros(len(xs))
            width = 0.8 if x == "outer" else None
            for key, lab, col, ht in (
                ("time_refit", "refit", SURROGATE, ""),
                ("time_predict", "predict", ORACLE, "//"),
                ("time_inner", "inner", REFERENCE, ".."),
            ):
                if key in s:
                    v = np.nan_to_num(s[key])
                    ax.bar(
                        xs,
                        v,
                        bottom=bottom,
                        width=width,
                        color=col,
                        hatch=ht,
                        edgecolor="white",
                        lw=0.3,
                        label=lab,
                    )
                    bottom += v
            ax.set_ylabel("seconds per outer")
            ax.legend(frameon=False, fontsize=_lfs(8), ncols=3)
    axes.ravel()[-1].set_xlabel("outer iteration" if x == "outer" else "oracle calls")
    return fig


# -------------------------------------------------------------- (f) single-ended


def plot_single_ended(
    history: SingleEndedHistory, *, energy_unit: str = "eV", title: str | None = None
) -> Figure:
    """Curvature along the dimer mode and true force against oracle calls.

    Top: surrogate curvature per outer step (teal), the measured curvature as
    open magenta diamonds joined to the surrogate value at the same step by a
    thin segment; coral ticks mark batches of oracle calls spent on a spectrum.
    Bottom: true force (log) against the tolerance.
    """
    _style()
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(6.6, 5.4), sharex=True, height_ratios=(1, 1), layout="constrained"
    )
    x = history.oracle_calls
    cur = convert_energy(history.curvature, energy_unit)
    a1.plot(
        x, cur, "-o", color=SURROGATE, ms=3.2, lw=1.2, label="surrogate mode curvature"
    )
    if history.curvature_measured is not None:
        m = np.isfinite(history.curvature_measured)
        if m.any():
            cm = convert_energy(history.curvature_measured[m], energy_unit)
            a1.plot(
                x[m],
                cm,
                ls="none",
                marker="D",
                ms=7,
                mfc="none",
                mec=ORACLE,
                mew=1.6,
                label="measured curvature",
            )
            if history.curvature_surrogate is not None:
                cs = convert_energy(history.curvature_surrogate[m], energy_unit)
                a1.plot(
                    x[m], cs, ls="none", marker="o", ms=5, color=SURROGATE, mec="black"
                )
                a1.vlines(x[m], cs, cm, color=NEUTRAL, lw=0.9)
    a1.axhline(0, color="black", lw=0.8)
    a1.set_ylabel(eigenvalue_axis_label(energy_unit, label="curvature"))
    for i, e in enumerate(history.escape):
        for a in (a1, a2):
            a.axvline(
                e.oracle_calls,
                color=ACQUISITION,
                lw=0.9,
                ls=":",
                alpha=0.9,
                label="spectrum batch" if (a is a1 and i == 0) else None,
            )
    a1.legend(frameon=False, fontsize=_lfs(8), loc="lower right")
    a2.plot(x, history.force, "-s", color=ORACLE, ms=3.2, lw=1.2, label="true force")
    if history.force_tolerance:
        a2.axhline(
            history.force_tolerance, color="black", ls="--", lw=1.0, label="tolerance"
        )
    a2.set_yscale("log")
    a2.set_ylabel(_FORCE_LABEL)
    a2.set_xlabel("oracle calls")
    a2.legend(frameon=False, fontsize=_lfs(8))
    if title:
        a1.set_title(title, loc="left")
    return fig


# ------------------------------------------------------------------ (g) campaign

_BASE_STYLE = [
    ("s", ORACLE),
    ("^", REFERENCE),
    ("D", ACQUISITION),
    ("v", HIGHLIGHT),
    ("P", NEUTRAL),
]


def plot_campaign_calls(
    table: CampaignTable,
    *,
    calls: str = "search_calls",
    log: bool = True,
    label: str | None = None,
) -> Figure:
    """Oracle calls per reaction, ours against comparison methods (dumbbell).

    One row per reaction, sorted by our calls. Filled teal circle: this
    campaign (a coral X where it did not converge or pass); open markers: each
    comparison method; a grey segment spans the smallest to the largest value.
    """
    _style()
    cells = [c for c in table.cells if getattr(c, calls) is not None]
    cells.sort(key=lambda c: getattr(c, calls))
    n = len(cells)
    fig, ax = plt.subplots(figsize=(6.4, 1.3 + 0.27 * n), layout="constrained")
    y = np.arange(n)
    for i, c in enumerate(cells):
        vals = [getattr(c, calls)] + [
            b[c.label] for b in table.baselines.values() if c.label in b
        ]
        ax.plot([min(vals), max(vals)], [i, i], color="#bdbdbd", lw=1.2, zorder=1)
    for (name, base), (mk, col) in zip(
        table.baselines.items(), _BASE_STYLE, strict=False
    ):
        pts = [(base[c.label], i) for i, c in enumerate(cells) if c.label in base]
        if pts:
            ax.plot(
                *zip(*pts, strict=True),
                ls="none",
                marker=mk,
                ms=6,
                mfc="none",
                mec=col,
                mew=1.5,
                label=name,
                zorder=2,
            )
    ours = [getattr(c, calls) for c in cells]
    ok = np.array([c.passed is not False and c.converged is not False for c in cells])
    ax.plot(
        np.array(ours)[ok],
        y[ok],
        ls="none",
        marker="o",
        ms=7,
        color=SURROGATE,
        label=label or table.name,
        zorder=3,
    )
    if (~ok).any():
        ax.plot(
            np.array(ours)[~ok],
            y[~ok],
            ls="none",
            marker="X",
            ms=9,
            color=ACQUISITION,
            mec="black",
            label="did not pass",
            zorder=3,
        )
    ax.set_yticks(y, [c.label for c in cells], fontsize=8)
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
        _plain_log_axis(ax.xaxis)
    ax.set_xlabel(
        "oracle calls" if calls != "search_calls" else "oracle calls in the search"
    )
    ax.grid(axis="x", color="#e6e6e6", lw=0.6)
    ax.legend(
        frameon=False,
        fontsize=_lfs(8),
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12 if n > _FORCE_LABEL_MIN_ROWS else -0.3),
    )
    return fig


def plot_campaign_matrix(
    table: CampaignTable, *, identity_tolerance: float = 0.05
) -> Figure:
    """Pass matrix: reactions by criteria.

    Filled teal circle: criterion met; coral X: not met; grey dash: not
    recorded. Columns: converged, index one (one imaginary mode at the
    located point), identity (|energy delta| within ``identity_tolerance``
    eV of the reference), passed.
    """
    _style()
    cols = ["converged", "index 1", "identity", "passed"]

    def row(c):
        ident = (
            None if c.energy_delta is None else abs(c.energy_delta) <= identity_tolerance
        )
        idx = None if c.index is None else c.index == 1
        return [c.converged, idx, ident, c.passed]

    n = len(table.cells)
    fig, ax = plt.subplots(figsize=(4.2, 1.1 + 0.25 * n), layout="constrained")
    for i, c in enumerate(table.cells):
        for j, v in enumerate(row(c)):
            if v is None:
                ax.plot(j, i, marker="_", color=NEUTRAL, ms=8)
            elif v:
                ax.plot(j, i, marker="o", color=SURROGATE, ms=7)
            else:
                ax.plot(j, i, marker="X", color=ACQUISITION, mec="black", ms=8)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="left")
    ax.set_yticks(range(n), [c.label for c in table.cells], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.xaxis.tick_top()
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    return fig


def plot_campaign_walls(
    table: CampaignTable, *, stages: Sequence[str] = ("band", "dimer", "validation")
) -> Figure:
    """Wall time per reaction as stacked horizontal bars, one segment per stage."""
    _style()
    cells = sorted(table.cells, key=lambda c: sum(c.wall.get(s, 0.0) for s in stages))
    n = len(cells)
    fig, ax = plt.subplots(figsize=(6.4, 1.3 + 0.27 * n), layout="constrained")
    left = np.zeros(n)
    for stage, col, ht in zip(
        stages,
        (SURROGATE, ORACLE, REFERENCE, ACQUISITION),
        ("", "//", "..", "xx"),
        strict=False,
    ):
        v = np.array([c.wall.get(stage, 0.0) for c in cells])
        if v.any():
            ax.barh(
                np.arange(n),
                v,
                left=left,
                color=col,
                hatch=ht,
                edgecolor="white",
                lw=0.4,
                label=stage,
            )
            left += v
    ax.set_yticks(range(n), [c.label for c in cells], fontsize=8)
    ax.set_xlabel("wall time (s)")
    ax.legend(frameon=False, fontsize=_lfs(8), ncols=len(stages), loc="lower right")
    return fig


# ---------------------------------------------------------------------- (h) scaling


def _layout_axis(table: ScalingTable, name: str):
    """Layout labels, cores and a dodge so layouts of one core count separate."""
    w, _ = table.series[name]
    w = np.asarray(w, dtype=float)
    labels = table.layouts.get(name) or [f"{int(v)}" for v in w]
    dodge = np.ones(len(w))
    for v in np.unique(w):
        idx = np.where(w == v)[0]
        for k, j in enumerate(idx):
            dodge[j] = 1.0 + 0.08 * (k - (len(idx) - 1) / 2)
    return w, labels, dodge


_SCALING_STYLES = [
    ("-", "o", SURROGATE),
    ("--", "s", ORACLE),
    ("-.", "^", REFERENCE),
    (":", "D", ACQUISITION),
]


def _draw_scaling(ax, table: ScalingTable, names, *, ylabel="speedup") -> None:
    allw = np.unique(
        np.concatenate(
            [np.asarray(table.series[n][0], dtype=float) for n in table.series]
        )
    )
    w0 = float(np.min(table.series[table.reference or next(iter(table.series))][0]))
    ax.plot(allw, allw / w0, ":", color="black", lw=1.4, label="ideal")
    for name, (ls, mk, col) in zip(names, _SCALING_STYLES * 4, strict=False):
        w, _labels, dodge = _layout_axis(table, name)
        _, sp = table.speedup(name)
        x = w * dodge
        best = np.array([sp[w == v].max() for v in np.unique(w)])
        ax.plot(np.unique(w), best, ls=ls, color=col, lw=1.0, alpha=0.6)
        rng = table.speedup_range(name)
        if rng is not None:
            ax.errorbar(
                x,
                sp,
                yerr=[sp - rng[0], rng[1] - sp],
                fmt="none",
                ecolor=col,
                elinewidth=1.0,
                capsize=2,
            )
        cap = np.asarray(table.capped.get(name, np.zeros(len(w), bool)))
        ax.plot(x[~cap], sp[~cap], ls="none", marker=mk, color=col, ms=5, label=name)
        if cap.any():
            ax.plot(
                x[cap],
                sp[cap],
                ls="none",
                marker=mk,
                mfc="white",
                mec=col,
                mew=1.4,
                ms=6,
                label=f"{name} (capped)",
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(allw, [f"{int(v)}" for v in allw])
    ax.set_xlabel("cores (ranks x threads)")
    ax.set_ylabel(ylabel)


def plot_scaling(table: ScalingTable, *, ylabel: str = "speedup") -> Figure:
    """Speedup against cores on log axes, with the ideal line.

    Speedup is ``T_ref(w0) / T(w)`` with ``w0`` the first worker count of the
    reference series, or of the series itself when the table names no
    reference. Layouts that share a core count are dodged sideways; one line
    joins the fastest layout per core count. Bars span the slowest to the
    fastest repetition and open markers are runs that hit a cap.
    """
    _style()
    fig, ax = plt.subplots(figsize=(4.8, 4.2), layout="constrained")
    _draw_scaling(ax, table, list(table.series), ylabel=ylabel)
    ax.legend(frameon=False, fontsize=_lfs(9), handlelength=2.4)
    return fig


def plot_strong_scaling(table: ScalingTable, name: str) -> Figure:
    """Four panels for one series (a cell), one tick per ranks x threads layout.

    Wall time with the slowest and fastest repetition; speedup against the
    ideal line; oracle calls of the search; seconds per call. The last two
    show when the search path changes with the layout, which wall time alone
    hides. Open markers are runs that hit a cap.
    """
    _style()
    w, labels, _ = _layout_axis(table, name)
    t = np.asarray(table.series[name][1], dtype=float)
    x = np.arange(len(w))
    cap = np.asarray(table.capped.get(name, np.zeros(len(w), bool)))
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), layout="constrained")
    (a1, a2), (a3, a4) = axes

    def pts(ax, y, color, marker, yerr=None):
        if yerr is not None:
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, capsize=2, lw=1.0)
        ax.plot(x, y, "-", color=color, lw=0.8, alpha=0.5)
        ax.plot(x[~cap], y[~cap], ls="none", marker=marker, color=color, ms=5)
        ax.plot(
            x[cap],
            y[cap],
            ls="none",
            marker=marker,
            mfc="white",
            mec=color,
            mew=1.4,
            ms=6,
        )

    lo = t - np.asarray(table.time_min.get(name, t))
    hi = np.asarray(table.time_max.get(name, t)) - t
    pts(a1, t, SURROGATE, "o", [lo, hi])
    a1.set_yscale("log")
    a1.set_ylabel("wall time (s)")
    _draw_scaling(a2, table, [name])
    a2.legend(frameon=False, fontsize=_lfs(9), handlelength=2.4)
    calls = np.asarray(table.calls.get(name, np.full(len(w), np.nan)), dtype=float)
    pts(a3, calls, REFERENCE, "^")
    a3.set_ylabel("oracle calls in the search")
    pts(a4, t / calls, ACQUISITION, "D")
    a4.set_ylabel("seconds per call")
    for ax in (a3, a4):
        if not np.isfinite(calls).any():
            ax.text(0.5, 0.5, "no call counts", transform=ax.transAxes, ha="center")
    for ax in (a1, a3, a4):
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("ranks x threads")
    fig.suptitle(name, x=0.02, ha="left", fontsize=11)
    return fig


def plot_efficiency_table(table: ScalingTable) -> Figure:
    """Parallel efficiency (speedup per ideal speedup) as an annotated heat table.

    One row per ranks x threads layout, one column per series; a cell without
    that layout stays blank. Efficiency is against each series' first point.
    """
    _style()
    names = list(table.series)
    order: dict[tuple, str] = {}
    for n in names:
        w, labels, _ = _layout_axis(table, n)
        for wv, lab in zip(w, labels, strict=True):
            order[(wv, lab)] = lab
    rows = sorted(order, key=lambda k: (k[0], k[1]))
    grid = np.full((len(rows), len(names)), np.nan)
    for j, n in enumerate(names):
        w, labels, _ = _layout_axis(table, n)
        sp = table.speedup(n)[1]
        w0 = float(np.min(table.series[table.reference or n][0]))
        for wv, lab, s_ in zip(w, labels, sp, strict=True):
            grid[rows.index((wv, lab)), j] = s_ / (wv / w0)
    cmap = LinearSegmentedColormap.from_list(
        "eff", ["#ffffff", RUHI_COLORS["sky"], SURROGATE]
    )
    fig, ax = plt.subplots(
        figsize=(1.6 + 1.0 * len(names), 0.9 + 0.3 * len(rows)), layout="constrained"
    )
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1.0, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(names)):
            if np.isfinite(grid[i, j]):
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if grid[i, j] > _EFFICIENCY_DARK else "black",
                )
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=8)
    ax.set_ylabel("ranks x threads")
    ax.set_title("parallel efficiency", loc="left", fontsize=10)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    return fig


def plot_pop_efficiencies(
    layouts: Sequence[str],
    metrics: dict[str, Sequence[float]],
    *,
    title: str | None = None,
) -> Figure:
    """Fixed-work efficiencies per layout (load balance, communication, ...).

    ``metrics`` maps a name to one value in [0, 1] per layout; each name is a
    line with its own marker so the panel reads without colour.
    """
    _style()
    fig, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")
    x = np.arange(len(layouts))
    for (name, vals), (ls, mk, col) in zip(
        metrics.items(), _SCALING_STYLES * 4, strict=False
    ):
        ax.plot(
            x,
            np.asarray(vals, dtype=float),
            ls=ls,
            marker=mk,
            color=col,
            ms=5,
            label=name,
        )
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_xticks(x, list(layouts), rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("ranks x threads")
    ax.set_ylabel("efficiency")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=_lfs(8))
    if title:
        ax.set_title(title, loc="left")
    return fig


def plot_time_breakdown(
    layouts: Sequence[str],
    parts: dict[str, Sequence[float]],
    total: Sequence[float] | None = None,
    *,
    title: str | None = None,
) -> Figure:
    """Stacked wall time per component for each ranks x threads layout.

    Bars stack the components in the order given; with ``total`` the rest of
    the total is drawn as a grey ``other`` segment, and the total is printed
    above the bar. Segments differ in hatch as well as colour. Components
    that nest (a stage and its own sub-stage) must not both be passed.
    """
    _style()
    fig, ax = plt.subplots(figsize=(1.6 + 0.7 * len(layouts), 4.2), layout="constrained")
    x = np.arange(len(layouts))
    bottom = np.zeros(len(layouts))
    palette = [SURROGATE, ORACLE, REFERENCE, ACQUISITION, HIGHLIGHT, NEUTRAL]
    hatches = ["", "//", "..", "xx", "\\\\", "++"]
    for k, (name, vals) in enumerate(parts.items()):
        v = np.nan_to_num(np.asarray(vals, dtype=float))
        ax.bar(
            x,
            v,
            bottom=bottom,
            color=palette[k % len(palette)],
            hatch=hatches[k % len(hatches)],
            edgecolor="white",
            lw=0.5,
            label=name,
        )
        bottom += v
    if total is not None:
        t = np.asarray(total, dtype=float)
        rest = np.clip(t - bottom, 0, None)
        ax.bar(
            x,
            rest,
            bottom=bottom,
            color="#d0d0d0",
            edgecolor="white",
            lw=0.5,
            label="other",
        )
        for xi, ti in zip(x, t, strict=True):
            if np.isfinite(ti):
                ax.annotate(
                    f"{ti:.0f}",
                    (xi, ti),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
    ax.set_xticks(x, list(layouts), rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("ranks x threads")
    ax.set_ylabel("wall time (s)")
    ax.legend(frameon=False, fontsize=_lfs(8), ncols=2)
    if title:
        ax.set_title(title, loc="left")
    return fig


def _dodge_cores(cores: np.ndarray) -> np.ndarray:
    """Multiplicative offsets so layouts of one core count sit side by side."""
    cores = np.asarray(cores, dtype=float)
    dodge = np.ones(len(cores))
    for v in np.unique(cores):
        idx = np.where(cores == v)[0]
        for k, j in enumerate(idx):
            dodge[j] = 1.0 + 0.08 * (k - (len(idx) - 1) / 2)
    return dodge


def _best_per_core(cores, values):
    """Core counts and the largest value at each, for the joining line."""
    u = np.unique(cores)
    return u, np.array([np.nanmax(values[cores == v]) for v in u])


def plot_pop_factors(
    tables: PopTable | Sequence[PopTable],
    *,
    guide: float | None = 0.8,
    title: str | None = None,
) -> Figure:
    """Fixed-work speedup and the POP efficiency factors against cores.

    One row of two panels per table. Left: ``T(reference) / T(layout)`` on
    log axes, one marker per ``ranks x threads`` layout, layouts of one core
    count dodged sideways and a line through the fastest layout per core
    count; no ideal line is drawn. Right: the fixed-work efficiency (speedup
    per core ratio) and every factor of ``table.metrics`` on one 0 to 1 axis
    against cores, each with its own marker and line style, with the
    ``guide`` level (0.8, the threshold the POP methodology uses) as a dotted
    line. A row is titled by its table's cell and quantity.
    """
    _style()
    tables = [tables] if isinstance(tables, PopTable) else list(tables)
    fig, axes = plt.subplots(
        len(tables),
        2,
        figsize=(8.0, 3.4 * len(tables)),
        layout="constrained",
        width_ratios=(1.0, 1.35),
        squeeze=False,
    )
    styles = [*_SCALING_STYLES, ("-", "v", HIGHLIGHT), ("--", "P", NEUTRAL)] * 2
    for table, (a1, a2) in zip(tables, axes, strict=True):
        cores = np.asarray(table.cores, dtype=float)
        dodge = _dodge_cores(cores)
        x = cores * dodge
        sp = table.speedup()
        u, best = _best_per_core(cores, sp)
        a1.plot(u, best, "-", color=SURROGATE, lw=1.0, alpha=0.6)
        a1.plot(x, sp, ls="none", marker="o", color=SURROGATE, ms=5)
        for xi, yi, lab, d in zip(x, sp, table.layouts, dodge, strict=True):
            # the lower-dodged layout of a shared core count labels to the left
            a1.annotate(
                lab,
                (xi, yi),
                xytext=(-4 if d < 1.0 else 4, -9 if d <= 1.0 else 3),
                textcoords="offset points",
                ha="right" if d < 1.0 else "left",
                fontsize=7,
                color=NEUTRAL,
            )
        a1.set_xscale("log", base=2)
        a1.set_yscale("log", base=2)
        a1.set_xticks(u, [f"{int(v)}" for v in u])
        yt = 2.0 ** np.arange(0, int(np.ceil(np.log2(np.nanmax(sp)))) + 1)
        a1.set_yticks(yt, [f"{v:g}" for v in yt])
        a1.yaxis.set_minor_formatter(FuncFormatter(lambda _v, _p: ""))
        a1.set_xlabel("cores (ranks x threads)")
        a1.set_ylabel(f"fixed-work speedup, {table.quantity}")
        series = {"fixed-work efficiency": table.efficiency(), **table.metrics}
        for (name, raw), (ls, mk, col) in zip(series.items(), styles, strict=False):
            vals = np.asarray(raw, dtype=float)
            u, best = _best_per_core(cores, vals)
            a2.plot(u, best, ls=ls, color=col, lw=1.0, alpha=0.6)
            a2.plot(x, vals, ls="none", marker=mk, color=col, ms=5, label=name)
        if guide is not None:
            a2.axhline(guide, color="black", lw=0.8, ls=":")
            a2.annotate(
                f"{guide:g}",
                (u[0], guide),
                xytext=(0, 2),
                textcoords="offset points",
                fontsize=7,
                color=NEUTRAL,
            )
        a2.set_xscale("log", base=2)
        a2.set_xticks(u, [f"{int(v)}" for v in u])
        a2.set_xlabel("cores (ranks x threads)")
        a2.set_ylabel("efficiency")
        # a superlinear point (cache effects at few ranks) stays in view
        top = max(
            [1.0]
            + [float(np.nanmax(np.asarray(v, dtype=float))) for v in series.values()]
        )
        a2.set_ylim(0, top + 0.05)
        a2.legend(frameon=False, fontsize=_lfs(8), loc="lower left")
        if table.cell or len(tables) > 1:
            a1.set_title(f"{table.cell or ''}, {table.quantity}".strip(", "), loc="left")
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=11)
    return fig


_COMPONENT_PALETTE = [
    ORACLE,
    SURROGATE,
    REFERENCE,
    ACQUISITION,
    HIGHLIGHT,
    "#7b3294",
    "#008837",
    NEUTRAL,
]
_COMPONENT_HATCHES = ["", "//", "..", "xx", "\\\\", "++", "oo", ""]
UNATTRIBUTED_COLOR = "#d0d0d0"


def plot_component_breakdown(
    table: ComponentTable,
    layout: str,
    *,
    cells: Sequence[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Where the wall of one layout goes, per cell, with the Amdahl bound.

    Left: one horizontal bar per cell at ``layout`` stacking the median
    seconds of each component in the order of the table, the rest of the
    total as a grey ``unattributed`` segment, and the total at the bar end.
    Right: for every component the bound ``1 / (1 - f)`` on the whole-run
    speedup if that component alone took no time, ``f`` being its share of
    the wall; one marker per cell. A component that is a large share of the
    wall is the only one whose removal moves the total much.
    """
    _style()
    cells = list(cells or table.cells)
    for c in cells:
        if (c, layout) not in table.total:
            msg = f"{c} has no layout {layout!r}; it has {table.layouts.get(c, [])}"
            raise ValueError(msg)
    names = [*table.components, "unattributed"]
    styles = {
        n: (
            _COMPONENT_PALETTE[k % len(_COMPONENT_PALETTE)],
            _COMPONENT_HATCHES[k % len(_COMPONENT_HATCHES)],
        )
        for k, n in enumerate(table.components)
    }
    styles["unattributed"] = (UNATTRIBUTED_COLOR, "")
    fig, (a1, a2) = plt.subplots(
        1,
        2,
        figsize=(8.6, 1.6 + 0.3 * len(names)),
        layout="constrained",
        width_ratios=(1.5, 1.0),
    )
    y = np.arange(len(cells))
    left = np.zeros(len(cells))
    for name in names:
        vals = np.array(
            [
                table.unattributed(c, layout)
                if name == "unattributed"
                else table.parts[(c, layout)].get(name, 0.0)
                for c in cells
            ]
        )
        col, hatch = styles[name]
        a1.barh(
            y,
            vals,
            left=left,
            color=col,
            hatch=hatch,
            edgecolor="white",
            lw=0.5,
            height=0.62,
            label=name,
        )
        left += vals
    for yi, c in zip(y, cells, strict=True):
        a1.annotate(
            f"{table.total[(c, layout)]:.0f} s",
            (left[yi], yi),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    a1.set_yticks(y, cells)
    a1.set_ylim(len(cells) - 0.4, -0.6)
    a1.set_xlim(0, left.max() * 1.14)
    a1.set_xlabel("wall time (s)")
    a1.legend(
        frameon=False,
        fontsize=_lfs(8),
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
    )
    markers = ["o", "s", "^", "D", "v"]
    colors = [SURROGATE, ACQUISITION, REFERENCE, ORACLE, NEUTRAL]
    rows = np.arange(len(names))
    step = 0.6 / max(len(cells), 1)
    finite = []
    for k, c in enumerate(cells):
        bounds = table.amdahl_bounds(c, layout)
        vals = np.array([bounds.get(n, np.nan) for n in names])
        finite += [v for v in vals if np.isfinite(v)]
        off = (k - (len(cells) - 1) / 2) * step
        col = colors[k % len(colors)]
        a2.plot(
            vals,
            rows + off,
            ls="none",
            marker=markers[k % len(markers)],
            color=col,
            ms=4.5,
            label=c,
        )
        for v, r in zip(vals, rows + off, strict=True):
            if np.isfinite(v):
                a2.annotate(
                    f"{v:.2f}x",
                    (v, r),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=7,
                    color=col,
                )
    a2.axvline(1.0, color="black", lw=0.8, ls=":")
    a2.set_yticks(rows, names, fontsize=8)
    a2.set_ylim(len(names) - 0.5, -0.5)
    a2.set_xlabel("bound on whole-run speedup\nif the component took no time")
    a2.set_xlim(0.95, (max(finite) if finite else 2.0) * 1.3)
    a2.legend(frameon=False, fontsize=_lfs(8), loc="upper right")
    fig.suptitle(title or f"{layout} (ranks x threads)", x=0.02, ha="left", fontsize=11)
    return fig


# -------------------------------------------------------------- per-case boards


_ROW_PITCH_IN = 0.27
_PANEL_WIDTH_IN = 4.2
_CHAR_PT = 3.9  # width of one digit at 7 pt, with a little slack


def _board_axes(boards, labels_pad_in=None):
    """One axes per board, top-aligned, with the same row pitch and bar size.

    The height of a panel is its row count times a fixed pitch, so a board
    with fewer rows leaves no empty rows in its frame.
    """
    n_max = max(len(b.rows) for b in boards)
    left_in = [
        0.25 + 0.075 * max((len(r.label) for r in b.rows), default=4) for b in boards
    ]
    if labels_pad_in is not None:
        left_in = [labels_pad_in] * len(boards)
    top_in, bottom_in, right_in, gap_in = 0.45, 1.1, 0.35, 0.15
    width = sum(left_in) + _PANEL_WIDTH_IN * len(boards) + right_in * len(boards) + gap_in
    height = top_in + n_max * _ROW_PITCH_IN + bottom_in
    fig = plt.figure(figsize=(width, height))
    axes = []
    x_in = 0.0
    for b, left in zip(boards, left_in, strict=True):
        x_in += left
        h_in = len(b.rows) * _ROW_PITCH_IN
        axes.append(
            fig.add_axes(
                (
                    x_in / width,
                    (bottom_in + (n_max - len(b.rows)) * _ROW_PITCH_IN) / height,
                    _PANEL_WIDTH_IN / width,
                    h_in / height,
                )
            )
        )
        x_in += _PANEL_WIDTH_IN + right_in
    return fig, axes, n_max


def _board_title(b: CaseBoard) -> str:
    return f"{b.name}: {b.n_certified}/{len(b.rows)} first-order saddles"


def plot_cases_calls(boards: Sequence[CaseBoard]) -> Figure:
    """Oracle evaluations per case: search and independent validation, stacked.

    One panel per board, one horizontal bar per case in the order given. Teal:
    search; coral: independent validation; the total sits at the bar end. A
    case that is not certified is hatched. Baseline markers (``CaseBoard.baselines``)
    sit on the case's row, open where the method did not converge.
    """
    _style()
    fig, axes, _ = _board_axes(boards)
    methods: dict[str, tuple[str, str]] = {}
    any_uncert = False
    for ax, b in zip(axes, boards, strict=True):
        y = np.arange(len(b.rows))
        s = np.array([r.search_calls for r in b.rows])
        v = np.array([r.validation_calls for r in b.rows])
        cert = np.array([r.certified for r in b.rows])
        for sel, hatch in ((cert, ""), (~cert, "////")):
            ax.barh(
                y[sel],
                s[sel],
                color=SURROGATE,
                hatch=hatch,
                edgecolor="white",
                lw=0.4,
                height=0.7,
            )
            ax.barh(
                y[sel],
                v[sel],
                left=s[sel],
                color=ACQUISITION,
                hatch=hatch,
                edgecolor="white",
                lw=0.4,
                height=0.7,
            )
        any_uncert |= bool((~cert).any())
        top = float((s + v).max()) if len(b.rows) else 1.0
        for yi, tot in zip(y, s + v, strict=True):
            ax.annotate(
                f"{tot:.0f}",
                (tot, yi),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
            )
        for k, (name, per_case) in enumerate(b.baselines.items()):
            mk, col = _BASE_STYLE[k % len(_BASE_STYLE)]
            methods.setdefault(name, (mk, col))
            for r_i, r in enumerate(b.rows):
                if r.case in per_case:
                    calls, conv = per_case[r.case]
                    top = max(top, calls)
                    ax.plot(
                        calls,
                        r_i,
                        ls="none",
                        marker=mk,
                        ms=6,
                        mec=col,
                        mew=1.5,
                        mfc=col if conv else "white",
                        zorder=3,
                    )
        ax.set_yticks(y, [r.label for r in b.rows], fontsize=8)
        ax.set_ylim(len(b.rows) - 0.5, -0.5)
        ax.set_xlim(0, top * 1.12)
        ax.set_xlabel("Oracle evaluations")
        pt_per_unit = _PANEL_WIDTH_IN * 72 / (top * 1.12)
        for yi, si in zip(y, s, strict=True):
            txt = f"{si:.0f}"
            if si * pt_per_unit >= len(txt) * _CHAR_PT + 6:
                ax.annotate(
                    txt,
                    (si / 2, yi),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    bbox={"facecolor": SURROGATE, "edgecolor": "none", "pad": 1.0},
                )
        ax.set_title(_board_title(b), loc="left", fontsize=10)
        ax.grid(axis="x", color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
    handles = [
        Patch(facecolor=SURROGATE, label="search"),
        Patch(facecolor=ACQUISITION, label="independent validation"),
    ]
    if any_uncert:
        handles.append(
            Patch(
                facecolor="#bdbdbd",
                hatch="////",
                edgecolor="white",
                label="not certified",
            )
        )
    handles += [
        Line2D([], [], ls="none", marker=mk, mec=col, mfc=col, ms=6, label=name)
        for name, (mk, col) in methods.items()
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=_lfs(8),
        ncols=len(handles),
        loc="lower center",
    )
    return fig


def _wall_tick(wall: float, sd: float | None) -> str:
    """Bar label. With a deviation, both numbers share its printed place.

    The place is two significant figures of the deviation, and at most
    hundredths. Without one, a time of at least ten seconds is an integer.
    """
    if not sd or sd <= 0:
        return f"{wall:,.0f}" if wall >= _WALL_INT_MIN else f"{wall:.1f}"
    places = int(max(0, min(2, 1 - np.floor(np.log10(sd)))))
    return f"{wall:,.{places}f} ± {sd:,.{places}f}"


def plot_cases_wall(
    boards: Sequence[CaseBoard], *, reference_label: str = "reference"
) -> Figure:
    """Wall time per case on a log axis, one bar each, with its value.

    A thin vertical tick on the case's row marks ``reference_wall_s`` when the
    table has one (the value then sits inside the bar end, off the tick);
    ``reference_label`` names it in the legend. ``wall_sd`` draws a horizontal
    error bar, the sample standard deviation of repeated runs, and the label
    writes that deviation beside the time. The axis spans
    half a decade below the fastest and a third above the slowest case.
    Not-certified cases are hatched.
    """
    _style()
    fig, axes, _ = _board_axes(boards)
    have_ref = False
    have_uncert = False
    have_sd = False

    def _lo(r) -> float:
        if r.wall_sd and r.wall_s > r.wall_sd:
            return r.wall_s - r.wall_sd
        return r.wall_s

    def _hi(r) -> float:
        top = max(r.wall_s, r.reference_wall_s or 0)
        if r.wall_sd:
            top = max(top, r.wall_s + r.wall_sd)
        return top

    lo_all = min(_lo(r) for b in boards for r in b.rows)
    lo = lo_all * 10**-0.5  # half a decade below the fastest case
    hi_all = max(_hi(r) for b in boards for r in b.rows)
    hi = hi_all * 10 ** (1 / 3)  # a third of a decade above the slowest
    for ax, b in zip(axes, boards, strict=True):
        y = np.arange(len(b.rows))
        w = np.array([r.wall_s for r in b.rows])
        cert = np.array([r.certified for r in b.rows])
        for sel, hatch in ((cert, ""), (~cert, "////")):
            ax.barh(
                y[sel],
                w[sel] - lo,
                left=lo,
                color=SURROGATE,
                hatch=hatch,
                edgecolor="white",
                lw=0.4,
                height=0.7,
            )
        have_uncert |= bool((~cert).any())
        for yi, wi, row in zip(y, w, b.rows, strict=True):
            txt = _wall_tick(wi, row.wall_sd)
            # With a reference tick the value goes inside the bar end so it
            # never reads as the tick's value; without one it follows the bar.
            # A deviation stays outside: the bar end is too narrow for it.
            inside = (
                bool(row.reference_wall_s)
                and not row.wall_sd
                and wi / lo >= _WALL_INSIDE_MIN_RATIO
            )
            ax.annotate(
                txt,
                (wi, yi),
                xytext=(-3 if inside else 3, 0),
                textcoords="offset points",
                ha="right" if inside else "left",
                va="center",
                fontsize=7,
                color="white" if inside else "black",
                bbox={"facecolor": SURROGATE, "edgecolor": "none", "pad": 1.0}
                if inside
                else None,
            )
        sd = np.array([r.wall_sd or 0.0 for r in b.rows])
        if np.any(sd > 0):
            have_sd = True
            ax.errorbar(
                w,
                y,
                xerr=sd,
                fmt="none",
                ecolor="black",
                elinewidth=0.7,
                capsize=1.5,
                capthick=0.7,
                zorder=4,
            )
        for yi, r in zip(y, b.rows, strict=True):
            if r.reference_wall_s:
                have_ref = True
                ax.vlines(
                    r.reference_wall_s,
                    yi - 0.4,
                    yi + 0.4,
                    color="black",
                    lw=1.6,
                    zorder=3,
                )
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        _plain_log_axis(ax.xaxis)
        ax.xaxis.set_major_locator(LogLocator(subs=(1.0,)))
        ax.xaxis.set_minor_locator(LogLocator(subs=(3.0,)))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.tick_params(axis="x", which="both", length=4, pad=3)
        ax.tick_params(axis="x", which="minor", labelsize=plt.rcParams["xtick.labelsize"])
        ax.set_yticks(y, [r.label for r in b.rows], fontsize=8)
        ax.set_ylim(len(b.rows) - 0.5, -0.5)
        ax.set_xlabel("Wall time (s)")
        ax.set_title(_board_title(b), loc="left", fontsize=10)
        ax.grid(axis="x", color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
    handles = []
    if have_ref:
        handles.append(
            Line2D(
                [],
                [],
                color="black",
                lw=0,
                marker="|",
                ms=10,
                mew=1.6,
                label=reference_label,
            )
        )
    if have_sd:
        handles.append(
            Line2D(
                [],
                [],
                color="black",
                lw=0.7,
                marker="|",
                ms=6,
                mew=0.7,
                label="sample sd",
            )
        )
    if have_uncert:
        handles.append(
            Patch(
                facecolor="#bdbdbd",
                hatch="////",
                edgecolor="white",
                label="not certified",
            )
        )
    if handles:
        fig.legend(
            handles=handles,
            frameon=False,
            fontsize=_lfs(8),
            ncols=len(handles),
            loc="lower center",
        )
    return fig
