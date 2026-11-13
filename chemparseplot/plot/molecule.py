"""Vector ball-and-stick drawings on matplotlib axes.

Atoms are discs in angstrom; text sizes and line widths are in points. A
structure therefore keeps legible labels at any panel size, every panel of a
figure shares one renderer, and the saved figure stays vector.

```{versionadded} 1.14.0
```
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from chemparseplot.plot.theme import RUHI_COLORS

INK = "#1C2421"
# Fills follow the usual element conventions in the tones of the Ruhi palette;
# elements that meet in one structure (O and Br, S and Mg) keep distinct hues.
ELEMENT_COLORS = {
    "H": "#FFFFFF",
    "C": "#AFB3B6",
    "N": "#298EE6",
    "O": RUHI_COLORS["coral"],
    "F": "#C5E59B",
    "Mg": "#8CCB7B",
    "Si": "#E9CFA6",
    "P": "#F99A55",
    "S": RUHI_COLORS["sunshine"],
    "Cl": "#45B97C",
    "Br": "#A8573B",
    "Pt": "#D2E7FA",
}
FALLBACK_ELEMENT_COLOR = "#D9DCDE"


@dataclass(frozen=True)
class MoleculeStyle:
    """Sizes and colours of a ball-and-stick drawing.

    Lengths that describe atoms are fractions of the covalent radius; every
    other length is in points, so it does not change with the drawing scale.
    """

    radius_scale: float = 0.5
    outline: str = INK
    outline_width: float = 0.6
    bond_width: float = 1.5
    bond_width_min: float = 1.0
    bond_width_max: float = 4.6
    bond_casing: float = 0.45
    weak_order: float = 0.5
    weak_width: float = 1.3
    label_size: float = 7.0
    label_gap: float = 1.0
    label_halo: float = 1.6
    label_color: str = INK
    label_directions: int = 24
    label_rings: int = 4
    element_colors: Mapping[str, str] = field(default_factory=lambda: ELEMENT_COLORS)

    def color_of(self, symbol: str) -> str:
        return self.element_colors.get(symbol, FALLBACK_ELEMENT_COLOR)

    def bond_linewidth(self, order: float) -> float:
        """Line width in points of the coloured part of a bond.

        A dotted bond grows from half of ``weak_width`` at order zero towards
        one and a half times it at ``weak_order``, so a half-formed bond reads
        heavier than a faint contact. It never exceeds the stroke of the
        thinnest solid bond, so the drawn weight does not fall as the order
        rises through ``weak_order``.
        """
        if order < self.weak_order:
            dots = self.weak_width * (0.5 + max(order, 0.0) / self.weak_order)
            return min(dots, self.bond_width_min + 2 * self.bond_casing)
        return float(
            np.clip(order * self.bond_width, self.bond_width_min, self.bond_width_max)
        )

    def bond_stroke(self, order: float) -> float:
        """Drawn weight in points: the coloured width plus the casing of a solid bond."""
        width = self.bond_linewidth(order)
        return width if order < self.weak_order else width + 2 * self.bond_casing


def atom_radii(symbols: Sequence[str], scale: float = 0.5) -> np.ndarray:
    """Disc radii in angstrom: ``scale`` times the covalent radius."""
    from ase.data import atomic_numbers, covalent_radii  # noqa: PLC0415

    return np.array([scale * covalent_radii[atomic_numbers[s]] for s in symbols])


def text_extent(text: str, size: float) -> tuple[float, float]:
    """Width and height in points of ``text`` in the current font family."""
    from matplotlib.font_manager import FontProperties  # noqa: PLC0415
    from matplotlib.textpath import TextPath  # noqa: PLC0415

    box = TextPath((0, 0), text, size=size, prop=FontProperties()).get_extents()
    return float(box.width), float(box.height)


def points_per_unit(ax) -> float:
    """Points per data unit along x of an axes with settled limits."""
    width = ax.get_position().width * ax.figure.get_figwidth() * 72
    lo, hi = ax.get_xlim()
    return width / (hi - lo)


def frame_structure(fig, box, centre, scale):
    """Add an axes at ``box`` (points) drawing ``scale`` points per angstrom.

    ``box`` is ``(x0, y0, width, height)`` from the lower-left corner of the
    figure and ``centre`` is the data point shown in its middle.
    """
    x0, y0, w, h = box
    fig_w, fig_h = (v * 72 for v in fig.get_size_inches())
    ax = fig.add_axes((x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h))
    ax.set_axis_off()
    ax.set_xlim(centre[0] - w / (2 * scale), centre[0] + w / (2 * scale))
    ax.set_ylim(centre[1] - h / (2 * scale), centre[1] + h / (2 * scale))
    return ax


def draw_discs(
    ax,
    xy,
    depth,
    radii,
    *,
    facecolors,
    edgecolors=INK,
    linewidths=0.6,
    hatches=None,
    zorder=2.0,
):
    """Draw one disc per atom, nearer atoms over farther ones.

    ``depth`` grows towards the viewer. Scalar styles apply to every atom;
    sequences give one value per atom. Returns the patches in atom order.
    """
    from matplotlib.patches import Circle  # noqa: PLC0415

    xy = np.asarray(xy, dtype=float)
    n = len(xy)

    def per_atom(value):
        if isinstance(value, str) or not isinstance(value, Sequence | np.ndarray):
            return [value] * n
        return list(value)

    faces, edges, widths = (
        per_atom(facecolors),
        per_atom(edgecolors),
        per_atom(linewidths),
    )
    marks = per_atom(hatches)
    rank = np.argsort(np.argsort(np.asarray(depth, dtype=float), kind="stable"))
    patches = []
    for k in range(n):
        patch = Circle(
            xy[k],
            radii[k],
            facecolor=faces[k],
            edgecolor=edges[k],
            linewidth=widths[k],
            hatch=marks[k],
            zorder=zorder + rank[k] / n,
        )
        ax.add_patch(patch)
        patches.append(patch)
    return patches


def _box_hits_circle(box, centre, radius) -> bool:
    x0, y0, x1, y1 = box
    nearest = np.array([min(max(centre[0], x0), x1), min(max(centre[1], y0), y1)])
    return float(np.linalg.norm(nearest - centre)) < radius


def _box_hits_segment(box, p, q) -> bool:
    """Liang-Barsky clip of the segment ``p``-``q`` against ``box``."""
    x0, y0, x1, y1 = box
    t0, t1 = 0.0, 1.0
    d = q - p
    for denom, numer in (
        (-d[0], p[0] - x0),
        (d[0], x1 - p[0]),
        (-d[1], p[1] - y0),
        (d[1], y1 - p[1]),
    ):
        if denom == 0:
            if numer < 0:
                return False
            continue
        t = numer / denom
        if denom < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def _boxes_touch(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _grown(box, pad):
    return (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)


def place_labels(centres, radii, segments, labels, size, *, style=None):
    """Choose a spot beside each labelled atom, clear of atoms and labels.

    All lengths are in points inside a box of ``size`` ``(width, height)``
    whose origin is the lower-left corner. ``segments`` lists the bonds as
    ``(start, end, half_width)`` and ``labels`` maps an atom index to its
    text. Among the spots that touch no disc and no other label, the one
    crossing the fewest bonds wins, then the nearest, then the one pointing
    away from the other atoms. An atom with no free spot takes the candidate
    with the fewest collisions, so a crowded structure still draws.
    Returns ``{index: (x, y)}`` label centres.
    """
    style = style or MoleculeStyle()
    w, h = size
    centres = np.asarray(centres, dtype=float)
    directions = [
        np.array([math.cos(a), math.sin(a)])
        for a in np.linspace(0, 2 * math.pi, style.label_directions, endpoint=False)
    ]
    boxes, placed = [], {}
    for k, text in labels.items():
        tw, th = text_extent(text, style.label_size)
        away = np.zeros(2)
        for m in range(len(centres)):
            if m != k:
                diff = centres[k] - centres[m]
                away += diff / max(float(np.linalg.norm(diff)), 1e-6) ** 3
        norm = float(np.linalg.norm(away))
        away = away / norm if norm > 0 else np.array([0.0, 1.0])
        best = None
        for ring in range(style.label_rings):
            for u in directions:
                reach = (
                    radii[k]
                    + style.label_gap
                    + ring * th
                    + abs(u[0]) * tw / 2
                    + abs(u[1]) * th / 2
                )
                c = centres[k] + reach * u
                box = (c[0] - tw / 2, c[1] - th / 2, c[0] + tw / 2, c[1] + th / 2)
                outside = box[0] < 0 or box[1] < 0 or box[2] > w or box[3] > h
                roomy = _grown(box, style.label_gap)
                collisions = sum(
                    _box_hits_circle(roomy, centres[m], radii[m])
                    for m in range(len(centres))
                ) + sum(_boxes_touch(roomy, other) for other in boxes)
                crossings = sum(
                    _box_hits_segment(_grown(box, half + style.label_gap), p, q)
                    for p, q, half in segments
                )
                turn = math.acos(max(-1.0, min(1.0, float(u @ away)))) / math.pi
                score = (collisions + outside, crossings, ring, turn)
                if best is None or score < best[0]:
                    best = (score, c, box)
        boxes.append(best[2])
        placed[k] = (float(best[1][0]), float(best[1][1]))
    return placed


def draw_molecule(
    ax,
    symbols,
    positions,
    bonds,
    *,
    labels=None,
    style=None,
    cmap=None,
    norm=None,
):
    """Draw atoms as depth-sorted discs and bonds as lines on ``ax``.

    ``positions`` holds the drawing plane in its first two columns and the
    depth, growing towards the viewer, in the third. ``bonds`` lists
    ``(i, j, order)`` with zero-based indices. A bond below
    ``style.weak_order`` is dotted; the others are solid with a width that
    grows with the order. With ``cmap`` and ``norm`` the order also sets the
    colour. ``labels`` maps an atom index to the text placed beside it.

    The limits of ``ax`` must already hold the structure at equal scale on
    both axes (see :func:`frame_structure`). Returns the drawn artists.
    """
    from matplotlib.lines import Line2D  # noqa: PLC0415
    from matplotlib.patheffects import Normal, Stroke, withStroke  # noqa: PLC0415

    style = style or MoleculeStyle()
    positions = np.asarray(positions, dtype=float)
    xy, depth = positions[:, :2], positions[:, 2]
    n = len(symbols)
    radii = atom_radii(symbols, style.radius_scale)
    rank = np.argsort(np.argsort(depth, kind="stable"))
    lines = []
    for i, j, order in bonds:
        weak = order < style.weak_order
        color = style.outline if cmap is None else cmap(norm(order))
        width = style.bond_linewidth(order)
        line = Line2D(
            [xy[i, 0], xy[j, 0]],
            [xy[i, 1], xy[j, 1]],
            linewidth=width,
            color=color,
            linestyle=(0, (0.1, 1.9)) if weak else "-",
            dash_capstyle="round",
            solid_capstyle="butt",
            zorder=2 + (min(rank[i], rank[j]) - 0.5) / n,
        )
        if not weak and style.bond_casing > 0:
            line.set_path_effects(
                [
                    Stroke(
                        linewidth=width + 2 * style.bond_casing, foreground=style.outline
                    ),
                    Normal(),
                ]
            )
        ax.add_line(line)
        lines.append(line)
    discs = draw_discs(
        ax,
        xy,
        depth,
        radii,
        facecolors=[style.color_of(s) for s in symbols],
        edgecolors=style.outline,
        linewidths=style.outline_width,
    )
    texts = []
    if labels:
        scale = points_per_unit(ax)
        origin = np.array([ax.get_xlim()[0], ax.get_ylim()[0]])
        size = (
            (ax.get_xlim()[1] - origin[0]) * scale,
            (ax.get_ylim()[1] - origin[1]) * scale,
        )
        centres = (xy - origin) * scale
        spots = place_labels(
            centres,
            radii * scale,
            [
                (
                    centres[i],
                    centres[j],
                    style.bond_linewidth(order) / 2 + style.bond_casing,
                )
                for i, j, order in bonds
            ],
            labels,
            size,
            style=style,
        )
        for k, spot in spots.items():
            where = np.asarray(spot) / scale + origin
            texts.append(
                ax.text(
                    where[0],
                    where[1],
                    labels[k],
                    fontsize=style.label_size,
                    color=style.label_color,
                    ha="center",
                    va="center",
                    zorder=4,
                    path_effects=[
                        withStroke(linewidth=style.label_halo, foreground="white")
                    ],
                )
            )
    return {"bonds": lines, "atoms": discs, "labels": texts}
