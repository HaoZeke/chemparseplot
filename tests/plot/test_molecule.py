"""Vector ball-and-stick drawings: depth order, bond styles and label room."""

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex

from chemparseplot.plot.molecule import (
    MoleculeStyle,
    atom_radii,
    draw_disc_field,
    draw_discs,
    draw_molecule,
    frame_structure,
    place_labels,
    points_per_unit,
)

pytest.importorskip("ase")

WATER = (
    ["O", "H", "H"],
    np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.4], [-0.24, 0.93, -0.4]]),
)


def _framed(scale=30.0, box=(0, 0, 120, 90)):
    fig = plt.figure(figsize=(2.0, 1.5))
    return fig, frame_structure(fig, box, (0.3, 0.4), scale)


def test_frame_draws_the_requested_points_per_angstrom():
    fig, ax = _framed(scale=30.0)
    assert points_per_unit(ax) == pytest.approx(30.0)
    lo, hi = ax.get_ylim()
    assert (hi - lo) * 30.0 == pytest.approx(90.0)
    assert not ax.axison
    plt.close(fig)


def test_discs_stack_by_depth_and_take_per_atom_styles():
    fig, ax = _framed()
    patches = draw_discs(
        ax,
        [[0, 0], [0.2, 0], [0.4, 0]],
        [2.0, 0.0, 1.0],
        [0.3, 0.3, 0.3],
        facecolors=["#ffffff", "#000000", "#ff0000"],
        edgecolors="#222222",
        linewidths=[0.0, 0.5, 1.0],
        hatches=[None, None, "xxxx"],
    )
    order = np.argsort([p.get_zorder() for p in patches])
    assert list(order) == [1, 2, 0]
    assert [to_hex(p.get_facecolor()) for p in patches] == [
        "#ffffff",
        "#000000",
        "#ff0000",
    ]
    assert [p.get_linewidth() for p in patches] == [0.0, 0.5, 1.0]
    assert patches[2].get_hatch() == "xxxx"
    plt.close(fig)


def test_bond_order_sets_width_dots_and_colour():
    symbols, positions = WATER
    fig, ax = _framed()
    cmap, norm = colormaps["ruhi_bond_order"], Normalize(0, 3)
    drawn = draw_molecule(
        ax, symbols, positions, [(0, 1, 1.0), (0, 2, 0.3)], cmap=cmap, norm=norm
    )
    strong, weak = drawn["bonds"]
    assert strong.get_linestyle() == "-"
    assert weak.get_linestyle() != "-"
    assert strong.get_linewidth() > weak.get_linewidth()
    assert to_hex(strong.get_color()) == to_hex(cmap(norm(1.0)))
    assert to_hex(weak.get_color()) == to_hex(cmap(norm(0.3)))
    # a bond lies under both of its atoms
    for line, (i, j) in zip(drawn["bonds"], ((0, 1), (0, 2)), strict=True):
        assert line.get_zorder() < min(
            drawn["atoms"][i].get_zorder(), drawn["atoms"][j].get_zorder()
        )
    plt.close(fig)


def test_drawn_bond_weight_never_falls_as_the_order_rises():
    orders = np.linspace(0.0, 3.5, 351)
    # the second style asks for dots wider than its thinnest solid bond
    for style in (MoleculeStyle(), MoleculeStyle(weak_width=4.0, bond_casing=0.0)):
        strokes = [style.bond_stroke(order) for order in orders]
        assert all(b >= a for a, b in zip(strokes, strokes[1:], strict=False))
        assert style.bond_linewidth(9.0) == style.bond_width_max
        assert style.bond_stroke(1.0) == style.bond_linewidth(1.0) + 2 * style.bond_casing
    # with the default sizes the dots grow with the order
    default = MoleculeStyle()
    assert default.bond_linewidth(0.1) < default.bond_linewidth(0.3)
    assert default.bond_linewidth(0.3) < default.bond_linewidth(0.45)


def test_labels_sit_outside_every_disc_in_points():
    symbols, positions = WATER
    fig, ax = _framed()
    style = MoleculeStyle(label_size=7.0)
    drawn = draw_molecule(
        ax,
        symbols,
        positions,
        [(0, 1, 1.0), (0, 2, 1.0)],
        labels={0: "O1", 1: "H2", 2: "H3"},
        style=style,
    )
    assert [t.get_text() for t in drawn["labels"]] == ["O1", "H2", "H3"]
    assert {t.get_fontsize() for t in drawn["labels"]} == {7.0}
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in drawn["labels"]]
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            assert not boxes[a].overlaps(boxes[b])
    radii = atom_radii(symbols, style.radius_scale)
    for box in boxes:
        for centre, radius in zip(positions[:, :2], radii, strict=True):
            cx, cy = ax.transData.transform(centre)
            reach = radius * points_per_unit(ax) * fig.dpi / 72
            nearest = np.array([np.clip(cx, box.x0, box.x1), np.clip(cy, box.y0, box.y1)])
            assert np.hypot(*(nearest - (cx, cy))) >= reach - 1.0
    plt.close(fig)


def test_label_avoids_a_thick_bond_and_falls_back_when_crowded():
    centres = np.array([[40.0, 30.0], [80.0, 30.0]])
    radii = np.array([6.0, 6.0])
    bond = [(centres[0], centres[1], 3.0)]
    spots = place_labels(centres, radii, bond, {0: "C1"}, (120.0, 60.0))
    x, y = spots[0]
    # the label keeps off the bond line and off the disc
    assert abs(y - 30.0) > 3.0 or x < 40.0
    assert np.hypot(x - 40.0, y - 30.0) > 6.0
    # no room at all: the least crowded candidate is still returned
    tight = place_labels(centres, radii, bond, {0: "C1", 1: "C2"}, (1.0, 1.0))
    assert set(tight) == {0, 1}


def test_disc_field_keeps_depth_order_in_one_collection_per_hatch_run(tmp_path):
    rng = np.random.default_rng(3)
    n = 600
    xy = rng.uniform(0, 20, (n, 2))
    depth = rng.uniform(0, 5, n)
    moving = np.zeros(n, dtype=bool)
    moving[np.argsort(depth)[-4:]] = True
    faces = np.where(moving, "#F1DB4B", "#D2E7FA")

    def framed():
        fig = plt.figure(figsize=(3.0, 3.0))
        return fig, frame_structure(fig, (0, 0, 216, 216), (10, 10), 9.0)

    fig, ax = framed()
    field = draw_disc_field(
        ax,
        xy,
        depth,
        1.25,
        facecolors=list(faces),
        edgecolors="#1C2421",
        linewidths=0.4,
        hatches=["xxxx" if m else None for m in moving],
    )
    # the four nearest atoms are hatched, so the field is two runs
    assert len(field) == 2
    assert [len(c.get_offsets()) for c in field] == [n - 4, 4]
    assert field[0].get_hatch() is None and field[1].get_hatch() == "xxxx"
    assert field[0].get_zorder() < field[1].get_zorder()
    back = np.asarray(field[0].get_offsets())
    expected = xy[np.argsort(depth, kind="stable")][: n - 4]
    np.testing.assert_allclose(back, expected)
    # marker diameter in points equals the disc diameter at the axes scale
    assert np.sqrt(field[0].get_sizes()[0]) == pytest.approx(2 * 1.25 * 9.0)
    compact = tmp_path / "field.pdf"
    fig.savefig(compact)
    plt.close(fig)

    fig, ax = framed()
    draw_discs(
        ax,
        xy,
        depth,
        np.full(n, 1.25),
        facecolors=list(faces),
        edgecolors="#1C2421",
        linewidths=0.4,
    )
    patches = tmp_path / "patches.pdf"
    fig.savefig(patches)
    plt.close(fig)
    assert compact.stat().st_size < patches.stat().st_size / 3
