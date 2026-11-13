"""Geometry and sparse-property invariants for reaction interpretation."""

import numpy as np
import pytest
from ase import Atoms

from chemparseplot.plot.reaction import align_reaction, read_wbo


def test_alignment_preserves_distances_and_chirality():
    a = Atoms("CHNO", positions=[[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    b = a.copy()
    b.rotate(47, "x")
    b.translate([3, 4, 5])
    result = align_reaction([a, b], reference=0)
    np.testing.assert_allclose(result[0].positions, result[1].positions, atol=1e-12)
    for source, rotated in zip((a, b), result, strict=True):
        np.testing.assert_allclose(
            source.get_all_distances(), rotated.get_all_distances(), atol=1e-12
        )

        def volume(frame):
            return np.linalg.det(frame.positions[1:] - frame.positions[0])

        assert volume(source) * volume(rotated) > 0


def test_missing_wbo_is_not_a_measured_zero(tmp_path):
    path = tmp_path / "wbo"
    path.write_text("1 2 0.750D+00\n")
    values = read_wbo(path, 3)
    assert values[(1, 2)] == 0.75
    assert (1, 3) not in values
    path.write_text("1 4 0.5\n")
    with pytest.raises(ValueError, match="Invalid"):
        read_wbo(path, 3)


def test_measured_bond_styles_replace_geometric_edges(tmp_path):
    import xyzrender as xr

    from chemparseplot.plot.reaction import apply_wbo_style

    path = tmp_path / "water.xyz"
    path.write_text("3\nwater\nO 0 0 0\nH .95 0 0\nH -.2 .93 0\n")
    original = xr.load(str(path), quick=True)
    graph = apply_wbo_style(original, {(1, 2): 0.4, (1, 3): 1.1, (2, 3): 0.05}).graph
    a, b, c = sorted(graph.nodes)
    assert graph[a][b]["NCI"] is True
    assert graph[a][c]["NCI"] is False
    assert graph[a][b]["bond_color_override"] != graph[a][c]["bond_color_override"]
    assert graph[a][b]["bond_width_override"] < graph[a][c]["bond_width_override"]
    assert not graph.has_edge(b, c)
    assert "bond_color_override" not in original.graph[a][b]


def test_molecular_alignment_ignores_container_periodicity():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    a = Atoms("CHNO", positions=positions + 15, cell=[30, 30, 30], pbc=True)
    b = a.copy()
    b.rotate(47, "x", center="COP")
    b.translate([0.2, -0.3, 0.1])
    result = align_reaction([a, b], reference=1)
    for original, aligned in zip((a, b), result, strict=True):
        np.testing.assert_allclose(
            original.get_all_distances(), aligned.get_all_distances(), atol=1e-12
        )
        assert not aligned.pbc.any()
        assert original.pbc.all()
    np.testing.assert_allclose(result[0].positions, result[1].positions, atol=1e-12)


def test_gallery_places_one_titled_tile_per_reaction(tmp_path):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    from chemparseplot.plot.reaction import plot_reaction_gallery

    image = tmp_path / "tile.png"
    pixels = np.zeros((20, 20, 4))
    pixels[5:15, 5:15] = [0.2, 0.4, 0.6, 1.0]
    plt.imsave(image, pixels)
    bond = {"label": "C1\u2013O2", "distance_A": 1.94, "wbo": 0.41}
    tiles = [
        {
            "title": f"case {i}",
            "subtitle": "ring opening",
            "image": str(image),
            "bonds": [bond] if i % 2 else [],
        }
        for i in range(5)
    ]
    fig = plot_reaction_gallery(tiles, columns=3)
    texts = [t.get_text() for t in fig.texts]
    assert [f"case {i}" for i in range(5)] == [t for t in texts if t.startswith("case")]
    assert texts.count("ring opening") == 5
    assert texts.count("C1\u2013O2: 1.94 Å / 0.41") == 2
    # five structure panels and the bond-order colour bar
    assert len(fig.axes) == 6
    plt.close(fig)
    with pytest.raises(ValueError, match="one column"):
        plot_reaction_gallery(tiles, columns=0)


def test_bond_order_between_separated_atoms_draws_no_edge(tmp_path):
    import xyzrender as xr

    from chemparseplot.plot.reaction import apply_wbo_style

    path = tmp_path / "radicals.xyz"
    path.write_text("3\nradicals\nC 0 0 0\nH 1.09 0 0\nH -3.85 0 0\n")
    graph = apply_wbo_style(
        xr.load(str(path), quick=True), {(1, 2): 0.98, (1, 3): 0.75}
    ).graph
    a, b, c = sorted(graph.nodes)
    assert graph.has_edge(a, b)
    assert not graph.has_edge(a, c)
    # a forming H-H contact of a dissociation saddle keeps its edge
    hh = tmp_path / "hh.xyz"
    hh.write_text("3\nhh\nC 0 0 0\nH 1.1 0 0\nH 1.1 1.29 0\n")
    graph_hh = apply_wbo_style(xr.load(str(hh), quick=True), {(2, 3): 0.15}).graph
    p, r, s = sorted(graph_hh.nodes)
    assert graph_hh.has_edge(r, s)
    # a stretched transition-state contact within the reach keeps its edge
    graph = apply_wbo_style(
        xr.load(str(path), quick=True), {(1, 3): 0.75}, max_contact=4.0
    ).graph
    assert graph.has_edge(a, c)


def test_atlas_rows_make_room_for_every_bond_line(tmp_path):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    from chemparseplot.plot.reaction import plot_reaction_atlas

    image = tmp_path / "state.png"
    plt.imsave(image, np.ones((8, 8, 4)))
    bond = {"label": "C1\u2013C2", "distance_A": 2.2, "wbo": 0.3}

    def case(n):
        return {
            "title": "row",
            "states": [
                {"title": "s", "image": str(image), "bonds": [bond] * n} for _ in range(4)
            ],
        }

    short = plot_reaction_atlas([case(1)] * 4)
    tall = plot_reaction_atlas([case(3)] * 4)
    assert tall.get_figheight() > short.get_figheight()
    # the colour bar sits below the lowest bond annotation
    tall.canvas.draw()
    bar = tall.axes[-1].get_position()
    lowest = min(t.get_window_extent().y0 for t in tall.texts if "\u2013" in t.get_text())
    assert bar.y1 * tall.bbox.height < lowest
    plt.close("all")


def _bonding_case(title, comparator, shift=0.0):
    symbols = ["C", "O", "H"]
    positions = np.array([[0.0, 0.0, 0.0], [1.9 + shift, 0.2, 0.3], [-0.6, 0.9, -0.2]])
    orders = {(1, 2): 0.4, (1, 3): 0.95}
    bond = {
        "atom_i": 1,
        "atom_j": 2,
        "label": "C1-O2",
        "distance_A": 1.94,
        "wbo": 0.41,
    }

    def state(name, subtitle=""):
        return {
            "title": name,
            "symbols": symbols,
            "positions": positions,
            "bond_orders": orders,
            "bonds": [bond],
            "subtitle": subtitle,
        }

    return {
        "title": title,
        "states": [
            state("Reactant"),
            state("Saddle", "1 negative; 0 unresolved\nmax |F| = 0.050 eV/Å"),
            state("Product"),
            state(comparator),
        ],
    }


def test_visible_bond_orders_follow_the_edge_rule():
    from chemparseplot.plot.reaction import visible_bond_orders

    symbols = ["C", "H", "H"]
    positions = np.array([[0.0, 0, 0], [1.09, 0, 0], [-3.85, 0, 0]])
    kept = visible_bond_orders(
        symbols, positions, {(1, 2): 0.98, (1, 3): 0.75, (2, 3): 0.05}
    )
    assert kept == [(0, 1, 0.98)]
    assert visible_bond_orders(symbols, positions, {(1, 3): 0.75}, max_contact=4.0) == [
        (0, 2, 0.75)
    ]


def test_bonding_atlas_is_vector_with_one_scale_per_row():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import AxesImage

    from chemparseplot.plot.molecule import points_per_unit
    from chemparseplot.plot.reaction import plot_bonding_atlas

    cases = [_bonding_case("first", "Method A"), _bonding_case("second", "Method B", 1.5)]
    fig = plot_bonding_atlas(cases, width=6.0, note_size=7.0)
    assert fig.get_figwidth() == 6.0
    # eight structures, the key overlay and the colour bar; no raster panel
    structures = [ax for ax in fig.axes if ax.patches and not ax.axison][:8]
    assert len(fig.axes) == 10
    assert not any(isinstance(c, AxesImage) for ax in fig.axes for c in ax.get_children())
    scales = [points_per_unit(ax) for ax in structures]
    assert scales[:4] == pytest.approx([scales[0]] * 4)
    assert scales[4:] == pytest.approx([scales[4]] * 4)
    assert scales[4] < scales[0]
    texts = [t.get_text() for t in fig.texts]
    # shared column titles are printed once, the differing column once per row
    assert texts.count("Reactant") == texts.count("Saddle") == texts.count("Product") == 1
    assert texts.count("Method A") == texts.count("Method B") == 1
    assert texts.count("C1-O2: 1.94 Å / 0.41") == 8
    assert "max |F| = 0.050 eV/Å" in texts
    assert min(t.get_fontsize() for t in fig.texts) >= 7.0
    labels = [t for ax in structures for t in ax.texts]
    assert {t.get_text() for t in labels} == {"C1", "O2"}
    assert min(t.get_fontsize() for t in labels) >= 6.5
    plt.close(fig)
    with pytest.raises(ValueError, match="same number"):
        plot_bonding_atlas([cases[0], {"title": "x", "states": cases[1]["states"][:3]}])


def test_bonding_atlas_scale_label_clears_the_ticks_and_the_rows():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    from chemparseplot.plot.reaction import plot_bonding_atlas

    fig = plot_bonding_atlas([_bonding_case("only", "Method A")], width=6.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bar = fig.axes[-1]
    label = next(t for t in fig.texts if t.get_text() == "Wiberg bond order")
    label_box = label.get_window_extent(renderer)
    ticks = [t.get_window_extent(renderer) for t in bar.get_xticklabels()]
    assert [t.get_text() for t in bar.get_xticklabels()] == [
        "0",
        "0.5",
        "1",
        "1.5",
        "2",
        "2.5",
        "3",
    ]
    assert not any(label_box.overlaps(t) for t in ticks)
    # the key sits below the lowest annotation of the rows
    notes = [t for t in fig.texts if " Å / " in t.get_text()]
    lowest = min(t.get_window_extent(renderer).y0 for t in notes)
    assert bar.get_window_extent(renderer).y1 < lowest
    assert label_box.y1 < lowest
    plt.close(fig)


def test_bonding_atlas_moves_row_headers_clear_of_a_long_title():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    from chemparseplot.plot.reaction import plot_bonding_atlas

    other = _bonding_case("second", "Method B")
    short = plot_bonding_atlas([_bonding_case("row", "Method A"), other], width=6.0)
    long_title = "a reaction title long enough to run under the last column " * 2
    tall = plot_bonding_atlas([_bonding_case(long_title, "Method A"), other], width=6.0)
    assert tall.get_figheight() > short.get_figheight()
    fig_h = tall.get_figheight() * tall.dpi
    tall.canvas.draw()
    renderer = tall.canvas.get_renderer()
    title = next(t for t in tall.texts if t.get_text().startswith("(a)"))
    name = next(t for t in tall.texts if t.get_text() == "Method A")
    assert not title.get_window_extent(renderer).overlaps(
        name.get_window_extent(renderer)
    )
    assert 0 < name.get_window_extent(renderer).y0 < fig_h
    plt.close("all")
