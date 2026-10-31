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
