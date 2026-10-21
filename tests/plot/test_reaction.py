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
    graph = apply_wbo_style(original, {(1,2): .4, (1,3): 1.1, (2,3): .05}).graph
    a,b,c = sorted(graph.nodes)
    assert graph[a][b]["NCI"] is True
    assert graph[a][c]["NCI"] is False
    assert graph[a][b]["bond_color_override"] != graph[a][c]["bond_color_override"]
    assert graph[a][b]["bond_width_override"] < graph[a][c]["bond_width_override"]
    assert not graph.has_edge(b,c)
    assert "bond_color_override" not in original.graph[a][b]
