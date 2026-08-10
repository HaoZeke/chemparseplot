# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Pure-python coverage of the landscape descriptors and the merge tree.

The sketch-map binaries and featomic are external; everything here
exercises the descriptor math and the disconnectivity pipeline only.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from chemparseplot.plot.disconnectivity import (
    Node,
    dedup,
    iter_nodes,
    knn_graph,
    layout,
    merge_tree,
    plot_disconnectivity,
    spectral_basins,
)
from chemparseplot.plot.landscape import cn_matrix, median_sigma


def dimer_chain(n, spacing):
    """A 1D chain of n atoms as a flat coordinate row."""
    pos = np.zeros((n, 3))
    pos[:, 0] = np.arange(n) * spacing
    return pos.ravel()


def test_cn_matrix_permutation_invariant():
    rng = np.random.default_rng(3)
    pos = rng.normal(size=(8, 3))
    perm = rng.permutation(8)
    a = cn_matrix([pos.ravel()], ["X"] * 8, cutoff=1.5)
    b = cn_matrix([pos[perm].ravel()], ["X"] * 8, cutoff=1.5)
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_cn_matrix_species_restriction():
    pos = np.zeros((4, 3))
    pos[:, 0] = [0.0, 1.0, 5.0, 6.0]
    full = cn_matrix([pos.ravel()], ["O", "H", "O", "H"], cutoff=1.5)
    only_o = cn_matrix([pos.ravel()], ["O", "H", "O", "H"], cutoff=1.5, cn_species="O")
    assert full.shape[1] == 4
    assert only_o.shape[1] == 2


def test_cn_matrix_distinguishes_compact_from_stretched():
    tight = dimer_chain(6, 1.1)
    loose = dimer_chain(6, 3.0)
    m = cn_matrix([tight, loose], ["X"] * 6, cutoff=1.5)
    assert m[0].sum() > m[1].sum()


def test_median_sigma_positive():
    rng = np.random.default_rng(5)
    m = rng.normal(size=(50, 4))
    sigma = median_sigma(m, rng)
    assert sigma > 0


def test_dedup_keeps_lowest_energy_representative():
    m = np.array([[0.0], [0.01], [5.0]])
    es = np.array([2.0, 1.0, 3.0])
    reps = dedup(m, es, tol=0.5)
    assert 1 in reps  # the lower-energy twin survives
    assert 0 not in reps
    assert 2 in reps


def test_merge_tree_two_basins_merge_once():
    # Two well-separated pairs: components must merge only at the top.
    m = np.array([[0.0], [0.1], [10.0], [10.1]])
    es = np.array([-4.0, -3.0, -3.9, -2.9])
    adjacency = knn_graph(m, k=1)
    levels = np.linspace(es.min(), es.max() + 1.0, 20)[1:]
    root = merge_tree(adjacency, es, levels)
    members = sorted(n.leaf for n in iter_nodes(root) if n.leaf is not None)
    assert members == [0, 1, 2, 3]
    assert root.emin == -4.0
    layout(root)
    xs = [n.x for n in iter_nodes(root) if n.leaf is not None]
    assert sorted(xs) == [0.0, 1.0, 2.0, 3.0]


def test_merge_tree_deep_chain_no_recursion():
    n = 3000
    m = np.arange(n, dtype=float).reshape(-1, 1)
    es = np.linspace(-10.0, -1.0, n)
    adjacency = knn_graph(m, k=2)
    levels = np.linspace(es.min(), es.max(), 40)[1:]
    root = merge_tree(adjacency, es, levels)
    layout(root)
    assert len([x for x in iter_nodes(root) if x.leaf is not None]) == n


def test_spectral_basins_shapes():
    rng = np.random.default_rng(7)
    m = np.vstack([rng.normal(0, 0.1, size=(10, 2)), rng.normal(5, 0.1, size=(10, 2))])
    adjacency = knn_graph(m, k=3)
    assign = spectral_basins(adjacency, 2, rng, matrix=m)
    assert assign.shape == (20,)
    # The two clouds separate into distinct basins.
    assert len({int(a) for a in assign[:10]}) == 1
    assert assign[0] != assign[-1]


def test_plot_disconnectivity_end_to_end():
    rng = np.random.default_rng(9)
    m = np.vstack([rng.normal(0, 0.2, size=(15, 3)), rng.normal(4, 0.2, size=(15, 3))])
    es = np.concatenate([rng.uniform(-10, -8, size=15), rng.uniform(-9, -7, size=15)])
    fig, ax = plt.subplots()
    reps, root = plot_disconnectivity(ax, m, es, dedup_tol=0.01, knn=3)
    assert len(reps) <= 30
    assert root.emin == pytest.approx(es.min())
    plt.close(fig)


def test_node_tracks_argmin():
    a = Node(-5.0, leaf=0)
    b = Node(-7.0, leaf=1)
    parent = Node(-4.0, children=[a, b])
    assert parent.emin == -7.0
    assert parent.argmin == 1
