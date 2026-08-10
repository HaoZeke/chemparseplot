# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Approximate disconnectivity graphs from minima databases.

The classic disconnectivity graph needs transition states; a hopping
campaign records only quenched minima. These helpers draw the superbasin
merge tree of a descriptor-space k-nearest-neighbour graph instead: at each
energy level the minima below it split into connected components, and
components merge as the level rises. That approximation is a rendering of
the database made for reading the figure; it should never feed a reported
statistic.

Basins can be lumped for display by spectral clustering on the same graph.
Everything is iterative: with a thousand minima the merge chains grow
deeper than any recursion limit.

```{versionadded} 1.10.0
```
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, laplacian

from chemparseplot.plot.theme import RUHI_COLORS

BASIN_PALETTE = [
    RUHI_COLORS[c] for c in ["teal", "sky", "magenta", "coral", "sunshine"]
]


def dedup(matrix, energies, tol):
    """Merge near-identical minima, keeping the lowest-energy representative.

    Returns the kept row indices, lowest energy first.

    ```{versionadded} 1.10.0
    ```
    """
    order = np.argsort(energies)
    rep_idx = []
    for i in order:
        if any(np.linalg.norm(matrix[i] - matrix[j]) < tol for j in rep_idx):
            continue
        rep_idx.append(i)
    return np.array(rep_idx)


def knn_graph(matrix, k=6):
    """Symmetric k-nearest-neighbour adjacency in descriptor space.

    ```{versionadded} 1.10.0
    ```
    """
    d = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    n = len(matrix)
    rows, cols = [], []
    for i in range(n):
        for j in np.argsort(d[i])[:k]:
            rows += [i, j]
            cols += [j, i]
    data = np.ones(len(rows))
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def bridge_components(adjacency, matrix, weight=0.1):
    """Weakly connect graph components through their closest point pairs.

    Spectral clustering assumes a connected graph; a k-NN graph over a
    minima database often fragments. One weak edge per component pair
    (closest points in descriptor space) restores the geometry without
    drowning the strong intra-basin structure.

    ```{versionadded} 1.10.0
    ```
    """
    n_comp, labels = connected_components(adjacency, directed=False)
    if n_comp == 1:
        return adjacency
    adjacency = adjacency.tolil()
    for a in range(n_comp):
        for b in range(a + 1, n_comp):
            ia = np.where(labels == a)[0]
            ib = np.where(labels == b)[0]
            d = np.linalg.norm(
                matrix[ia][:, None, :] - matrix[ib][None, :, :], axis=-1
            )
            i, j = np.unravel_index(int(np.argmin(d)), d.shape)
            adjacency[ia[i], ib[j]] = weight
            adjacency[ib[j], ia[i]] = weight
    return adjacency.tocsr()


def spectral_basins(adjacency, n_basins, rng, matrix=None):
    """Display-only basin assignment by spectral clustering on the graph.

    With ``matrix`` given, disconnected components are weakly bridged
    through their closest descriptor pairs first.

    ```{versionadded} 1.10.0
    ```
    """
    if matrix is not None:
        adjacency = bridge_components(adjacency, matrix)
    lap = laplacian(adjacency.astype(float), normed=True)
    k = min(n_basins, adjacency.shape[0] - 1)
    # Dense eigendecomposition: minima databases stay around a thousand
    # nodes, and iterative smallest-magnitude solvers are unreliable on
    # the near-singular Laplacians of disconnected graphs.
    _, vecs = np.linalg.eigh(np.asarray(lap.todense()))
    x = vecs[:, :k]
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    # Farthest-point seeding: deterministic given the first pick, and it
    # cannot drop both centers into one embedded cloud.
    first = int(rng.integers(len(x)))
    centers = [x[first]]
    while len(centers) < k:
        d = np.min(
            [np.linalg.norm(x - c, axis=1) for c in centers], axis=0
        )
        centers.append(x[int(np.argmax(d))])
    centers = np.array(centers)
    for _ in range(60):
        assign = np.argmin(
            np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=-1), axis=1
        )
        new = np.array(
            [
                x[assign == c].mean(axis=0) if (assign == c).any() else centers[c]
                for c in range(k)
            ]
        )
        if np.allclose(new, centers):
            break
        centers = new
    return assign


class Node:
    """A merge-tree node; leaves carry their minimum's index.

    ```{versionadded} 1.10.0
    ```
    """

    __slots__ = ("argmin", "children", "emin", "leaf", "level", "x")

    def __init__(self, level, children=None, leaf=None):
        self.level = level
        self.children = children or []
        self.leaf = leaf
        self.x = None
        if self.leaf is not None:
            self.emin = level
            self.argmin = leaf
        else:
            self.emin = min(c.emin for c in self.children)
            self.argmin = min(self.children, key=lambda c: c.emin).argmin


def merge_tree(adjacency, energies, levels):
    """Superbasin merge tree over an energy-threshold sweep.

    ```{versionadded} 1.10.0
    ```
    """
    n = len(energies)
    leaves = [Node(energies[i], leaf=i) for i in range(n)]
    active = {}
    prev_sets = []
    for lev in levels:
        mask = energies <= lev
        if not mask.any():
            continue
        sub = np.where(mask)[0]
        _, labels = connected_components(
            adjacency[sub][:, sub], directed=False
        )
        comps = {}
        for local, g in enumerate(sub):
            comps.setdefault(labels[local], []).append(g)
        new_sets = [frozenset(v) for v in comps.values()]
        for s in new_sets:
            if s in active:
                continue
            parts = [p for p in prev_sets if p <= s and p in active]
            merged = [active[p] for p in parts]
            covered = frozenset().union(*parts) if parts else frozenset()
            for p in parts:
                del active[p]
            extra = sorted(s - covered, key=lambda i: energies[i])
            if not merged:
                node = leaves[extra[0]]
                for m in extra[1:]:
                    node = Node(
                        max(energies[m], node.level), children=[node, leaves[m]]
                    )
            else:
                node = merged[0] if len(merged) == 1 else Node(lev, children=merged)
                for m in extra:
                    node = Node(lev, children=[node, leaves[m]])
            active[s] = node
        prev_sets = new_sets
    roots = list(active.values())
    if len(roots) > 1:
        return Node(levels[-1], children=roots)
    return roots[0]


def iter_nodes(root):
    """Post-order traversal without recursion.

    ```{versionadded} 1.10.0
    ```
    """
    out, stack = [], [(root, False)]
    while stack:
        node, done = stack.pop()
        if done or node.leaf is not None:
            out.append(node)
            continue
        stack.append((node, True))
        for c in reversed(node.children):
            stack.append((c, False))
    return out


def layout(root):
    """Leaf x positions in tree order, children sorted deepest-first.

    ```{versionadded} 1.10.0
    ```
    """
    for node in iter_nodes(root):
        if node.leaf is None:
            node.children.sort(key=lambda c: c.emin)
    x = 0.0
    for node in iter_nodes(root):
        if node.leaf is not None:
            node.x = x
            x += 1.0
        else:
            node.x = float(np.mean([c.x for c in node.children]))


def paint(root, assign, palette=None):
    """Colour each subtree by the spectral basin of its deepest minimum.

    ```{versionadded} 1.10.0
    ```
    """
    palette = palette or BASIN_PALETTE
    colors = {}
    for node in iter_nodes(root):
        colors[id(node)] = palette[int(assign[node.argmin]) % len(palette)]
    return colors


def draw(ax, root, colors, lw=1.1):
    """Vertical stems to the merge level, horizontal bars at merges.

    ```{versionadded} 1.10.0
    ```
    """
    for node in iter_nodes(root):
        if node.leaf is not None:
            continue
        for c in node.children:
            ax.plot(
                [c.x, c.x],
                [c.level, node.level],
                color=colors.get(id(c), "#444444"),
                lw=lw,
                solid_capstyle="round",
                zorder=2,
            )
        xs = [c.x for c in node.children]
        ax.plot(
            [min(xs), max(xs)],
            [node.level, node.level],
            color="#444444",
            lw=0.9,
            zorder=1,
        )


def plot_disconnectivity(
    ax,
    matrix,
    energies,
    *,
    dedup_tol=0.15,
    knn=6,
    n_levels=48,
    n_basins=5,
    rng=None,
):
    """Full pipeline: dedup, graph, merge tree, spectral paint, draw.

    Returns ``(kept_indices, root)`` so callers can label leaves.

    ```{versionadded} 1.10.0
    ```
    """
    rng = rng or np.random.default_rng(11)
    reps = dedup(matrix, energies, dedup_tol)
    m = matrix[reps]
    en = energies[reps]
    adjacency = knn_graph(m, knn)
    levels = np.linspace(en.min(), en.max(), n_levels)[1:]
    root = merge_tree(adjacency, en, levels)
    layout(root)
    assign = spectral_basins(adjacency, n_basins, rng, matrix=m)
    colors = paint(root, assign)
    draw(ax, root, colors)
    ax.set_xticks([])
    ax.set_ylabel("quenched energy")
    ax.spines["bottom"].set_visible(False)
    return reps, root
