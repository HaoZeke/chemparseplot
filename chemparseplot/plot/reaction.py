"""Molecular reaction panels from retained coordinates and xTB properties."""

from __future__ import annotations

from pathlib import Path

import numpy as np

WBO_COLUMNS = 3
MIN_BOND_CHANGE_A = 0.25
COMPARATOR_COLUMN = 3


def align_reaction(atoms_list, reference=1):
    """Keep atom identities and apply proper rigid rotations into one view."""
    from ase.build import minimize_rotation_and_translation  # noqa: PLC0415

    ref = atoms_list[reference].copy()
    ref.pbc = False
    ref.positions -= ref.positions.mean(axis=0)
    _, _, axes = np.linalg.svd(ref.positions, full_matrices=False)
    if np.linalg.det(axes) < 0:
        axes[-1] *= -1
    result = []
    for atoms in atoms_list:
        if not np.array_equal(atoms.numbers, ref.numbers):
            msg = "Reaction frames must have matching atom identities and order"
            raise ValueError(msg)
        frame = atoms.copy()
        frame.pbc = False
        minimize_rotation_and_translation(ref, frame)
        frame.positions = frame.positions @ axes.T
        if not np.allclose(
            atoms.get_all_distances(), frame.get_all_distances(), rtol=1e-12, atol=1e-10
        ):
            msg = "Molecular alignment must preserve every interatomic distance"
            raise ValueError(msg)
        result.append(frame)
    return result


def read_wbo(path, natoms):
    """Parse xTB's sparse, one-based Wiberg bond-order table.

    Missing entries remain missing: a print threshold is not a measured zero.
    """
    bonds = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != WBO_COLUMNS:
            msg = "Expected two atom indices and a Wiberg bond order"
            raise ValueError(msg)
        i, j = int(fields[0]), int(fields[1])
        value = float(fields[2].replace("D", "E"))
        key = tuple(sorted((i, j)))
        if (
            not (1 <= i <= natoms and 1 <= j <= natoms)
            or i == j
            or not np.isfinite(value)
        ):
            msg = "Invalid Wiberg bond-order entry"
            raise ValueError(msg)
        if key in bonds:
            msg = "Duplicate Wiberg bond-order pair"
            raise ValueError(msg)
        bonds[key] = value
    return bonds


def changed_bonds(reactant, product, limit=3):
    """Largest endpoint distance changes among initially or finally close pairs."""
    if not np.array_equal(reactant.numbers, product.numbers):
        msg = "Endpoint atom order differs"
        raise ValueError(msg)
    from ase.data import covalent_radii  # noqa: PLC0415

    dr, dp = reactant.get_all_distances(), product.get_all_distances()
    pairs = []
    for i in range(len(reactant)):
        for j in range(i + 1, len(reactant)):
            cutoff = 1.35 * (
                covalent_radii[reactant.numbers[i]] + covalent_radii[reactant.numbers[j]]
            )
            change = abs(dr[i, j] - dp[i, j])
            if min(dr[i, j], dp[i, j]) <= cutoff and change >= MIN_BOND_CHANGE_A:
                pairs.append((change, i + 1, j + 1))
    return [(i, j) for _, i, j in sorted(pairs, reverse=True)[:limit]]


def apply_wbo_style(molecule, bond_orders, *, minimum=0.1, weak=0.5, vmax=3.0):
    """Replace geometric connectivity with measured Wiberg bond orders.

    Atom numbering follows the source file. Dotted edges encode WBO below
    ``weak``; colour and width use a common scale across all structures.
    """
    from matplotlib import colormaps  # noqa: PLC0415
    from matplotlib.colors import to_hex  # noqa: PLC0415

    result = molecule.copy()
    graph = result.graph
    nodes = sorted(graph.nodes)
    if nodes not in (list(range(len(nodes))), list(range(1, len(nodes) + 1))):
        msg = "Renderer atom indices must follow contiguous source-file order"
        raise ValueError(msg)
    for (i, j), value in bond_orders.items():
        if (
            not (1 <= i <= len(nodes) and 1 <= j <= len(nodes))
            or i == j
            or not np.isfinite(value)
        ):
            msg = "Invalid Wiberg bond-order index or value"
            raise ValueError(msg)
    graph.remove_edges_from(list(graph.edges))
    palette = colormaps["viridis"]
    for (i, j), value in bond_orders.items():
        if value < minimum:
            continue
        graph.add_edge(
            nodes[i - 1],
            nodes[j - 1],
            bond_order=value,
            NCI=value < weak,
            bond_color_override=to_hex(palette(np.clip(value / vmax, 0, 1))),
            bond_width_override=2.0 + 3.0 * min(value, vmax),
        )
    return result


def render_reaction_state(
    xyz,
    output,
    *,
    cube=None,
    charge=0,
    multiplicity=1,
    iso=0.02,
    bond_orders=None,
    labels=None,
):
    """Render a fixed common orientation; density is a qualitative surface."""
    import xyzrender as xr  # noqa: PLC0415

    mol = xr.load(str(cube or xyz), charge=charge, multiplicity=multiplicity, quick=True)
    if bond_orders is not None:
        mol = apply_wbo_style(mol, bond_orders)
    kwargs = {
        "config": "paton",
        "hy": True,
        "bo": False,
        "orient": False,
        "transparent": True,
        "canvas_size": 700,
        "labels": labels,
        "label_font_size": 22,
        "output": str(output),
    }
    if cube is not None:
        kwargs.update(dens=True, iso=iso, opacity=0.3, dens_color="teal")
    xr.render(mol, **kwargs)


def plot_reaction_atlas(cases, *, density=False, width=7.2):
    """Render images and measured bond annotations in aligned reaction rows."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from chemparseplot.plot.neb import _crop_transparent_rgba  # noqa: PLC0415

    row_height = 2.65
    height = row_height * len(cases) + 0.7
    fig = plt.figure(figsize=(width, height))
    left, right, gap = 0.025, 0.985, 0.02
    column_width = (right - left - 3 * gap) / 4
    for row, case in enumerate(cases):
        top = 1 - (0.08 + row * row_height) / height
        fig.text(
            left,
            top,
            f"{'abcdefghijklmnopqrstuvwxyz'[row]}) {case['title']}",
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        for col, state in enumerate(case["states"]):
            x = left + col * (column_width + gap)
            center = x + column_width / 2
            fig.text(
                center,
                top - 0.25 / height,
                state["title"],
                ha="center",
                va="top",
                fontsize=9,
                color="#a13b4c" if col == COMPARATOR_COLUMN else "#222222",
            )
            ax = fig.add_axes([x, top - 1.98 / height, column_width, 1.45 / height])
            img = plt.imread(state["density_image" if density else "image"])
            ax.imshow(_crop_transparent_rgba(img))
            ax.axis("off")
            fig.text(
                center,
                top - 2.02 / height,
                state.get("subtitle", ""),
                ha="center",
                va="top",
                fontsize=7,
            )
            lines = []
            for bond in state["bonds"]:
                wbo = "n/r" if bond["wbo"] is None else f"{bond['wbo']:.2f}"
                lines.append(f"{bond['label']}: {bond['distance_A']:.2f} Å / {wbo}")
            fig.text(
                center,
                top - 2.33 / height,
                "\n".join(lines),
                ha="center",
                va="top",
                fontsize=6.5,
            )
    from matplotlib.cm import ScalarMappable  # noqa: PLC0415
    from matplotlib.colors import Normalize  # noqa: PLC0415

    cax = fig.add_axes([0.30, 0.065, 0.40, 0.025])
    cbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 3), cmap="viridis"),
        cax=cax,
        orientation="horizontal",
    )
    cbar.set_label("Wiberg bond order", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.text(
        0.5,
        0.005,
        "Dotted: WBO < 0.5; omitted: WBO < 0.1; n/r: not reported",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    return fig
