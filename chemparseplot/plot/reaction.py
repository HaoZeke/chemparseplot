"""Molecular reaction panels from retained coordinates and xTB properties."""

from __future__ import annotations

from pathlib import Path

import numpy as np

WBO_COLUMNS = 3
MIN_BOND_CHANGE_A = 0.25
COMPARATOR_COLUMN = 3


def align_reaction(atoms_list, reference=1, *, weights=None):
    """Keep atom identities and apply proper rigid rotations into one view.

    Every frame is fitted onto the ``reference`` frame and then rotated into
    the principal axes of that frame, so the first two coordinates span its
    plane of largest extent. ``weights`` chooses the fit: ``None`` weighs all
    atoms equally, ``"mass"`` weighs each atom by its mass. With mass weights
    the heavy-atom frame stays put and a migrating hydrogen is seen to move.

    ```{versionchanged} 1.14.0
    Added ``weights``.
    ```
    """
    from ase.build import minimize_rotation_and_translation  # noqa: PLC0415

    if weights not in (None, "mass"):
        msg = f"Unknown alignment weights {weights!r}; use None or 'mass'"
        raise ValueError(msg)
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
        if weights == "mass":
            frame.positions = _weighted_fit(
                ref.positions, frame.positions, ref.get_masses()
            )
        else:
            minimize_rotation_and_translation(ref, frame)
        frame.positions = frame.positions @ axes.T
        if not np.allclose(
            atoms.get_all_distances(), frame.get_all_distances(), rtol=1e-12, atol=1e-10
        ):
            msg = "Molecular alignment must preserve every interatomic distance"
            raise ValueError(msg)
        result.append(frame)
    return result


def _weighted_fit(target, positions, weights):
    """Proper rotation and translation of ``positions`` onto ``target`` (Kabsch).

    The weighted centroids coincide afterwards and the weighted squared
    deviation is least among proper rotations.
    """
    w = np.asarray(weights, dtype=float)[:, None]
    centre_t = (w * target).sum(axis=0) / w.sum()
    centre_p = (w * positions).sum(axis=0) / w.sum()
    covariance = (w * (positions - centre_p)).T @ (target - centre_t)
    left, _, right = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(left @ right)) or 1.0
    rotation = left @ np.diag([1.0, 1.0, sign]) @ right
    return (positions - centre_p) @ rotation + centre_t


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


def apply_wbo_style(
    molecule, bond_orders, *, minimum=0.1, weak=0.5, vmax=3.0, max_contact=2.5
):
    """Replace geometric connectivity with measured Wiberg bond orders.

    Atom numbering follows the source file. Dotted edges encode WBO below
    ``weak``; colour and width use a common scale across all structures.
    A pair farther apart than ``max_contact`` times its covalent-radius sum
    gets no edge: a closed-shell density at a near-degenerate geometry can
    assign bond order to separated radical centres, which is not a bond.
    """
    from ase.data import covalent_radii  # noqa: PLC0415
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
        a, b = graph.nodes[nodes[i - 1]], graph.nodes[nodes[j - 1]]
        if "position" in a and "position" in b:
            reach = max_contact * (
                covalent_radii[a["atomic_number"]] + covalent_radii[b["atomic_number"]]
            )
            if np.linalg.norm(np.subtract(a["position"], b["position"])) > reach:
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
    from matplotlib import font_manager  # noqa: PLC0415
    from resvg_py import svg_to_bytes  # noqa: PLC0415

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
    }
    if cube is not None:
        kwargs.update(dens=True, iso=iso, opacity=0.3, dens_color="teal")
    svg = str(xr.render(mol, **kwargs))
    Path(output).with_suffix(".svg").write_text(svg)
    font = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans Mono", weight="bold")
    )
    Path(output).write_bytes(
        svg_to_bytes(
            svg_string=svg,
            zoom=3,
            font_files=[font],
            skip_system_fonts=True,
            font_family="DejaVu Sans Mono",
            monospace_family="DejaVu Sans Mono",
        )
    )


def plot_reaction_atlas(cases, *, density=False, width=7.2):
    """Render images and measured bond annotations in aligned reaction rows."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from chemparseplot.plot.neb import _crop_transparent_rgba  # noqa: PLC0415

    line_height = 0.13
    row_heights = [
        2.4 + line_height * max(len(s["bonds"]) for s in case["states"]) for case in cases
    ]
    margin = 0.75
    height = sum(row_heights) + margin
    fig = plt.figure(figsize=(width, height))
    left, right, gap = 0.025, 0.985, 0.02
    column_width = (right - left - 3 * gap) / 4
    offsets = np.concatenate([[0.0], np.cumsum(row_heights)[:-1]])
    for row, case in enumerate(cases):
        top = 1 - (0.08 + offsets[row]) / height
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

    cax = fig.add_axes([0.30, 0.3 / height, 0.40, 0.12 / height])
    cbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 3), cmap="viridis"),
        cax=cax,
        orientation="horizontal",
    )
    cbar.set_label("Wiberg bond order", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return fig


def plot_reaction_gallery(tiles, *, columns=4, density=False, width=7.2):
    """Render one structure per reaction in a grid of labelled tiles.

    Each tile holds a ``title`` (the name other figures use for the
    reaction), an optional ``subtitle`` naming the chemistry, the rendered
    ``image`` (or ``density_image``) and the measured ``bonds`` to annotate,
    in the format of :func:`plot_reaction_atlas`.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.cm import ScalarMappable  # noqa: PLC0415
    from matplotlib.colors import Normalize  # noqa: PLC0415

    from chemparseplot.plot.neb import _crop_transparent_rgba  # noqa: PLC0415

    if columns < 1:
        msg = "A reaction gallery needs at least one column"
        raise ValueError(msg)
    rows = -(-len(tiles) // columns)
    tile_height = 2.35
    height = tile_height * rows + 0.75
    fig = plt.figure(figsize=(width, height))
    left, right, gap = 0.02, 0.98, 0.015
    column_width = (right - left - (columns - 1) * gap) / columns
    for index, tile in enumerate(tiles):
        row, col = divmod(index, columns)
        top = 1 - (0.08 + row * tile_height) / height
        x = left + col * (column_width + gap)
        center = x + column_width / 2
        fig.text(
            center,
            top,
            tile["title"],
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
        )
        if tile.get("subtitle"):
            fig.text(
                center,
                top - 0.2 / height,
                tile["subtitle"],
                ha="center",
                va="top",
                fontsize=6.5,
                color="#444444",
            )
        ax = fig.add_axes([x, top - 1.72 / height, column_width, 1.3 / height])
        img = plt.imread(tile["density_image" if density else "image"])
        ax.imshow(_crop_transparent_rgba(img))
        ax.axis("off")
        lines = []
        for bond in tile.get("bonds", []):
            wbo = "n/r" if bond["wbo"] is None else f"{bond['wbo']:.2f}"
            lines.append(f"{bond['label']}: {bond['distance_A']:.2f} Å / {wbo}")
        fig.text(
            center,
            top - 1.76 / height,
            "\n".join(lines),
            ha="center",
            va="top",
            fontsize=6,
        )
    cax = fig.add_axes([0.30, 0.18 / height, 0.40, 0.12 / height])
    cbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 3), cmap="viridis"),
        cax=cax,
        orientation="horizontal",
    )
    cbar.set_label("Wiberg bond order", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return fig


def visible_bond_orders(symbols, positions, bond_orders, *, minimum=0.1, max_contact=2.5):
    """Zero-based ``(i, j, order)`` for the Wiberg pairs a drawing keeps.

    The rule is that of :func:`apply_wbo_style`: an order below ``minimum``
    or a pair farther apart than ``max_contact`` covalent-radius sums gets no
    edge. ``bond_orders`` uses the one-based keys of :func:`read_wbo`.
    """
    from ase.data import atomic_numbers, covalent_radii  # noqa: PLC0415

    positions = np.asarray(positions, dtype=float)
    radii = [covalent_radii[atomic_numbers[s]] for s in symbols]
    kept = []
    for (i, j), value in sorted(bond_orders.items()):
        a, b = i - 1, j - 1
        if value < minimum:
            continue
        if np.linalg.norm(positions[a] - positions[b]) > max_contact * (
            radii[a] + radii[b]
        ):
            continue
        kept.append((a, b, float(value)))
    return kept


def bond_annotation(bond):
    """One line of a bond annotation: label, distance and Wiberg order."""
    wbo = "n/r" if bond["wbo"] is None else f"{bond['wbo']:.2f}"
    return f"{bond['label']}: {bond['distance_A']:.2f} Å / {wbo}"


def plot_bonding_atlas(
    cases,
    *,
    width=7.0,
    style=None,
    cmap="ruhi_bond_order",
    vmax=3.0,
    title_size=9.0,
    header_size=8.0,
    note_size=7.0,
    max_scale=20.0,
    structure_aspect=0.72,
    highlight_column=COMPARATOR_COLUMN,
    highlight_color="#B8463F",
    text_color="#1C2421",
):
    """Vector reaction rows drawn by one renderer, one scale per row.

    Each case holds a ``title`` and its ``states``. A state holds ``title``,
    ``symbols``, ``positions`` (aligned, drawing plane in the first two
    columns), ``bond_orders`` (the mapping :func:`read_wbo` returns), the
    ``bonds`` to annotate and an optional ``subtitle`` printed above them.
    Atoms of the annotated bonds are numbered unless the state gives its own
    ``labels`` (zero-based index to text).

    Every structure of a row shares one scale in points per angstrom, so the
    comparator can be compared with the saddle by eye. A column whose state
    titles agree in every row is headed once, above the first row; the other
    columns are named on each row's title line, or on a line of their own
    when the row title reaches them. Text sizes are in points at the given
    ``width``; include the figure at that width to keep them. An element key,
    the bond-order scale and the dotted-bond note share one line below the
    rows, with the scale label beside the bar and the ticks beneath it.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from ase.data import atomic_numbers  # noqa: PLC0415
    from matplotlib import colormaps  # noqa: PLC0415
    from matplotlib.cm import ScalarMappable  # noqa: PLC0415
    from matplotlib.colors import Normalize  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415
    from matplotlib.patches import Circle  # noqa: PLC0415

    from chemparseplot.plot.molecule import (  # noqa: PLC0415
        MoleculeStyle,
        atom_radii,
        draw_molecule,
        frame_structure,
        text_extent,
    )

    style = style or MoleculeStyle()
    cmap = colormaps[cmap] if isinstance(cmap, str) else cmap
    norm = Normalize(0, vmax)
    columns = len(cases[0]["states"])
    if any(len(case["states"]) != columns for case in cases):
        msg = "Every reaction row needs the same number of states"
        raise ValueError(msg)
    total = width * 72
    side, gutter, leading = 2.0, 5.0, 1.24
    column = (total - 2 * side - (columns - 1) * gutter) / columns
    title_h, header_h, note_h = (
        leading * s for s in (title_size, header_size, note_size)
    )

    def centre_of(col):
        return side + col * (column + gutter) + column / 2

    def labels_of(state):
        if "labels" in state:
            return dict(state["labels"])
        found = {}
        for bond in state["bonds"]:
            for key in ("atom_i", "atom_j"):
                if key in bond:
                    index = bond[key] - 1
                    found[index] = f"{state['symbols'][index]}{index + 1}"
        return found

    def notes_of(state):
        head = [line for line in state.get("subtitle", "").split("\n") if line]
        return head + [bond_annotation(b) for b in state["bonds"]]

    shared = [
        len({case["states"][col]["title"] for case in cases}) == 1
        for col in range(columns)
    ]
    rows = []
    for index, case in enumerate(cases):
        drawn = []
        for state in case["states"]:
            positions = np.asarray(state["positions"], dtype=float)
            radii = atom_radii(state["symbols"], style.radius_scale)[:, None]
            labels = labels_of(state)
            drawn.append(
                {
                    "state": state,
                    "positions": positions,
                    "lo": (positions[:, :2] - radii).min(axis=0),
                    "hi": (positions[:, :2] + radii).max(axis=0),
                    "labels": labels,
                    "notes": notes_of(state),
                }
            )
        tallest = max(
            (
                text_extent(t, style.label_size)[1]
                for d in drawn
                for t in d["labels"].values()
            ),
            default=0.0,
        )
        pad = tallest + style.label_gap + style.label_halo
        span = np.max([d["hi"] - d["lo"] for d in drawn], axis=0)
        scale = min(
            max_scale,
            (column - 2 * pad) / span[0],
            (structure_aspect * column - 2 * pad) / span[1],
        )
        title = f"({'abcdefghijklmnopqrstuvwxyz'[index]}) {case['title']}"
        own = [col for col in range(columns) if not shared[col]]
        reach = side + _bold_extent(title, title_size)
        clear = all(
            reach + gutter
            < centre_of(col)
            - text_extent(drawn[col]["state"]["title"], header_size)[0] / 2
            for col in own
        )
        rows.append(
            {
                "title": title,
                "drawn": drawn,
                "scale": scale,
                "box": span[1] * scale + 2 * pad,
                "lines": max(len(d["notes"]) for d in drawn),
                "own": own,
                "own_line": bool(own) and not clear,
            }
        )

    row_gap, bar_h = 0.7 * note_h, 0.8 * note_size
    key_h = bar_h + 2.0 + 1.4 * note_h
    height = (
        (header_h if any(shared) else 0.0)
        + sum(
            title_h
            + (header_h if row["own_line"] else 0.0)
            + row["box"]
            + row["lines"] * note_h
            + row_gap
            for row in rows
        )
        + key_h
    )
    fig = plt.figure(figsize=(width, height / 72))

    def write(x, y, text, size, **kwargs):
        kwargs.setdefault("color", text_color)
        kwargs.setdefault("va", "baseline")
        return fig.text(x / total, y / height, text, fontsize=size, **kwargs)

    def header(col, y, text):
        write(
            centre_of(col),
            y,
            text,
            header_size,
            ha="center",
            color=highlight_color if col == highlight_column else text_color,
        )

    y = height
    if any(shared):
        y -= header_h
        for col in range(columns):
            if shared[col]:
                header(col, y + 0.3 * header_size, cases[0]["states"][col]["title"])
    for row in rows:
        y -= title_h
        write(side, y + 0.3 * title_size, row["title"], title_size, fontweight="bold")
        if row["own_line"]:
            y -= header_h
        for col in row["own"]:
            header(
                col,
                y + 0.3 * (header_size if row["own_line"] else title_size),
                row["drawn"][col]["state"]["title"],
            )
        y -= row["box"]
        for col, d in enumerate(row["drawn"]):
            state = d["state"]
            ax = frame_structure(
                fig,
                (side + col * (column + gutter), y, column, row["box"]),
                (d["lo"] + d["hi"]) / 2,
                row["scale"],
            )
            draw_molecule(
                ax,
                state["symbols"],
                d["positions"],
                visible_bond_orders(
                    state["symbols"], d["positions"], state["bond_orders"]
                ),
                labels=d["labels"],
                style=style,
                cmap=cmap,
                norm=norm,
            )
            for k, line in enumerate(d["notes"]):
                write(
                    centre_of(col),
                    y - (k + 1) * note_h + 0.3 * note_size,
                    line,
                    note_size,
                    ha="center",
                )
        y -= row["lines"] * note_h + row_gap

    # Key line: elements, the bond-order scale and the dotted-bond note.
    overlay = fig.add_axes((0, 0, 1, 1), zorder=-1)
    overlay.set_axis_off()
    overlay.set_xlim(0, total)
    overlay.set_ylim(0, height)
    bar_y = 1.4 * note_h + 2.0
    mid = bar_y + bar_h / 2
    baseline = mid - 0.36 * note_size
    present = {s for row in rows for d in row["drawn"] for s in d["state"]["symbols"]}
    x = side + 2.0
    disc = 0.42 * note_size
    for symbol in sorted(present, key=lambda s: atomic_numbers[s]):
        overlay.add_patch(
            Circle(
                (x + disc, mid),
                disc,
                facecolor=style.color_of(symbol),
                edgecolor=style.outline,
                linewidth=style.outline_width,
            )
        )
        write(x + 2 * disc + 2.0, baseline, symbol, note_size)
        x += 2 * disc + 2.0 + text_extent(symbol, note_size)[0] + 6.0
    label = "Wiberg bond order"
    note = f"dotted: order below {style.weak_order:g}"
    sample = 16.0
    bar_x0 = x + 10.0 + text_extent(label, note_size)[0] + 6.0
    bar_x1 = total - side - text_extent(note, note_size)[0] - sample - 4.0 - 12.0
    write(bar_x0 - 6.0, baseline, label, note_size, ha="right")
    cax = fig.add_axes(
        (bar_x0 / total, bar_y / height, (bar_x1 - bar_x0) / total, bar_h / height)
    )
    bar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal"
    )
    ticks = np.arange(0, vmax + 1e-9, 0.5)
    bar.set_ticks(ticks, labels=[f"{t:g}" for t in ticks])
    bar.ax.tick_params(
        axis="x", labelsize=note_size, length=2.0, width=0.5, pad=1.5, colors=text_color
    )
    bar.outline.set_linewidth(0.5)
    bar.outline.set_edgecolor(text_color)
    sample_x = bar_x1 + 12.0
    overlay.add_line(
        Line2D(
            [sample_x, sample_x + sample],
            [mid, mid],
            linewidth=style.weak_width,
            color=cmap(norm(0.6 * style.weak_order)),
            linestyle=(0, (0.1, 1.9)),
            dash_capstyle="round",
        )
    )
    write(sample_x + sample + 4.0, baseline, note, note_size)
    return fig


def _bold_extent(text, size):
    """Width in points of ``text`` set bold in the current font family."""
    from matplotlib.font_manager import FontProperties  # noqa: PLC0415
    from matplotlib.textpath import TextPath  # noqa: PLC0415

    prop = FontProperties(weight="bold")
    return float(TextPath((0, 0), text, size=size, prop=prop).get_extents().width)
