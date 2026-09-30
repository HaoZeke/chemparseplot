# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Primitive rings and the bridges between them, drawn with xyzrender.

The rings are the ones ``pydseams.yoda.ringNetwork`` returns for the neighbour
list this module builds. xyzrender can also paint ``hull="rings"`` from its
own bond perception. That detector is not used here. The hulls, the dashed
bridges, and the optional query markers are a picture of the list handed to
``ringNetwork``.

The neighbour list is not periodic. A distance cutoff is a number the caller
measures between the longest bond and the shortest nonbonded contact. Passing
the ice default of 3.5 angstrom builds a different graph. The command is
``rgpycrumbs geom plt-rings``, a PEP 723 script the rgpycrumbs dispatcher
runs under ``uv``. ``rgpycrumbs geom plt-rings-track`` applies the same
census to every frame of a trajectory. A hop is a change of terminal ring,
identified by its atom set. The integer ``ringNetwork`` returns for one
frame is not that identity. A query coordinate is the caller's Wannier
centre. The label is not the coordinate that enters a mean square
displacement.
"""

from __future__ import annotations

import csv
import math
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

# Okabe-Ito, cycled when a molecule has more faces than colours.
RING_COLOURS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)
JUNCTION_COLOUR = "#C0392B"


@dataclass(frozen=True)
class RingReport:
    """Rings and the complement of their edges, for one neighbour list."""

    n_atoms: int
    bonds: tuple[tuple[int, int], ...]
    rings: tuple[tuple[int, ...], ...]
    junctions: tuple[tuple[int, int], ...]
    pendants: tuple[tuple[int, int], ...]
    other: tuple[tuple[int, int], ...]
    depth_gap: tuple[tuple[int, int, int], ...]

    @property
    def census(self) -> dict[int, int]:
        """Ring counts keyed by size."""
        counts: dict[int, int] = {}
        for ring in self.rings:
            counts[len(ring)] = counts.get(len(ring), 0) + 1
        return dict(sorted(counts.items()))


@dataclass(frozen=True)
class Assignment:
    """Nearest ring centroid or junction midpoint for one query point."""

    query: int
    kind: str
    owner: int
    distance: float
    margin: float


def _import_pydseams():
    try:
        import pydseams as ds
    except ImportError as exc:
        msg = (
            "pydseamslib is required to enumerate primitive rings. "
            "Install the rings extra, or run with uv and a PEP 723 block "
            "that depends on pydseamslib."
        )
        raise RuntimeError(msg) from exc
    return ds


def _import_xyzrender():
    try:
        from rgpycrumbs._aux import enable_library_auto_deps, ensure_import

        enable_library_auto_deps()
        return ensure_import("xyzrender")
    except ImportError:
        pass
    try:
        import xyzrender
    except ImportError as exc:
        msg = (
            "xyzrender>=0.3.8 is required to draw primitive rings. "
            "Install with: pip install 'xyzrender>=0.3.8'"
        )
        raise RuntimeError(msg) from exc
    return xyzrender


def _as_edges(
    bonds: Sequence[tuple[int, int]],
    n_atoms: int,
) -> tuple[tuple[int, int], ...]:
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    for a, b in bonds:
        if a == b:
            msg = f"bond ({a}, {b}) is a self-loop"
            raise ValueError(msg)
        if not (0 <= a < n_atoms and 0 <= b < n_atoms):
            msg = f"bond ({a}, {b}) is outside 0..{n_atoms - 1}"
            raise ValueError(msg)
        edge = (a, b) if a < b else (b, a)
        if edge not in seen:
            seen.add(edge)
            edges.append(edge)
    return tuple(edges)


def _adjacency(n_atoms: int, edges: Sequence[tuple[int, int]]) -> list[set[int]]:
    adj = [set() for _ in range(n_atoms)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _nlist(adj: Sequence[set[int]]) -> list[list[int]]:
    """Row format ringNetwork reads: the vertex, then its neighbours."""
    return [[i, *sorted(nbrs)] for i, nbrs in enumerate(adj)]


def _girth(adj: list[set[int]], u: int, v: int) -> int | None:
    """Length of a shortest cycle through uv, or None when uv is a bridge."""
    adj[u].remove(v)
    adj[v].remove(u)
    try:
        dist = {u: 0}
        queue: deque[int] = deque([u])
        while queue:
            cur = queue.popleft()
            for nxt in adj[cur]:
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        if v not in dist:
            return None
        return dist[v] + 1
    finally:
        adj[u].add(v)
        adj[v].add(u)


def _ring_edges(rings: Sequence[Sequence[int]]) -> tuple[set[tuple[int, int]], set[int]]:
    covered: set[tuple[int, int]] = set()
    on_ring: set[int] = set()
    for ring in rings:
        on_ring.update(ring)
        size = len(ring)
        for i, vertex in enumerate(ring):
            a, b = vertex, ring[(i + 1) % size]
            covered.add((a, b) if a < b else (b, a))
    return covered, on_ring


def report_from_bonds(
    n_atoms: int,
    bonds: Sequence[tuple[int, int]],
    *,
    max_depth: int = 6,
) -> RingReport:
    """Enumerate primitive rings of an explicit neighbour list.

    ``max_depth`` is the largest ring ``ringNetwork`` generates. An edge
    that still lies on a cycle of length 12 or less, and that is absent
    from the returned rings, is recorded in ``depth_gap``. That absence is
    truncation. An edge with no cycle is a bridge. A junction is a bridge,
    or a truncated edge, whose two ends both lie on a returned ring.
    """
    edges = _as_edges(bonds, n_atoms)
    adj = _adjacency(n_atoms, edges)
    raw = _import_pydseams().yoda.ringNetwork(_nlist(adj), int(max_depth))
    rings: list[tuple[int, ...]] = []
    for ring in raw:
        verts = [int(v) for v in ring]
        if len(verts) >= 2 and verts[0] == verts[-1]:
            verts = verts[:-1]
        rings.append(tuple(verts))
    covered, on_ring = _ring_edges(rings)
    junctions: list[tuple[int, int]] = []
    pendants: list[tuple[int, int]] = []
    other: list[tuple[int, int]] = []
    depth_gap: list[tuple[int, int, int]] = []
    for edge in edges:
        if edge in covered:
            continue
        a, b = edge
        if a in on_ring and b in on_ring:
            junctions.append(edge)
        elif a in on_ring or b in on_ring:
            pendants.append(edge)
        else:
            other.append(edge)
        girth = _girth(adj, a, b)
        if girth is not None and girth <= 12:
            depth_gap.append((a, b, girth))
    return RingReport(
        n_atoms=n_atoms,
        bonds=edges,
        rings=tuple(rings),
        junctions=tuple(junctions),
        pendants=tuple(pendants),
        other=tuple(other),
        depth_gap=tuple(depth_gap),
    )


def bonds_within_cutoff(
    symbols: Sequence[str],
    coords: np.ndarray,
    cutoff: float,
) -> tuple[tuple[int, int], ...]:
    """Heavy-atom pairs at a positive distance of at most ``cutoff`` angstrom.

    Hydrogen is left out. On a thiophene the covalent heavy graph and the
    3.5 angstrom graph are different molecules, so the cutoff is an argument
    and not a default.
    """
    xyz = np.asarray(coords, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        msg = f"coords must have shape (n, 3), got {xyz.shape}"
        raise ValueError(msg)
    if len(symbols) != len(xyz):
        msg = f"{len(symbols)} symbols and {len(xyz)} coordinates"
        raise ValueError(msg)
    heavy = [i for i, el in enumerate(symbols) if el.upper() != "H"]
    limit = float(cutoff)
    edges: list[tuple[int, int]] = []
    for i, a in enumerate(heavy):
        for b in heavy[i + 1 :]:
            dist = float(np.linalg.norm(xyz[a] - xyz[b]))
            if 0.0 < dist <= limit:
                edges.append((a, b) if a < b else (b, a))
    return tuple(edges)


def ring_report(
    symbols: Sequence[str],
    coords: np.ndarray,
    *,
    bonds: Sequence[tuple[int, int]] | None = None,
    cutoff: float | None = None,
    max_depth: int = 6,
) -> RingReport:
    """Primitive rings of an explicit bond list or of a heavy-atom cutoff."""
    xyz = np.asarray(coords, dtype=float)
    if bonds is not None and cutoff is not None:
        msg = "pass bonds or cutoff, not both"
        raise ValueError(msg)
    if bonds is None and cutoff is None:
        msg = "pass bonds, or a cutoff inside the bond/nonbonded gap"
        raise ValueError(msg)
    if bonds is None:
        bonds = bonds_within_cutoff(symbols, xyz, float(cutoff))
    return report_from_bonds(len(symbols), bonds, max_depth=max_depth)


def _owners(
    coords: np.ndarray,
    report: RingReport,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    points: list[np.ndarray] = []
    labels: list[tuple[str, int]] = []
    for i, ring in enumerate(report.rings):
        points.append(coords[list(ring)].mean(axis=0))
        labels.append(("ring", i))
    for i, (a, b) in enumerate(report.junctions):
        points.append(0.5 * (coords[a] + coords[b]))
        labels.append(("junction", i))
    if not points:
        return np.zeros((0, 3)), labels
    return np.vstack(points), labels


def assign_points(
    coords: np.ndarray,
    report: RingReport,
    queries: np.ndarray,
) -> tuple[Assignment, ...]:
    """Label each query by the nearest ring centroid or junction midpoint.

    The label is a name for the centre. The centre's own coordinate is what
    enters a mean square displacement.
    """
    xyz = np.asarray(coords, dtype=float)
    pts = np.asarray(queries, dtype=float)
    if pts.size == 0:
        return ()
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    owners, labels = _owners(xyz, report)
    if len(labels) < 2:
        msg = "assignment needs at least two owners"
        raise ValueError(msg)
    out: list[Assignment] = []
    for q, point in enumerate(pts):
        dist = np.linalg.norm(owners - point, axis=1)
        order = np.argsort(dist)
        kind, owner = labels[int(order[0])]
        out.append(
            Assignment(
                query=q,
                kind=kind,
                owner=owner,
                distance=float(dist[order[0]]),
                margin=float(dist[order[1]] - dist[order[0]]),
            )
        )
    return tuple(out)


def read_structure(
    path: str | PathLike[str],
) -> tuple[list[str], np.ndarray, tuple[tuple[int, int], ...] | None]:
    """Read an XYZ (no bonds) or a V2000 SDF (bonds included)."""
    text = Path(path).read_text()
    suffix = Path(path).suffix.lower()
    if suffix == ".sdf" or "V2000" in text.splitlines()[3:4]:
        return _read_sdf(text)
    return _read_xyz(text)


def _read_xyz(
    text: str,
) -> tuple[list[str], np.ndarray, None]:
    lines = [line for line in text.splitlines() if line.strip()]
    n_atoms = int(lines[0].split()[0])
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(symbols) != n_atoms:
        msg = f"XYZ count {n_atoms} but read {len(symbols)} atoms"
        raise ValueError(msg)
    return symbols, np.asarray(coords, dtype=float), None


def _read_sdf(
    text: str,
) -> tuple[list[str], np.ndarray, tuple[tuple[int, int], ...]]:
    lines = text.splitlines()
    counts = lines[3].split()
    n_atoms, n_bonds = int(counts[0]), int(counts[1])
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[4 : 4 + n_atoms]:
        parts = line.split()
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
        symbols.append(parts[3])
    bonds: list[tuple[int, int]] = []
    for line in lines[4 + n_atoms : 4 + n_atoms + n_bonds]:
        parts = line.split()
        a, b = int(parts[0]) - 1, int(parts[1]) - 1
        bonds.append((a, b) if a < b else (b, a))
    return symbols, np.asarray(coords, dtype=float), tuple(bonds)


def _write_xyz(path: Path, symbols: Sequence[str], coords: np.ndarray) -> None:
    rows = [str(len(symbols)), "primitive rings"]
    for symbol, (x, y, z) in zip(symbols, coords, strict=True):
        rows.append(f"{symbol} {x:.8f} {y:.8f} {z:.8f}")
    path.write_text("\n".join(rows) + "\n")


def _one_indexed(groups: Sequence[Sequence[int]]) -> list[list[int]]:
    return [[i + 1 for i in group] for group in groups]


def render_primitive_rings(
    symbols: Sequence[str],
    coords: np.ndarray,
    output: str | PathLike[str],
    *,
    bonds: Sequence[tuple[int, int]] | None = None,
    cutoff: float | None = None,
    max_depth: int = 6,
    queries: np.ndarray | None = None,
    config: str = "paton",
    canvas_size: int = 900,
) -> tuple[RingReport, tuple[Assignment, ...]]:
    """Draw one hull per primitive ring and a dashed stroke on each junction.

    Query points, when given, are a second structure drawn in place. They
    are not aligned onto the molecule. Each is an He marker at the caller's
    coordinate, which is the stand-in for a Wannier centre.
    """
    xyz = np.asarray(coords, dtype=float)
    report = ring_report(
        symbols,
        xyz,
        bonds=bonds,
        cutoff=cutoff,
        max_depth=max_depth,
    )
    assignments: tuple[Assignment, ...] = ()
    if queries is not None:
        assignments = assign_points(xyz, report, np.asarray(queries, dtype=float))

    xr = _import_xyzrender()
    from xyzrender.types import OverlayConfig

    colours = [RING_COLOURS[i % len(RING_COLOURS)] for i in range(len(report.rings))]
    hull = _one_indexed(report.rings) or None
    ts_bonds = _one_indexed(report.junctions) or None
    overlay = None
    overlay_config = None
    if queries is not None and len(np.asarray(queries)) > 0:
        overlay_symbols = ["He"] * len(np.asarray(queries).reshape(-1, 3))
        overlay_config = OverlayConfig(
            color=JUNCTION_COLOUR,
            atom_scale=1.6,
            bond_width=0.0,
        )

    out = Path(output)
    with tempfile.TemporaryDirectory() as tmp:
        xyz_path = Path(tmp) / "molecule.xyz"
        _write_xyz(xyz_path, symbols, xyz)
        kwargs = {
            "config": config,
            "canvas_size": canvas_size,
            "hy": True,
            "transparent": True,
            "orient": True,
            "hull": hull,
            "hull_color": colours or None,
            "hull_opacity": 0.28,
            "hull_edge": True,
            "ts_bonds": ts_bonds,
            "ts_color": JUNCTION_COLOUR,
            "output": out,
        }
        if queries is not None and len(np.asarray(queries)) > 0:
            overlay = Path(tmp) / "queries.xyz"
            query_xyz = np.asarray(queries, dtype=float).reshape(-1, 3)
            _write_xyz(overlay, overlay_symbols, query_xyz)
            kwargs["overlay"] = overlay
            kwargs["overlay_config"] = overlay_config
            kwargs["auto_align"] = False
        xr.render(xyz_path, **kwargs)
    return report, assignments


def format_report(report: RingReport, assignments: Sequence[Assignment] = ()) -> str:
    """One plain-text table of the census, the junctions, and the labels."""
    census = (
        ", ".join(f"{size}:{count}" for size, count in report.census.items()) or "none"
    )
    lines = [
        f"atoms {report.n_atoms}",
        f"bonds {len(report.bonds)}",
        f"rings {len(report.rings)}",
        f"census {census}",
        f"junctions {len(report.junctions)}",
        f"pendants {len(report.pendants)}",
        f"depth_gap {len(report.depth_gap)}",
    ]
    for a, b in report.junctions:
        lines.append(f"junction {a} {b}")
    for a, b, girth in report.depth_gap:
        lines.append(f"depth_gap {a} {b} {girth}")
    for item in assignments:
        lines.append(
            f"query {item.query} {item.kind} {item.owner} "
            f"d {item.distance:.4f} margin {item.margin:.4f}"
        )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Sample:
    """One query on one frame, after rings have been matched by atom set."""

    frame: int
    query: int
    kind: str
    owner: int
    atoms: tuple[int, ...]
    chain: float
    terminal_atoms: tuple[int, ...] | None
    terminal_chain: float | None
    distance: float
    margin: float
    label_hop: bool
    block_hop: bool
    hop_length: int | None


@dataclass(frozen=True)
class Trajectory:
    """Labels, junction visits, and terminal-ring hops for every query."""

    samples: tuple[Sample, ...]
    n_frames: int
    n_queries: int

    def label_hops(self, query: int | None = None) -> int:
        """Owner changes, including a visit to a junction."""
        return sum(
            1
            for sample in self.samples
            if sample.label_hop and (query is None or sample.query == query)
        )

    def block_hops(self, query: int | None = None) -> int:
        """Changes of the terminal ring. A junction frame keeps that ring."""
        return sum(
            1
            for sample in self.samples
            if sample.block_hop and (query is None or sample.query == query)
        )

    def neighbor_hops(self, query: int | None = None) -> int:
        """Block hops whose rings share an atom or a junction in that frame."""
        return sum(
            1
            for sample in self.samples
            if sample.block_hop
            and sample.hop_length == 1
            and (query is None or sample.query == query)
        )

    def junction_frames(self, query: int | None = None) -> int:
        """Frames whose nearest owner is a bridge between rings."""
        return sum(
            1
            for sample in self.samples
            if sample.kind == "junction" and (query is None or sample.query == query)
        )


def _owner_atoms(report: RingReport, kind: str, owner: int) -> tuple[int, ...]:
    if kind == "ring":
        return tuple(sorted(report.rings[owner]))
    if kind == "junction":
        a, b = report.junctions[owner]
        return tuple(sorted((a, b)))
    return ()


def _backbone_ranks(
    coords: np.ndarray,
    report: RingReport,
    previous_axis: np.ndarray | None = None,
) -> tuple[dict[int, int], np.ndarray | None]:
    """Order ring centroids along their leading axis.

    The stored axis keeps its sign from the previous frame, so a rigid
    molecule does not reverse the chain coordinate. The rank is still not
    the identity of a ring. The atom set is.
    """
    n_rings = len(report.rings)
    if n_rings == 0:
        return {}, previous_axis
    if n_rings == 1:
        return {0: 0}, previous_axis
    centroids = np.stack([coords[list(ring)].mean(axis=0) for ring in report.rings])
    centered = centroids - centroids.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = np.asarray(vt[0], dtype=float)
    if previous_axis is not None and float(np.dot(axis, previous_axis)) < 0.0:
        axis = -axis
    projection = centroids @ axis
    order = np.argsort(projection, kind="mergesort")
    ranks = {int(ring_index): rank for rank, ring_index in enumerate(order)}
    return ranks, axis


def _ring_adjacency(report: RingReport) -> list[set[int]]:
    """Rings that share an atom, or that a junction joins."""
    sets = [set(ring) for ring in report.rings]
    adj: list[set[int]] = [set() for _ in sets]
    for i, left in enumerate(sets):
        for j in range(i + 1, len(sets)):
            if left & sets[j]:
                adj[i].add(j)
                adj[j].add(i)
    members: dict[int, list[int]] = {}
    for index, ring in enumerate(report.rings):
        for atom in ring:
            members.setdefault(atom, []).append(index)
    for a, b in report.junctions:
        for i in members.get(a, []):
            for j in members.get(b, []):
                if i != j:
                    adj[i].add(j)
                    adj[j].add(i)
    return adj


def _ring_distance(adj: Sequence[set[int]], src: int, dst: int) -> int | None:
    if src == dst:
        return 0
    seen = {src}
    queue: deque[tuple[int, int]] = deque([(src, 0)])
    while queue:
        node, dist = queue.popleft()
        for nbr in adj[node]:
            if nbr in seen:
                continue
            if nbr == dst:
                return dist + 1
            seen.add(nbr)
            queue.append((nbr, dist + 1))
    return None


def _chain_value(
    report: RingReport,
    ranks: dict[int, int],
    kind: str,
    owner: int,
) -> float:
    if kind == "ring":
        return float(ranks[owner])
    a, b = report.junctions[owner]
    ends = []
    for atom in (a, b):
        on_atom = [ranks[i] for i, ring in enumerate(report.rings) if atom in ring]
        if not on_atom:
            msg = f"junction atom {atom} lies on no returned ring"
            raise ValueError(msg)
        ends.append(sum(on_atom) / len(on_atom))
    return float(sum(ends) / len(ends))


def align_trajectory(
    frames: Sequence[tuple[RingReport, tuple[Assignment, ...], np.ndarray]],
) -> Trajectory:
    """Match each frame's labels by atom set and count terminal-ring hops.

    ``owner`` on an :class:`Assignment` is the position of that ring in one
    call. The same atoms can come back at another position. The canonical
    owner is the atom set, first seen in frame order.
    """
    registry: dict[tuple[str, tuple[int, ...]], int] = {}
    previous: dict[int, tuple[str, tuple[int, ...]]] = {}
    terminal: dict[int, tuple[int, ...] | None] = {}
    samples: list[Sample] = []
    n_frames = len(frames)
    n_queries = 0
    axis: np.ndarray | None = None
    for frame_index, (report, assigned, coords) in enumerate(frames):
        xyz = np.asarray(coords, dtype=float)
        ranks, axis = _backbone_ranks(xyz, report, axis)
        by_atoms = {tuple(sorted(ring)): index for index, ring in enumerate(report.rings)}
        adjacency = _ring_adjacency(report)
        n_queries = max(n_queries, len(assigned))
        for item in assigned:
            atoms = _owner_atoms(report, item.kind, item.owner)
            key = (item.kind, atoms)
            if key not in registry:
                registry[key] = len(registry)
            label_hop = item.query in previous and previous[item.query] != key
            current = terminal.get(item.query)
            block_hop = False
            hop_length: int | None = None
            if item.kind == "ring":
                if current is not None and current != atoms:
                    block_hop = True
                    src = by_atoms.get(current)
                    dst = by_atoms.get(atoms)
                    if src is not None and dst is not None:
                        hop_length = _ring_distance(adjacency, src, dst)
                current = atoms
            chain = _chain_value(report, ranks, item.kind, item.owner)
            terminal_chain = None
            if current is not None:
                src_ring = by_atoms.get(current)
                if src_ring is not None:
                    terminal_chain = float(ranks[src_ring])
            samples.append(
                Sample(
                    frame=frame_index,
                    query=item.query,
                    kind=item.kind,
                    owner=registry[key],
                    atoms=atoms,
                    chain=chain,
                    terminal_atoms=current,
                    terminal_chain=terminal_chain,
                    distance=item.distance,
                    margin=item.margin,
                    label_hop=label_hop,
                    block_hop=block_hop,
                    hop_length=hop_length,
                )
            )
            previous[item.query] = key
            terminal[item.query] = current
    return Trajectory(tuple(samples), n_frames, n_queries)


def track_queries(
    frames: Sequence[
        tuple[Sequence[str], np.ndarray, Sequence[tuple[int, int]] | None]
    ],
    queries: np.ndarray,
    *,
    cutoff: float | None = None,
    max_depth: int = 6,
) -> Trajectory:
    """Label query points on each frame and count terminal-ring hops.

    ``frames`` is ``(symbols, coordinates, bonds)``. Bonds and ``cutoff``
    follow the same rule as :func:`ring_report`: one of them, not both.
    ``queries`` has shape ``(frame, point, 3)``. A single point per frame
    may be ``(frame, 3)``.
    """
    points = np.asarray(queries, dtype=float)
    if points.ndim == 2:
        points = points[:, None, :]
    if points.ndim != 3 or points.shape[-1] != 3:
        msg = "queries must have shape (frames, points, 3)"
        raise ValueError(msg)
    if len(frames) != points.shape[0]:
        msg = f"{len(frames)} molecule frames and {points.shape[0]} query frames"
        raise ValueError(msg)
    packed: list[tuple[RingReport, tuple[Assignment, ...], np.ndarray]] = []
    for frame_index, ((symbols, coords, bonds), pts) in enumerate(
        zip(frames, points, strict=True)
    ):
        report = ring_report(
            symbols,
            coords,
            bonds=bonds,
            cutoff=cutoff,
            max_depth=max_depth,
        )
        n_owners = len(report.rings) + len(report.junctions)
        if n_owners < 2:
            msg = (
                f"frame {frame_index} has {n_owners} ring or junction owners; "
                "ringNetwork returned nothing to track"
            )
            raise ValueError(msg)
        packed.append(
            (report, assign_points(coords, report, pts), np.asarray(coords, dtype=float))
        )
    return align_trajectory(packed)


def read_xyz_frames(text: str) -> list[tuple[list[str], np.ndarray]]:
    """Every frame of a multi-structure XYZ. Columns past x, y, z are ignored."""
    lines = text.splitlines()
    frames: list[tuple[list[str], np.ndarray]] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        n_atoms = int(lines[index].split()[0])
        body = lines[index + 2 : index + 2 + n_atoms]
        if len(body) != n_atoms:
            msg = f"XYZ count {n_atoms} but read {len(body)} atoms"
            raise ValueError(msg)
        symbols: list[str] = []
        coords: list[list[float]] = []
        for line in body:
            parts = line.split()
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        frames.append((symbols, np.asarray(coords, dtype=float)))
        index += 2 + n_atoms
    if not frames:
        msg = "XYZ file has no frames"
        raise ValueError(msg)
    return frames


def read_frames(
    path: str | PathLike[str],
) -> list[tuple[list[str], np.ndarray, tuple[tuple[int, int], ...] | None]]:
    """One SDF frame with its bonds, or every frame of an XYZ without bonds."""
    text = Path(path).read_text()
    suffix = Path(path).suffix.lower()
    if suffix == ".sdf" or "V2000" in text.splitlines()[3:4]:
        symbols, coords, bonds = _read_sdf(text)
        return [(symbols, coords, bonds)]
    return [(symbols, coords, None) for symbols, coords in read_xyz_frames(text)]


def read_query_frames(path: str | PathLike[str]) -> np.ndarray:
    """Query coordinates with shape ``(frame, point, 3)``."""
    frames = read_xyz_frames(Path(path).read_text())
    width = len(frames[0][0])
    points: list[np.ndarray] = []
    for symbols, coords in frames:
        if len(symbols) != width:
            msg = f"query frames mix {width} and {len(symbols)} points"
            raise ValueError(msg)
        points.append(coords)
    return np.stack(points)


def load_trajectory(
    molecule: str | PathLike[str],
    queries: str | PathLike[str],
    *,
    cutoff: float | None = None,
    max_depth: int = 6,
) -> Trajectory:
    """Track query frames on a molecule file.

    One molecule frame is reused for every query frame. That is a fixed
    bond graph with a moving centre. Several molecule frames pair with
    the query frames one to one, and an XYZ molecule then needs ``cutoff``.
    """
    frames = read_frames(molecule)
    points = read_query_frames(queries)
    if len(frames) == 1 and points.shape[0] != 1:
        frames = frames * points.shape[0]
    return track_queries(frames, points, cutoff=cutoff, max_depth=max_depth)


def format_trajectory(track: Trajectory) -> str:
    """Hop counts and the terminal-ring coordinate of each query."""
    lines = [
        f"frames {track.n_frames}",
        f"queries {track.n_queries}",
        f"label_hops {track.label_hops()}",
        f"block_hops {track.block_hops()}",
        f"neighbor_hops {track.neighbor_hops()}",
        f"junction_frames {track.junction_frames()}",
    ]
    for query in range(track.n_queries):
        rows = [sample for sample in track.samples if sample.query == query]
        chain = " ".join(
            "none" if sample.terminal_chain is None else f"{sample.terminal_chain:.0f}"
            for sample in rows
        )
        lines.append(
            f"query {query} label_hops {track.label_hops(query)} "
            f"block_hops {track.block_hops(query)} "
            f"neighbor_hops {track.neighbor_hops(query)} "
            f"junction_frames {track.junction_frames(query)}"
        )
        lines.append(f"query {query} terminal {chain}")
    return "\n".join(lines) + "\n"


def write_trajectory_csv(path: str | PathLike[str], track: Trajectory) -> None:
    """One row per query per frame."""
    fieldnames = [
        "frame",
        "query",
        "kind",
        "owner",
        "atoms",
        "chain",
        "terminal_atoms",
        "terminal_chain",
        "distance",
        "margin",
        "label_hop",
        "block_hop",
        "hop_length",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in track.samples:
            writer.writerow(
                {
                    "frame": sample.frame,
                    "query": sample.query,
                    "kind": sample.kind,
                    "owner": sample.owner,
                    "atoms": " ".join(str(atom) for atom in sample.atoms),
                    "chain": f"{sample.chain:.4f}",
                    "terminal_atoms": " ".join(
                        str(atom) for atom in (sample.terminal_atoms or ())
                    ),
                    "terminal_chain": (
                        ""
                        if sample.terminal_chain is None
                        else f"{sample.terminal_chain:.4f}"
                    ),
                    "distance": f"{sample.distance:.6f}",
                    "margin": f"{sample.margin:.6f}",
                    "label_hop": int(sample.label_hop),
                    "block_hop": int(sample.block_hop),
                    "hop_length": "" if sample.hop_length is None else sample.hop_length,
                }
            )


def chain_of_pentagons(n_rings: int, side: float = 1.40, gap: float = 1.45):
    """Flat pentagons joined by one bridge each. Coordinates are angstrom."""
    radius = side / (2.0 * math.sin(math.pi / 5.0))
    symbols: list[str] = []
    coords: list[list[float]] = []
    bonds: list[tuple[int, int]] = []
    span = 2.0 * radius + gap
    for ring in range(n_rings):
        base = ring * 5
        origin = ring * span
        for k in range(5):
            ang = -math.pi / 2.0 + k * 2.0 * math.pi / 5.0
            symbols.append("C")
            coords.append([origin + radius * math.cos(ang), radius * math.sin(ang), 0.0])
        for k in range(5):
            a, b = base + k, base + (k + 1) % 5
            bonds.append((a, b) if a < b else (b, a))
        if ring:
            bonds.append(((ring - 1) * 5, base + 2))
    return symbols, np.asarray(coords, dtype=float), tuple(bonds)
