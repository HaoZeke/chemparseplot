# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Terminal-ring hops along a query trajectory."""

from pathlib import Path

import numpy as np
import pytest

from chemparseplot.plot.rings import (
    Assignment,
    RingReport,
    align_trajectory,
    chain_of_pentagons,
    load_trajectory,
    read_xyz_frames,
    report_from_bonds,
    track_queries,
)

SDF = Path(__file__).parent / "data" / "sexithiophene.sdf"


def _report(rings, junctions=()):
    return RingReport(
        n_atoms=10,
        bonds=(),
        rings=rings,
        junctions=junctions,
        pendants=(),
        other=(),
        depth_gap=(),
    )


def _coords():
    coords = np.zeros((10, 3))
    coords[:, 0] = np.arange(10, dtype=float)
    return coords


def test_enumeration_index_is_not_the_ring():
    left = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))
    right = ((5, 6, 7, 8, 9), (0, 1, 2, 3, 4))
    junction = ((4, 5),)
    coords = _coords()
    same_atoms = align_trajectory(
        [
            (_report(left, junction), (Assignment(0, "ring", 0, 0.0, 1.0),), coords),
            (_report(right, junction), (Assignment(0, "ring", 1, 0.0, 1.0),), coords),
        ]
    )
    assert same_atoms.label_hops() == 0
    assert same_atoms.block_hops() == 0
    assert same_atoms.samples[0].atoms == same_atoms.samples[1].atoms

    swapped = align_trajectory(
        [
            (_report(left, junction), (Assignment(0, "ring", 0, 0.0, 1.0),), coords),
            (_report(right, junction), (Assignment(0, "ring", 0, 0.0, 1.0),), coords),
        ]
    )
    assert swapped.block_hops() == 1
    assert swapped.neighbor_hops() == 1
    assert swapped.samples[1].hop_length == 1


def test_xyz_frames_keep_every_structure():
    text = "1\nframe\nC 0 0 0\n1\nframe\nC 1 0 0\n"
    frames = read_xyz_frames(text)
    assert len(frames) == 2
    assert frames[1][1][0, 0] == 1.0


def test_query_frame_count_must_match(tmp_path):
    molecule = tmp_path / "mol.xyz"
    queries = tmp_path / "queries.xyz"
    molecule.write_text("1\n\nC 0 0 0\n1\n\nC 0 0 0\n")
    queries.write_text("1\n\nHe 0 0 0\n")
    with pytest.raises(ValueError, match="query frames"):
        load_trajectory(molecule, queries, cutoff=1.95)


def _centroid(coords, report, atoms):
    for ring in report.rings:
        if set(ring) == set(atoms):
            return coords[list(ring)].mean(axis=0)
    msg = f"ring {atoms} missing"
    raise AssertionError(msg)


@pytest.fixture
def pentagons():
    pytest.importorskip("pydseams")
    symbols, coords, bonds = chain_of_pentagons(3)
    report = report_from_bonds(len(symbols), bonds, max_depth=6)
    return symbols, coords, bonds, report


def test_junction_visit_is_one_neighbor_hop(pentagons):
    symbols, coords, bonds, report = pentagons
    start = _centroid(coords, report, (0, 1, 2, 3, 4))
    bridge = 0.5 * (coords[0] + coords[7])
    nxt = _centroid(coords, report, (5, 6, 7, 8, 9))
    frames = [(symbols, coords, bonds)] * 4
    queries = np.stack([start, start, bridge, nxt])
    track = track_queries(frames, queries, max_depth=6)
    assert track.label_hops() == 2
    assert track.block_hops() == 1
    assert track.neighbor_hops() == 1
    assert track.junction_frames() == 1
    hopped = [sample for sample in track.samples if sample.block_hop]
    assert hopped[0].hop_length == 1
    assert track.samples[0].distance < 1e-8
    assert track.samples[2].kind == "junction"
    assert track.samples[2].terminal_atoms == track.samples[0].terminal_atoms
    assert track.samples[3].terminal_atoms != track.samples[0].terminal_atoms


def test_skipping_a_ring_is_a_longer_hop(pentagons):
    symbols, coords, bonds, report = pentagons
    start = _centroid(coords, report, (0, 1, 2, 3, 4))
    far = _centroid(coords, report, (10, 11, 12, 13, 14))
    frames = [(symbols, coords, bonds)] * 2
    track = track_queries(frames, np.stack([start, far]), max_depth=6)
    assert track.block_hops() == 1
    assert track.neighbor_hops() == 0
    assert track.samples[1].hop_length == 2


def test_returning_from_a_junction_is_not_a_block_hop(pentagons):
    symbols, coords, bonds, report = pentagons
    start = _centroid(coords, report, (0, 1, 2, 3, 4))
    bridge = 0.5 * (coords[0] + coords[7])
    frames = [(symbols, coords, bonds)] * 3
    track = track_queries(frames, np.stack([start, bridge, start]), max_depth=6)
    assert track.label_hops() == 2
    assert track.block_hops() == 0
    assert track.samples[2].terminal_atoms == track.samples[0].terminal_atoms


def test_sexithiophene_midpoints_stay_on_their_rings():
    pytest.importorskip("pydseams")
    from chemparseplot.plot.rings import read_structure

    symbols, coords, bonds = read_structure(SDF)
    doubles = []
    for a, b in bonds:
        if {symbols[a], symbols[b]} != {"C"}:
            continue
        dist = float(np.linalg.norm(coords[a] - coords[b]))
        if dist < 1.39:
            doubles.append(0.5 * (coords[a] + coords[b]))
    queries = np.asarray(doubles)
    assert len(queries) == 12
    frames = [(symbols, coords, bonds)] * 3
    track = track_queries(frames, np.broadcast_to(queries, (3, 12, 3)), max_depth=6)
    assert track.label_hops() == 0
    assert track.block_hops() == 0
    assert track.n_queries == 12
    for query in range(12):
        rows = [
            sample.terminal_atoms
            for sample in track.samples
            if sample.query == query
        ]
        assert rows[0] == rows[1] == rows[2]


def test_sexithiophene_move_between_rings_is_one_block_hop():
    pytest.importorskip("pydseams")
    from chemparseplot.plot.rings import read_structure, report_from_bonds

    symbols, coords, bonds = read_structure(SDF)
    report = report_from_bonds(len(symbols), bonds, max_depth=6)
    doubles = []
    for a, b in bonds:
        if {symbols[a], symbols[b]} != {"C"}:
            continue
        dist = float(np.linalg.norm(coords[a] - coords[b]))
        if dist < 1.39:
            doubles.append(0.5 * (coords[a] + coords[b]))
    frames = [(symbols, coords, bonds)] * 2
    placed = track_queries(frames, np.broadcast_to(doubles, (2, 12, 3)), max_depth=6)
    groups: dict[tuple[int, ...], int] = {}
    for sample in placed.samples:
        if sample.frame == 0 and sample.terminal_atoms is not None:
            groups.setdefault(sample.terminal_atoms, sample.query)
    assert len(groups) == 6

    def ring_query(atom: int) -> tuple[tuple[int, ...], int] | None:
        for atoms, query in groups.items():
            if atom in atoms:
                return atoms, query
        return None

    pair = None
    for a, b in report.junctions:
        left = ring_query(a)
        right = ring_query(b)
        if left is not None and right is not None and left[0] != right[0]:
            pair = (left[1], right[1])
            break
    assert pair is not None
    queries = np.stack([doubles[pair[0]], doubles[pair[1]]])
    moved = track_queries(frames, queries, max_depth=6)
    assert moved.block_hops() == 1
    assert moved.samples[1].hop_length == 1
