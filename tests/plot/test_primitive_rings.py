# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Primitive-ring census and the xyzrender picture of it."""

from pathlib import Path

import numpy as np
import pytest

from chemparseplot.plot.rings import (
    assign_points,
    chain_of_pentagons,
    read_structure,
    render_primitive_rings,
    report_from_bonds,
)

SDF = Path(__file__).parent / "data" / "sexithiophene.sdf"
pydseams = pytest.importorskip("pydseams")


def test_chain_has_one_junction_per_link():
    _symbols, _coords, bonds = chain_of_pentagons(6)
    report = report_from_bonds(30, bonds, max_depth=6)
    assert report.census == {5: 6}
    assert report.junctions == ((0, 7), (5, 12), (10, 17), (15, 22), (20, 27))
    assert report.depth_gap == ()
    assert report.pendants == ()


def test_depth_four_drops_the_pentagons():
    _symbols, _coords, bonds = chain_of_pentagons(6)
    report = report_from_bonds(30, bonds, max_depth=4)
    assert report.rings == ()
    assert report.junctions == ()
    assert len(report.depth_gap) == 30


def test_shared_fused_edge_is_not_a_junction():
    # A 5-cycle and a 6-cycle that share the edge 0-1. The shared edge lies
    # on both faces. The outer walk is not a primitive ring.
    bonds = []
    for k in range(5):
        a, b = k, (k + 1) % 5
        bonds.append((a, b) if a < b else (b, a))
    outer = [0, 1, 5, 6, 7, 8]
    for k in range(len(outer)):
        a, b = outer[k], outer[(k + 1) % len(outer)]
        bonds.append((a, b) if a < b else (b, a))
    report = report_from_bonds(9, bonds, max_depth=9)
    assert report.census == {5: 1, 6: 1}
    assert (0, 1) not in report.junctions
    assert report.junctions == ()


def test_sexithiophene_bond_graph():
    symbols, coords, bonds = read_structure(SDF)
    assert bonds is not None
    report = report_from_bonds(len(symbols), bonds, max_depth=6)
    assert report.census == {5: 6}
    assert len(report.junctions) == 5
    assert len(report.pendants) == 14
    assert report.depth_gap == ()
    doubles = []
    for a, b in bonds:
        if {symbols[a], symbols[b]} != {"C"}:
            continue
        dist = float(((coords[a] - coords[b]) ** 2).sum() ** 0.5)
        if dist < 1.39:
            doubles.append(0.5 * (coords[a] + coords[b]))
    assigned = assign_points(coords, report, np.asarray(doubles))
    assert len(assigned) == 12
    assert all(item.kind == "ring" for item in assigned)


def test_cutoff_keeps_or_destroys_the_bridges():
    symbols, coords, bonds = read_structure(SDF)
    from chemparseplot.plot.rings import ring_report

    covalent = ring_report(symbols, coords, bonds=bonds, max_depth=6)
    kept = ring_report(symbols, coords, cutoff=1.95, max_depth=6)
    crowded = ring_report(symbols, coords, cutoff=3.50, max_depth=6)
    assert kept.census == covalent.census == {5: 6}
    assert len(kept.junctions) == 5
    assert crowded.census == {3: 110, 4: 5}
    assert crowded.junctions == ()


def test_render_writes_svg(tmp_path: Path):
    pytest.importorskip("xyzrender")
    symbols, coords, bonds = chain_of_pentagons(2)
    out = tmp_path / "rings.svg"
    query = coords[:5].mean(axis=0)
    report, assigned = render_primitive_rings(
        symbols,
        coords,
        out,
        bonds=bonds,
        queries=query.reshape(1, 3),
    )
    text = out.read_text()
    assert report.census == {5: 2}
    assert len(report.junctions) == 1
    assert assigned[0].kind == "ring"
    assert text.lstrip().startswith("<svg") or "<svg" in text[:200]
    assert len(text) > 500
