"""Incremental ring analysis agrees with independent full enumeration."""

import pydseams

from chemparseplot.plot.rings import report_from_bonds


def test_incremental_ring_reports_match_full_enumeration():
    cycle = [(i, (i + 1) % 6) for i in range(6)]
    updater = pydseams.yoda.RingUpdater(6)
    for bonds in [cycle, cycle, cycle[:-1], cycle, cycle + [(0, 3)]]:
        full = report_from_bonds(6, bonds, max_depth=6)
        incremental = report_from_bonds(6, bonds, max_depth=6, updater=updater)
        assert {frozenset(r) for r in incremental.rings} == {
            frozenset(r) for r in full.rings
        }
        assert incremental.census == full.census
        assert incremental.bonds == full.bonds
        assert incremental.junctions == full.junctions
        assert incremental.pendants == full.pendants
        assert incremental.other == full.other
        assert incremental.depth_gap == full.depth_gap
