# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Focused tests for shared NEB plotting helpers."""

import numpy as np
import pytest
from ase import Atoms

from chemparseplot.plot.structs import StructurePlacement


class TestNebPlotHelpers:
    def test_profile_strip_payload_returns_typed_entries(self):
        from chemparseplot.plot.neb import profile_strip_payload

        atoms_list = [Atoms("H"), Atoms("H"), Atoms("H")]
        payload = profile_strip_payload(
            atoms_list,
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 2.0, 0.0]),
            "crit_points",
            "energy",
        )

        assert all(isinstance(entry, StructurePlacement) for entry in payload)
        assert [entry.label for entry in payload] == ["R", "SP", "P"]

    def test_landscape_half_span_prefers_global_basis(self, monkeypatch):
        from chemparseplot.parse.eon.neb import NebOverlayStructure
        from chemparseplot.plot import neb as neb_plot

        recompute_calls = []

        def _fake_compute_projection_basis(*_args):
            recompute_calls.append(True)
            return "final-basis"

        def _fake_project_to_sd(_r, _p, basis):
            return np.zeros(1), np.array([2.0 if basis == "global-basis" else 20.0])

        monkeypatch.setattr(
            neb_plot, "compute_projection_basis", _fake_compute_projection_basis
        )
        monkeypatch.setattr(neb_plot, "project_to_sd", _fake_project_to_sd)

        half_span = neb_plot.landscape_half_span(
            (0.0, 4.0),
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
            [NebOverlayStructure(atoms=None, r=1.5, p=0.5, label="extra")],
            "global-basis",
        )

        assert half_span == pytest.approx(2.3)
        assert recompute_calls == []

    def test_save_plot_skips_tight_bbox_for_strip(self, monkeypatch, tmp_path):
        from chemparseplot.plot import neb as neb_plot

        saved = {}

        def _fake_savefig(path, **kwargs):
            saved[str(path)] = kwargs

        monkeypatch.setattr(neb_plot.plt, "savefig", _fake_savefig)

        strip_out = tmp_path / "strip.pdf"
        plain_out = tmp_path / "plain.pdf"

        neb_plot.save_plot(strip_out, 150, has_strip=True)
        neb_plot.save_plot(plain_out, 150, has_strip=False)

        assert "bbox_inches" not in saved[str(strip_out)]
        assert saved[str(plain_out)]["bbox_inches"] == "tight"
