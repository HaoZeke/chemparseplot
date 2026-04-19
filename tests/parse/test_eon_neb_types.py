# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

pytestmark = pytest.mark.pure


def test_load_structures_and_calculate_additional_rmsd_returns_overlay_records(
    monkeypatch,
):
    from chemparseplot.parse.eon import neb as neb_mod

    reactant = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    midpoint = Atoms("H", positions=[[0.1, 0.0, 0.0]])
    product = Atoms("H", positions=[[0.2, 0.0, 0.0]])
    add_atoms = Atoms("H", positions=[[0.3, 0.0, 0.0]])
    sp_atoms = Atoms("H", positions=[[0.4, 0.0, 0.0]])
    main_atoms = [reactant, midpoint, product]

    con_file = Path("neb.con")
    add_file = Path("extra.con")
    sp_file = Path("sp.con")

    monkeypatch.setattr(Path, "exists", lambda self: self == sp_file)

    def _fake_read(path, index=None):
        if path == con_file and index == ":":
            return main_atoms
        if path == add_file:
            return add_atoms
        if path == sp_file:
            return sp_atoms
        raise AssertionError((path, index))

    def _fake_rmsd(atoms_seq, _ira_instance, *, ref_atom, ira_kmax):
        assert ira_kmax == pytest.approx(14.0)
        atom = atoms_seq[0]
        if atom is sp_atoms and ref_atom is reactant:
            return np.array([1.25])
        if atom is sp_atoms and ref_atom is product:
            return np.array([2.5])
        if atom is add_atoms and ref_atom is reactant:
            return np.array([3.5])
        if atom is add_atoms and ref_atom is product:
            return np.array([4.5])
        raise AssertionError((atom, ref_atom))

    monkeypatch.setattr(neb_mod, "ase_read", _fake_read)
    monkeypatch.setattr(neb_mod, "calculate_rmsd_from_ref", _fake_rmsd)
    monkeypatch.setattr(neb_mod, "ira_mod", None)

    atoms_list, additional_data, sp_data = (
        neb_mod.load_structures_and_calculate_additional_rmsd(
            con_file,
            [(add_file, "")],
            14.0,
            sp_file=sp_file,
        )
    )

    assert atoms_list == main_atoms
    assert len(additional_data) == 1
    assert isinstance(additional_data[0], neb_mod.NebOverlayStructure)
    assert additional_data[0].atoms is add_atoms
    assert additional_data[0].r == pytest.approx(3.5)
    assert additional_data[0].p == pytest.approx(4.5)
    assert additional_data[0].label == "extra"

    assert isinstance(sp_data, neb_mod.NebOverlayStructure)
    assert sp_data.atoms is sp_atoms
    assert sp_data.r == pytest.approx(1.25)
    assert sp_data.p == pytest.approx(2.5)
    assert sp_data.label == "SP"
