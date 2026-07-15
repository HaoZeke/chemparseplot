# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
from chemparseplot.basetypes import (
    DimerOpt,
    MolGeom,
    SaddleMeasure,
    SpinID,
    nebiter,
    nebpath,
)
import numpy as np


def test_nebpath_nebiter():
    p = nebpath(0.5, 1.2, -0.74)
    it = nebiter(3, p)
    assert it.nebpath is p
    assert it.iteration == 3


def test_dimer_spin_mol_saddle():
    assert DimerOpt().saddle == "dimer"
    assert SpinID(1, "singlet").spin == "singlet"
    g = MolGeom(np.zeros((2, 3)), 0.0, np.zeros((2, 3)))
    assert g.pos.shape == (2, 3)
    assert SaddleMeasure().method == "not run"
