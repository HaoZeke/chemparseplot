# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
import numpy as np

from chemparseplot.parse.orca.geomscan import extract_energy_data
from chemparseplot.units import Q_

# Sample data snippet
sample_data = """
The Calculated Surface using the 'Actual Energy'
   7.55890395  -0.74239862
   7.32930292  -0.74349939
   7.09970189  -0.74467446

The Calculated Surface using the SCF energy
   7.55890395  -0.74239862
   7.32930292  -0.74349939
   7.09970189  -0.74467446
"""


def test_extract_actual_energy():
    x_expected = Q_([7.55890395, 7.32930292, 7.09970189], "bohr")
    y_expected = Q_([-0.74239862, -0.74349939, -0.74467446], "hartree")

    x_actual, y_actual = extract_energy_data(sample_data, "Actual Energy")

    assert np.allclose(x_actual.magnitude, x_expected.magnitude)
    assert np.allclose(y_actual.magnitude, y_expected.magnitude)
    assert x_actual.units == x_expected.units
    assert y_actual.units == y_expected.units


def test_extract_scf_energy():
    x_expected = Q_([7.55890395, 7.32930292, 7.09970189], "bohr")
    y_expected = Q_([-0.74239862, -0.74349939, -0.74467446], "hartree")

    x_scf, y_scf = extract_energy_data(sample_data, "SCF energy")

    assert np.allclose(x_scf.magnitude, x_expected.magnitude)
    assert np.allclose(y_scf.magnitude, y_expected.magnitude)
    assert x_scf.units == x_expected.units
    assert y_scf.units == y_expected.units


def test_empty_data():
    x_empty, y_empty = extract_energy_data("", "Actual Energy")
    assert x_empty.size == 0 and y_empty.size == 0


def test_malformed_data():
    malformed_data = "Some random text"
    x_malformed, y_malformed = extract_energy_data(malformed_data, "Actual Energy")
    assert x_malformed.size == 0 and y_malformed.size == 0
