# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
from rgpycrumbs.basetypes import nebiter

from chemparseplot.parse.orca.neb.interp import extract_interp_points
from chemparseplot.units import ureg


def test_extract_interp_points_valid_input():
    # Example valid text input
    text_input = """Iteration: 1
Images: Distance  (Bohr), Energy (Eh)
13.0 0.0 0.0
1.0 10.0 -0.5
Iteration: 2
Images: Distance  (Bohr), Energy (Eh)
0.0 0.2 0.3
1.0 20.0 -1.0
"""
    # Extract data
    result = extract_interp_points(text_input)

    # Check if result is a list of nebiter
    assert isinstance(result, list)
    assert all(isinstance(item, nebiter) for item in result)

    # Check if each nebiter contains a nebpath with correct values and units
    assert result[0].iteration == 1
    assert result[0].nebpath.norm_dist.magnitude[0] == 13.0
    assert result[0].nebpath.arc_dist.magnitude[0] == 0.0
    assert result[0].nebpath.energy.magnitude[0] == 0.0
    assert result[0].nebpath.norm_dist.units == ureg.Unit("dimensionless")
    assert result[0].nebpath.arc_dist.units == "bohr"
    assert result[0].nebpath.energy.units == "hartree"
    assert result[1].nebpath.norm_dist.magnitude[0] == 0.0
    assert result[1].nebpath.arc_dist.magnitude[0] == 0.2
    assert result[1].nebpath.energy.magnitude[0] == 0.3


def test_extract_interp_points_invalid_input():
    # Example invalid text input
    text_input = """This is not a valid input for the function."""
    result = extract_interp_points(text_input)
    # Expecting empty list for invalid input
    assert result == []
