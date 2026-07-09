# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Drive shipped chemparseplot.api entry points."""

from __future__ import annotations

import numpy as np
import pytest

from chemparseplot.api import (
    convert_energy_magnitude,
    normalize_energy_unit,
    suite_pins,
)


def test_normalize_and_convert_energy_library_path():
    assert normalize_energy_unit("ev") == "eV"
    kcal = convert_energy_magnitude(1.0, "kcal/mol")
    assert isinstance(kcal, (float, np.floating)) or hasattr(kcal, "shape")
    # 1 eV -> ~23.06 kcal/mol chemical convention
    val = float(np.asarray(kcal).reshape(-1)[0])
    assert 20.0 < val < 25.0
    back = float(np.asarray(convert_energy_magnitude(val, "eV", source_unit="kcal/mol")))
    assert abs(back - 1.0) < 1e-6


def test_suite_pins_returns_dict():
    pins = suite_pins()
    assert isinstance(pins, dict)
