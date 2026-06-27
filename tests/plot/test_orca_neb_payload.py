# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Focused tests for normalized ORCA NEB plotting payloads."""

import numpy as np
import pytest

from chemparseplot.plot.neb import (
    _normalize_orca_neb_plot_payload,
    _orca_saddle_index,
)


def test_orca_neb_payload_normalizes_units():
    payload = _normalize_orca_neb_plot_payload(
        {
            "energies": np.array([0.0, 1.0, 0.0]),
            "grad_r": np.array([0.0, 0.1, 0.0]),
            "grad_p": np.array([0.0, -0.1, 0.0]),
            "rmsd_r": np.array([0.0, 1.0, 2.0]),
            "rmsd_p": np.array([2.0, 1.0, 0.0]),
            "barrier_forward": 1.0,
        },
        "kcal/mol",
    )

    assert payload.n_images == 3
    assert payload.barrier_forward == 1.0
    assert payload.rmsd_r is not None
    assert payload.grad_r is not None
    assert payload.energies[1] > 20.0
    assert payload.grad_r[1] > 2.0


def test_orca_neb_payload_requires_energy_data():
    with pytest.raises(ValueError, match="No energy data"):
        _normalize_orca_neb_plot_payload({"energies": []}, "eV")


def test_orca_saddle_index_ignores_endpoint_maxima():
    assert _orca_saddle_index(np.array([1.0, 0.5, 0.0])) is None
    assert _orca_saddle_index(np.array([0.0, 1.0, 0.0])) == 1
