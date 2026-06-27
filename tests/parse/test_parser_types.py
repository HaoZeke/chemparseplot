# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Focused tests for shared parser result types."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.pure


def test_parse_orca_neb_returns_typed_mapping_result():
    from chemparseplot.parse.orca.neb.opi_parser import parse_orca_neb
    from chemparseplot.parse.types import OrcaNebResult

    mock_output_cls = MagicMock()
    mock_output_inst = MagicMock()
    mock_output_cls.return_value = mock_output_inst
    mock_output_inst.parse.return_value = None
    mock_output_inst.terminated_normally.return_value = True
    mock_output_inst.num_results_gbw = 2
    mock_output_inst.get_final_energy.side_effect = [-100.0, -99.5]
    mock_output_inst.get_geometry.side_effect = AttributeError("no geom")
    mock_output_inst.get_gradient.side_effect = AttributeError("no grad")

    with patch(
        "chemparseplot.parse.orca.neb.opi_parser._get_opi_output",
        return_value=mock_output_cls,
    ):
        result = parse_orca_neb("test_job", working_dir=Path("/tmp"))

    assert isinstance(result, OrcaNebResult)
    assert result["n_images"] == 2
    assert result["converged"] is True
    assert np.asarray(result["energies"]).shape == (2,)


def test_orca_neb_result_defaults_optional_fields():
    from chemparseplot.parse.types import OrcaNebResult

    result = OrcaNebResult(energies=np.array([0.0, 0.5, 0.0]))

    assert result.n_images == 3
    assert result.rmsd_r is None
    assert result.barrier_forward is None
    assert result.source == "unknown"


def test_orca_neb_result_from_mapping_coerces_arrays():
    from chemparseplot.parse.types import OrcaNebResult

    result = OrcaNebResult.from_mapping(
        {
            "energies": [0.0, 0.5, 0.0],
            "rmsd_r": [0.0, 1.0, 2.0],
            "grad_r": [0.0, 0.1, 0.0],
            "converged": True,
            "barrier_forward": 0.5,
            "source": "opi",
        }
    )

    assert isinstance(result.energies, np.ndarray)
    assert isinstance(result.rmsd_r, np.ndarray)
    assert isinstance(result.grad_r, np.ndarray)
    assert result.converged is True
    assert result.barrier_forward == 0.5
    assert result.source == "opi"


def test_parser_attrs_behaves_like_mapping():
    from collections.abc import Mapping

    from chemparseplot.parse.types import ParserAttrs

    attrs = ParserAttrs(data={"nx": 2, "ny": 3})

    assert isinstance(attrs, Mapping)
    assert attrs["nx"] == 2
    assert list(attrs.keys()) == ["nx", "ny"]
