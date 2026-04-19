# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Focused tests for typed ChemGP JSONL parser records."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.pure


def test_parse_comparison_jsonl_returns_typed_summary(tmp_path):
    from collections.abc import Mapping

    from chemparseplot.parse.chemgp_jsonl import parse_comparison_jsonl
    from chemparseplot.parse.types import ParserAttrs

    lines = [
        json.dumps({"method": "neb", "oracle_calls": 5, "max_force": 0.3}),
        json.dumps({"summary": True, "total": 10}),
    ]
    path = tmp_path / "comparison.jsonl"
    path.write_text("\n".join(lines))

    data = parse_comparison_jsonl(path)

    assert data.summary is not None
    assert isinstance(data.summary, Mapping)
    assert isinstance(data.summary, ParserAttrs)
    assert data.summary["total"] == 10


def test_parse_gp_quality_jsonl_returns_typed_meta(tmp_path):
    from collections.abc import Mapping

    from chemparseplot.parse.chemgp_jsonl import parse_gp_quality_jsonl
    from chemparseplot.parse.types import ParserAttrs

    lines = [
        json.dumps(
            {
                "type": "grid_meta",
                "nx": 1,
                "ny": 1,
                "x_min": -2,
                "x_max": 2,
                "y_min": -2,
                "y_max": 2,
            }
        ),
        json.dumps(
            {
                "type": "grid",
                "n_train": 5,
                "ix": 0,
                "iy": 0,
                "x": 0.0,
                "y": 0.0,
                "true_e": -1.0,
                "gp_e": -0.9,
                "gp_var": 0.1,
            }
        ),
    ]
    path = tmp_path / "gp_quality.jsonl"
    path.write_text("\n".join(lines))

    data = parse_gp_quality_jsonl(path)

    assert isinstance(data.meta, Mapping)
    assert isinstance(data.meta, ParserAttrs)
    assert data.meta["nx"] == 1
    assert 5 in data.grids


def test_parse_gp_quality_jsonl_keeps_typed_training_points(tmp_path):
    from chemparseplot.parse.chemgp_jsonl import parse_gp_quality_jsonl

    lines = [
        json.dumps({"type": "grid_meta", "nx": 1, "ny": 1}),
        json.dumps(
            {"type": "train_point", "n_train": 5, "x": 1.0, "y": 2.0, "energy": -3.0}
        ),
        json.dumps(
            {
                "type": "grid",
                "n_train": 5,
                "ix": 0,
                "iy": 0,
                "x": 0.0,
                "y": 0.0,
                "true_e": -1.0,
                "gp_e": -0.9,
                "gp_var": 0.1,
            }
        ),
    ]
    path = tmp_path / "gp_quality_train.jsonl"
    path.write_text("\n".join(lines))

    data = parse_gp_quality_jsonl(path)

    grid = data.grids[5]
    assert grid.nx == 1
    assert grid.ny == 1
    assert grid.train_x == [1.0]
    assert grid.train_y == [2.0]
    assert grid.train_e == [-3.0]
