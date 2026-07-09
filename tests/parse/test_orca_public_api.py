# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Public ORCA parse surface: OPI stays behind chemparseplot APIs."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chemparseplot.parse.types import OrcaNebResult


def test_public_exports_from_parse_orca():
    from chemparseplot.parse import orca
    from chemparseplot.parse.orca import parse_orca_neb, parse_orca_neb_fallback
    from chemparseplot.parse.orca.neb import opi_available, parse_orca_neb as neb_parse

    assert parse_orca_neb is neb_parse
    assert callable(parse_orca_neb_fallback)
    assert callable(opi_available)
    assert "parse_orca_neb" in orca.__all__


def test_opi_loader_has_no_rgpycrumbs_dependency():
    root = Path(__file__).resolve().parents[2] / "chemparseplot" / "parse" / "orca"
    for name in ("_opi.py", "neb/opi_parser.py"):
        text = (root / name).read_text()
        tree = ast.parse(text, filename=name)
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("rgpycrumbs")
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("rgpycrumbs")
        # no runtime ensure_import helper (docstring may mention the old path)
        assert "from rgpycrumbs" not in text
        assert "import rgpycrumbs" not in text
        assert "ensure_import(" not in text
    opi_mod = (root / "_opi.py").read_text()
    assert "opi.output.core" in opi_mod
    assert "chemparseplot[opi]" in opi_mod


def test_opi_available_false_without_package():
    from chemparseplot.parse.orca import _opi

    _opi.reset_opi_cache()
    with patch("importlib.util.find_spec", return_value=None):
        _opi.reset_opi_cache()
        assert _opi.opi_available() is False
    _opi.reset_opi_cache()


def test_get_opi_output_class_import_error_message():
    from chemparseplot.parse.orca import _opi

    _opi.reset_opi_cache()
    with patch.object(_opi, "_opi_output_cls", None):
        with patch("importlib.import_module", side_effect=ImportError("no opi")):
            _opi.reset_opi_cache()
            with pytest.raises(ImportError, match="chemparseplot\\[opi\\]|parse_orca_neb"):
                _opi.get_opi_output_class()
    _opi.reset_opi_cache()


def test_parse_orca_neb_legacy_backend_uses_interp(tmp_path: Path):
    """backend=legacy must not require OPI."""
    from chemparseplot.parse.orca.neb import parse_orca_neb

    # Minimal .interp content that extract_interp_points understands may fail;
    # use fallback mock.
    with patch(
        "chemparseplot.parse.orca.neb.opi_parser.parse_orca_neb_fallback",
        return_value=OrcaNebResult(
            energies=np.array([0.0, 1.0, 0.5]),
            source="legacy_interp",
            n_images=3,
            converged=True,
        ),
    ) as fb:
        result = parse_orca_neb("job", working_dir=tmp_path, backend="legacy")
    fb.assert_called_once()
    assert result.source == "legacy_interp"
    assert result.n_images == 3


def test_parse_orca_neb_auto_falls_back_when_opi_missing(tmp_path: Path):
    from chemparseplot.parse.orca.neb import parse_orca_neb

    expected = OrcaNebResult(
        energies=np.array([0.0, 0.2]),
        source="legacy_interp",
        n_images=2,
        converged=True,
    )
    with patch(
        "chemparseplot.parse.orca.neb.opi_parser._parse_orca_neb_opi",
        side_effect=ImportError("no opi"),
    ):
        with patch(
            "chemparseplot.parse.orca.neb.opi_parser.parse_orca_neb_fallback",
            return_value=expected,
        ):
            result = parse_orca_neb("job", working_dir=tmp_path, backend="auto")
    assert result is expected


def test_parse_orca_neb_opi_backend_does_not_fallback(tmp_path: Path):
    from chemparseplot.parse.orca.neb import parse_orca_neb

    with patch(
        "chemparseplot.parse.orca.neb.opi_parser._parse_orca_neb_opi",
        side_effect=ImportError("no opi"),
    ):
        with pytest.raises(ImportError):
            parse_orca_neb("job", working_dir=tmp_path, backend="opi")


def test_parse_orca_neb_with_mocked_opi_still_works():
    """Existing mock contract: patch _get_opi_output."""
    from chemparseplot.parse.orca.neb.opi_parser import parse_orca_neb

    mock_output = MagicMock()
    mock_output.terminated_normally.return_value = True
    mock_output.num_results_gbw = 2
    mock_output.get_final_energy.side_effect = [0.0, 0.1]
    mock_output.get_geometry.side_effect = AttributeError("no geom")
    mock_output.get_gradient.side_effect = AttributeError("no grad")
    mock_output.orca_version = "6.1"

    with patch(
        "chemparseplot.parse.orca.neb.opi_parser._get_opi_output",
        return_value=lambda *a, **k: mock_output,
    ):
        # Output is called as Output(basename, working_dir=...)
        with patch(
            "chemparseplot.parse.orca.neb.opi_parser._get_opi_output",
            return_value=MagicMock(return_value=mock_output),
        ):
            result = parse_orca_neb("test_job", working_dir=Path("/tmp"), backend="opi")
    assert isinstance(result, OrcaNebResult)
    assert result.source == "opi"
    assert result.n_images == 2


def test_no_top_level_opi_import_in_parse_orca_tree():
    root = Path(__file__).resolve().parents[2] / "chemparseplot" / "parse" / "orca"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name.startswith(".") or "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "opi" or alias.name.startswith("opi."):
                        offenders.append(f"{path.name}: import {alias.name}")
            if isinstance(node, ast.ImportFrom) and node.module and (
                node.module == "opi" or node.module.startswith("opi.")
            ):
                offenders.append(f"{path.name}: from {node.module}")
    # Only _opi.py may load opi via importlib (string), not top-level import
    assert offenders == []
