# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Structural guard: eOn CON writes go through readcon, not ase.io.write."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

EON_PARSE = Path(__file__).resolve().parents[2] / "chemparseplot" / "parse" / "eon"


@pytest.mark.pure
def test_eon_parse_has_con_io_module():
    assert (EON_PARSE / "con_io.py").is_file()


@pytest.mark.pure
def test_no_ase_io_write_in_eon_parse_sources():
    offenders: list[str] = []
    for path in sorted(EON_PARSE.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {"ase.io", "ase"}:
                names = {alias.name for alias in node.names}
                if "write" in names:
                    offenders.append(f"{path.name}: from {node.module} import write")
            if isinstance(node, ast.Attribute) and node.attr == "write":
                # ase.io.write(...)
                val = node.value
                if isinstance(val, ast.Attribute) and val.attr == "io":
                    offenders.append(f"{path.name}: ase.io.write")
                if isinstance(val, ast.Name) and val.id in {"asewrite", "ase_write"}:
                    offenders.append(f"{path.name}: {val.id}.write")
    assert offenders == []


@pytest.mark.pure
def test_stitch_and_trajectory_import_con_io():
    stitch = (EON_PARSE / "stitch.py").read_text()
    common = (EON_PARSE / "_trajectory_common.py").read_text()
    assert "chemparseplot.parse.eon.con_io" in stitch
    assert "chemparseplot.parse.eon import con_io" in common or "parse.eon.con_io" in common
    assert "ase.io" not in stitch
