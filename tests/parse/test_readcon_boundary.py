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


@pytest.mark.pure
def test_eon_modules_do_not_import_readcon_at_module_level_except_typed_files():
    """Only legacy type-hint modules may keep top-level readcon imports."""
    allowed = {"min_trajectory.py", "dimer_trajectory.py"}
    offenders = []
    for path in sorted(EON_PARSE.glob("*.py")):
        if path.name in allowed or path.name.startswith("_"):
            # _trajectory_common should not top-level import readcon anymore
            if path.name == "_trajectory_common.py":
                text = path.read_text()
                assert "import readcon" not in text.split("def ")[0]
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "readcon" or alias.name.startswith("readcon."):
                        offenders.append(path.name)
            if isinstance(node, ast.ImportFrom) and node.module == "readcon":
                offenders.append(path.name)
    assert offenders == []


@pytest.mark.pure
def test_con_io_is_single_readcon_entry_point():
    text = (EON_PARSE / "con_io.py").read_text()
    assert "readcon>=0.7" in text
    assert "def read_con_frames" in text
    assert "def write_con_frames" in text
