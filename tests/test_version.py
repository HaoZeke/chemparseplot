# SPDX-License-Identifier: MIT
"""The hatch-vcs version file stays inside the package."""

from __future__ import annotations

import importlib
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import chemparseplot

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_file_is_inside_the_package():
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    version_file = config["tool"]["hatch"]["build"]["hooks"]["vcs"]["version-file"]
    assert Path(version_file).parts[0] == "chemparseplot"


def test_dunder_version_matches_metadata():
    assert chemparseplot.__version__ == version("chemparseplot")
    assert "__version__" in chemparseplot.__all__


def test_dunder_version_falls_back_to_metadata(monkeypatch):
    monkeypatch.setitem(sys.modules, "chemparseplot._version", None)
    try:
        reloaded = importlib.reload(chemparseplot)
        assert reloaded.__version__ == version("chemparseplot")
    finally:
        monkeypatch.undo()
        importlib.reload(chemparseplot)
