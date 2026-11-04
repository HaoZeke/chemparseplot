# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""chemparseplot public surface with deferred heavy submodules.

``parse`` and ``units`` remain attributes; they load via importlib on first
access so a bare ``import chemparseplot`` does not construct the pint registry
or pull the full parse package graph until needed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

try:
    from chemparseplot._version import __version__
except ImportError:
    # A source tree that hatch-vcs has not built yet has no _version.py.
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("chemparseplot")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

__all__ = ["__version__", "parse", "units"]

if TYPE_CHECKING:
    from chemparseplot import parse as parse
    from chemparseplot import units as units


def __getattr__(name: str) -> Any:
    if name == "parse":
        return importlib.import_module("chemparseplot.parse")
    if name == "units":
        return importlib.import_module("chemparseplot.units")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
