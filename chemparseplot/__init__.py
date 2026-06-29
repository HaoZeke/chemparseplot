# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""chemparseplot public surface with deferred heavy submodules.

``parse`` and ``units`` remain importable as attributes, but ``units`` (pint)
is only loaded on first access so a bare ``import chemparseplot`` stays light
when only parsers are needed via ``chemparseplot.parse``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["parse", "units"]

if TYPE_CHECKING:
    from chemparseplot import parse as parse
    from chemparseplot import units as units


def __getattr__(name: str) -> Any:
    if name == "parse":
        from chemparseplot import parse as _parse

        return _parse
    if name == "units":
        from chemparseplot import units as _units

        return _units
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
