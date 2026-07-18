# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""eOn trajectory parsers.

``con_io`` is always importable (readcon loads on first CON call).
``frame_series`` needs the ``neb`` extra (readcon + polars + ase) and is
lazy so bare ``import chemparseplot.parse`` stays light.
"""

from chemparseplot.parse.eon import con_io

__all__ = ["con_io", "frame_series"]


def __getattr__(name: str):
    if name == "frame_series":
        from chemparseplot.parse.eon import frame_series as _frame_series

        return _frame_series
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
