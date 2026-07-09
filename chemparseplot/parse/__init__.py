# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

from chemparseplot.parse import eon, orca, patterns

__all__ = ["eon", "grammar", "orca", "patterns"]

# Lazy imports for modules with optional heavy deps (h5py, pandas, parsimonious)
# Import directly: from chemparseplot.parse.chemgp_hdf5 import read_h5_table
# Or: from chemparseplot.parse import plumed, projection
# Grammar track: from chemparseplot.parse.grammar import parse_xyz_text


def __getattr__(name: str):
    if name == "grammar":
        from chemparseplot.parse import grammar as _grammar

        return _grammar
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
