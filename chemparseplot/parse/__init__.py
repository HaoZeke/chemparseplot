# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

from chemparseplot.parse import eon, orca, patterns

__all__ = ["eon", "orca", "patterns"]

# Lazy imports for modules with optional heavy deps (h5py, pandas)
# Import directly: from chemparseplot.parse.chemgp_hdf5 import read_h5_table
# Or: from chemparseplot.parse import plumed, projection
