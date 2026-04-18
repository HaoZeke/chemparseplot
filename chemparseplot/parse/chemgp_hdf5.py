# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""HDF5 file I/O utilities for ChemGP data.

This module provides functions for reading structured data from ChemGP HDF5
output files. The HDF5 layout mirrors the Julia common_plot.jl helpers.

HDF5 Layout
-----------
- ``grids/<name>``: 2D arrays with attrs x_range, y_range, x_length, y_length
- ``table/<name>``: group of same-length 1D arrays
- ``paths/<name>``: point sequences (x, y or rAB, rBC)
- ``points/<name>``: point sets (x, y or pc1, pc2)
- Root attrs: metadata scalars

.. versionadded:: 1.7.0
    Extracted from chemgp.plt_gp to standalone module.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ArrayGroup(Mapping[str, np.ndarray]):
    """Named mapping of arrays loaded from an HDF5 group."""

    values: dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True, slots=True)
class MetadataAttrs(Mapping[str, Any]):
    """Named mapping of HDF5 root metadata attributes."""

    values: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


def read_h5_table(f: Any, name: str = "table") -> Any:
    """Read a group of same-length vectors as a DataFrame.

    Parameters
    ----------
    f
        Open HDF5 file object
    name
        Name of the table group (default: "table")

    Returns
    -------
    DataFrame
        DataFrame with columns from the HDF5 group
    """
    import pandas as pd

    g = f[name]
    cols = {}
    for k in g.keys():
        arr = g[k][()]
        if arr.dtype.kind in {"S", "O"}:
            cols[k] = arr.astype(str).tolist()
        else:
            cols[k] = arr.tolist()
    return pd.DataFrame(cols)


def read_h5_grid(
    f: Any, name: str
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Read a 2D grid with optional axis ranges.

    Parameters
    ----------
    f
        Open HDF5 file object
    name
        Name of the grid dataset

    Returns
    -------
    tuple
        (data, x_coords, y_coords) where x_coords and y_coords may be None
        if axis range attributes are not present
    """
    ds = f[f"grids/{name}"]
    data = ds[()]
    x_coords = None
    y_coords = None

    if "x_range" in ds.attrs and "x_length" in ds.attrs:
        lo, hi = ds.attrs["x_range"]
        n = int(ds.attrs["x_length"])
        x_coords = np.linspace(lo, hi, n)

    if "y_range" in ds.attrs and "y_length" in ds.attrs:
        lo, hi = ds.attrs["y_range"]
        n = int(ds.attrs["y_length"])
        y_coords = np.linspace(lo, hi, n)

    return data, x_coords, y_coords


def read_h5_path(f: Any, name: str) -> ArrayGroup:
    """Read a path (ordered point sequence).

    Parameters
    ----------
    f
        Open HDF5 file object
    name
        Name of the path dataset

    Returns
    -------
    ArrayGroup
        Named mapping of coordinate names to arrays
    """
    g = f[f"paths/{name}"]
    return ArrayGroup(values={k: g[k][()] for k in g.keys()})


def read_h5_points(f: Any, name: str) -> ArrayGroup:
    """Read a point set.

    Parameters
    ----------
    f
        Open HDF5 file object
    name
        Name of the points dataset

    Returns
    -------
    ArrayGroup
        Named mapping of coordinate names to arrays
    """
    g = f[f"points/{name}"]
    return ArrayGroup(values={k: g[k][()] for k in g.keys()})


def read_h5_metadata(f: Any) -> MetadataAttrs:
    """Read root-level metadata attributes.

    Parameters
    ----------
    f
        Open HDF5 file object

    Returns
    -------
    MetadataAttrs
        Named mapping of metadata attributes
    """
    return MetadataAttrs(values={k: f.attrs[k] for k in f.attrs.keys()})


def validate_hdf5_structure(
    f: Any, required_groups: list[str] | None = None
) -> list[str]:
    """Validate HDF5 file has expected structure.

    Parameters
    ----------
    f
        Open HDF5 file object
    required_groups
        List of required group names (default: ["grids", "table"])

    Returns
    -------
    list[str]
        List of missing groups (empty if all present)

    Raises
    ------
    ValueError
        If required groups are missing
    """
    if required_groups is None:
        required_groups = ["grids", "table"]

    missing = [g for g in required_groups if g not in f]
    if missing:
        msg = f"Invalid HDF5 structure. Missing groups: {missing}"
        raise ValueError(msg)
    return missing
