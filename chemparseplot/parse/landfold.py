# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Load landfold FES artifacts for chemparseplot.

Landfold owns ``F = -kT ln(rho/rhomax)``. This module only reads the
``landfold.fes.v1`` dict and the CSV from ``landfold fes --csv``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from chemparseplot.parse.types import LANDFOLD_FES_SCHEMA, LandfoldFesResult

__all__ = ["LANDFOLD_FES_SCHEMA", "LandfoldFesResult", "load_fes_csv", "load_fes_result"]


def load_fes_result(data: Mapping[str, Any]) -> LandfoldFesResult:
    """Validate a ``landfold.fes.v1`` mapping from ``fes_xy_result``."""
    return LandfoldFesResult.from_mapping(data)


def load_fes_csv(path: str | Path, *, kt: float = 1.0) -> LandfoldFesResult:
    """Read the four-column ``# x y F rho`` grid written by ``landfold fes``.

    Rows are ``iy``-major, ``ix``-minor, with a blank line after each
    ``y`` row. Empty bins are ``nan`` in ``F``.
    """
    path = Path(path)
    raw = np.loadtxt(path, comments="#", dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 3:
        msg = f"{path} is not a landfold FES CSV (need columns x y F [rho])"
        raise ValueError(msg)
    if raw.shape[0] == 0:
        msg = f"{path} has no FES samples"
        raise ValueError(msg)
    xs = _unique_appearance(raw[:, 0])
    ys = _unique_appearance(raw[:, 1])
    nx, ny = xs.size, ys.size
    if nx * ny != raw.shape[0]:
        msg = (
            f"{path} is not a regular grid: {raw.shape[0]} rows, "
            f"{nx} unique x, {ny} unique y"
        )
        raise ValueError(msg)
    fes = raw[:, 2].reshape(ny, nx)
    if raw.shape[1] >= 4:
        rho = raw[:, 3].reshape(ny, nx)
    else:
        rho = np.zeros_like(fes)
    return LandfoldFesResult.from_mapping(
        {
            "schema": LANDFOLD_FES_SCHEMA,
            "x": xs,
            "y": ys,
            "free_energy": fes,
            "density": rho,
            "kt": kt,
            "metadata": {"source": str(path)},
        }
    )


def _unique_appearance(values: np.ndarray) -> np.ndarray:
    """Unique values in first-seen order (grid centres, not sorted unique)."""
    _, index = np.unique(np.round(values, decimals=12), return_index=True)
    return values[np.sort(index)]
