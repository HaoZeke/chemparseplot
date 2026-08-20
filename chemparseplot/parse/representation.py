# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Energy representation on a metric 2D plane (MethodsX).

The science lives in the plane and the field, not in occupancy. RMSD
``(r, p)`` (or a landfold ``(s1, s2)`` map) plus energies and, when
forces exist, synthetic path-tangent gradients. A rigid rotation takes
the RMSD plane to reaction progress ``s`` and orthogonal deviation
``d``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from chemparseplot.parse.neb_utils import compute_synthetic_gradients
from chemparseplot.parse.projection import compute_projection_basis, project_to_sd
from chemparseplot.parse.types import ENERGY_SCHEMA, EnergyRepresentation

__all__ = [
    "ENERGY_SCHEMA",
    "EnergyRepresentation",
    "from_path_forces",
    "load_energy_table",
    "rotate_to_progress",
]


def load_energy_table(
    path: str | Path,
    *,
    frame: str = "plane",
) -> EnergyRepresentation:
    """Read ``# x y energy [f_para] [step]`` (``chemparseplot.energy.v1``)."""
    path = Path(path)
    raw = np.loadtxt(path, comments="#", dtype=float)
    table_ndim = 2
    n_min = 3
    if raw.ndim != table_ndim or raw.shape[1] < n_min:
        msg = f"{path} is not an energy table (need columns x y energy)"
        raise ValueError(msg)
    payload: dict[str, Any] = {
        "schema": ENERGY_SCHEMA,
        "x": raw[:, 0],
        "y": raw[:, 1],
        "energy": raw[:, 2],
        "frame": frame,
        "metadata": {"source": str(path)},
    }
    if raw.shape[1] >= n_min + 1:
        payload["f_para"] = raw[:, 3]
    if raw.shape[1] >= n_min + 2:
        payload["step"] = raw[:, 4]
    return EnergyRepresentation.from_mapping(payload)


def from_path_forces(
    x,
    y,
    energy,
    f_para,
    *,
    frame: str = "rmsd",
    step=None,
    smooth: bool = True,
) -> EnergyRepresentation:
    """Build a representation and attach synthetic ``(grad_x, grad_y)``."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    energy = np.asarray(energy, dtype=float).reshape(-1)
    f_para = np.asarray(f_para, dtype=float).reshape(-1)
    grad_x, grad_y = compute_synthetic_gradients(x, y, f_para, smooth=smooth)
    payload: dict[str, Any] = {
        "schema": ENERGY_SCHEMA,
        "x": x,
        "y": y,
        "energy": energy,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "f_para": f_para,
        "frame": frame,
    }
    if step is not None:
        payload["step"] = step
    return EnergyRepresentation.from_mapping(payload)


def rotate_to_progress(
    rep: EnergyRepresentation | Mapping[str, Any],
) -> EnergyRepresentation:
    """Rotate ``(x, y)`` into MethodsX ``(s, d)``; rotate gradients with it."""
    if not isinstance(rep, EnergyRepresentation):
        rep = EnergyRepresentation.from_mapping(rep)
    basis = compute_projection_basis(rep.x, rep.y)
    s, d = project_to_sd(rep.x, rep.y, basis)
    grad_x = grad_y = None
    if rep.grad_x is not None and rep.grad_y is not None:
        grad_x = rep.grad_x * basis.u_a + rep.grad_y * basis.u_b
        grad_y = rep.grad_x * basis.v_a + rep.grad_y * basis.v_b
    return EnergyRepresentation(
        x=np.asarray(s, dtype=float),
        y=np.asarray(d, dtype=float),
        energy=rep.energy,
        grad_x=grad_x,
        grad_y=grad_y,
        f_para=rep.f_para,
        step=rep.step,
        frame="progress",
        xlabel=r"$s$",
        ylabel=r"$d$",
        schema=ENERGY_SCHEMA,
        metadata={**rep.metadata, "rotated_from": rep.frame},
    )
