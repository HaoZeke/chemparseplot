# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Physical field on a metric 2D plane.

MethodsX: RMSD ``(r, p)`` (optionally rotated to ``(s, d)``) plus
potential energy and synthetic path-tangent gradients.

Landfold: the χ-embed ``(s1, s2)`` is the plane. The field is a
structural order parameter on the high-D descriptors that built the
map (fcc/ico basin coordinate from coordination counts), not occupancy
invert.
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
    "basin_coordinate",
    "from_descriptor_cloud",
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


def basin_coordinate(descriptors, ref_a, ref_b) -> np.ndarray:
    """Path-independent basin coordinate in descriptor space.

    ``ξ = d(x, a) / (d(x, a) + d(x, b))`` is 0 at ``ref_a`` and 1 at
    ``ref_b``. For LJ38 that is fcc vs Mackay icosahedron in the n4..n13
    coordination-count space the χ map was built from.
    """
    data = np.asarray(descriptors, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ref_a = np.asarray(ref_a, dtype=float).reshape(-1)
    ref_b = np.asarray(ref_b, dtype=float).reshape(-1)
    table_ndim = 2
    if (
        data.ndim != table_ndim
        or data.shape[1] != ref_a.size
        or ref_a.size != ref_b.size
    ):
        msg = "descriptors must be (n, d) with d matching both references"
        raise ValueError(msg)
    finite = (
        np.isfinite(data).all()
        and np.isfinite(ref_a).all()
        and np.isfinite(ref_b).all()
    )
    if not finite:
        msg = "descriptors and references must be finite"
        raise ValueError(msg)
    if float(np.linalg.norm(ref_a - ref_b)) == 0.0:
        msg = "ref_a and ref_b must differ"
        raise ValueError(msg)
    dist_a = np.linalg.norm(data - ref_a, axis=1)
    dist_b = np.linalg.norm(data - ref_b, axis=1)
    denom = dist_a + dist_b
    if np.any(denom <= 0.0):
        msg = "descriptor coincides with both references"
        raise ValueError(msg)
    return dist_a / denom


def from_descriptor_cloud(
    s1,
    s2,
    descriptors,
    ref_a,
    ref_b,
) -> EnergyRepresentation:
    """Landfold plane plus basin coordinate of high-D descriptors."""
    s1 = np.asarray(s1, dtype=float).reshape(-1)
    s2 = np.asarray(s2, dtype=float).reshape(-1)
    xi = basin_coordinate(descriptors, ref_a, ref_b)
    if s1.shape != s2.shape or s1.shape != xi.shape:
        msg = "s1, s2, and descriptors must have the same number of rows"
        raise ValueError(msg)
    return EnergyRepresentation.from_mapping(
        {
            "schema": ENERGY_SCHEMA,
            "x": s1,
            "y": s2,
            "energy": xi,
            "frame": "landfold",
            "xlabel": r"$s_1$",
            "ylabel": r"$s_2$",
            "metadata": {"field": "basin_coordinate"},
        }
    )
