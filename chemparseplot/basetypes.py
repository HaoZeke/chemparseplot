# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Typed records for NEB / saddle parse results.

Canonical home for suite basetypes used by parsers and plotters. The hub
(``rgpycrumbs.basetypes``) re-exports these for compatibility.

.. versionadded:: 1.9.11
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DimerOpt",
    "MolGeom",
    "SaddleMeasure",
    "SpinID",
    "nebiter",
    "nebpath",
]


@dataclass(frozen=True, slots=True)
class nebpath:
    """NEB path samples: normalized arc, arc length, energy."""

    norm_dist: float
    arc_dist: float
    energy: float


@dataclass(frozen=True, slots=True)
class nebiter:
    """Typed record for one NEB iteration."""

    iteration: int
    nebpath: nebpath


@dataclass
class DimerOpt:
    """Configuration for a dimer-based saddle point search."""

    saddle: str = "dimer"
    rot: str = "lbfgs"
    trans: str = "lbfgs"


@dataclass
class SpinID:
    """Identifier combining molecule ID and spin state."""

    mol_id: int
    spin: str


@dataclass
class MolGeom:
    """Container for molecular geometry with energy and forces."""

    pos: np.ndarray
    energy: float
    forces: np.ndarray


@dataclass
class SaddleMeasure:
    """Aggregated measurements from a saddle point search."""

    pes_calls: int = 0
    iter_steps: int = 0
    tot_time: float = field(
        default_factory=lambda: datetime.timedelta(0).total_seconds()
    )
    saddle_energy: float = np.nan
    saddle_fmax: float = np.nan
    success: bool = False
    method: str = "not run"
    dimer_rot: str = "n/a"
    dimer_trans: str = "n/a"
    init_energy: float = np.nan
    barrier: float = np.nan
    mol_id: int = np.nan
    spin: str = "unknown"
    scf: float = np.nan
    termination_status: str = "not set"
