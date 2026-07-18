# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""ConFrame sequence → series / trajectory DTOs for eOn plots.

The plot packet is an ordered sequence of ``readcon.ConFrame`` (or duck-typed
frames with ``energy``, ``metadata``, ``to_ase`` / geometry). ASE and Matter
are adapters into this wire format; this module does not invent a second schema.

```{versionadded} 1.9.14
```
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from ase import Atoms

from chemparseplot.parse.eon._trajectory_common import (
    frame_rows_to_table,
    metadata_value,
)
from chemparseplot.parse.eon.dimer_trajectory import (
    DimerTrajectoryData,
    table_from_dimer_metadata,
)
from chemparseplot.parse.eon.min_trajectory import (
    MinTrajectoryData,
    table_from_min_metadata,
)

__all__ = [
    "atoms_list_from_frames",
    "dimer_trajectory_from_frames",
    "energies_from_frames",
    "min_trajectory_from_frames",
    "neb_path_arrays",
    "series_from_frames",
]


def energies_from_frames(frames: Sequence[Any]) -> np.ndarray:
    """Total energy (eV) per frame; raises if any frame lacks energy."""
    if not frames:
        msg = "energies_from_frames: empty frame sequence"
        raise ValueError(msg)
    out = np.empty(len(frames), dtype=float)
    for i, frame in enumerate(frames):
        e = metadata_value(frame, "energy")
        if e is None:
            msg = f"frame {i} has no energy stamp"
            raise ValueError(msg)
        out[i] = float(e)
    return out


def atoms_list_from_frames(frames: Sequence[Any]) -> list[Atoms]:
    """Convert frames to ASE Atoms with energy stored in ``atoms.info``."""
    atoms_list: list[Atoms] = []
    for i, frame in enumerate(frames):
        to_ase = getattr(frame, "to_ase", None)
        if not callable(to_ase):
            msg = f"frame {i} has no to_ase(); need readcon.ConFrame or equivalent"
            raise TypeError(msg)
        atoms = to_ase()
        e = metadata_value(frame, "energy")
        if e is not None:
            atoms.info["energy"] = float(e)
        atoms_list.append(atoms)
    return atoms_list


def series_from_frames(
    frames: Sequence[Any],
    columns: Sequence[str],
    *,
    allow_leading_incomplete: bool = False,
) -> pl.DataFrame:
    """Build a polars table of metadata columns from frames (empty if incomplete)."""
    return frame_rows_to_table(
        frames,
        columns,
        allow_leading_incomplete=allow_leading_incomplete,
    )


def neb_path_arrays(frames: Sequence[Any]) -> dict[str, Any]:
    """Extract NEB path series from stamped frames (writePathCon / path_frames).

    Returns
    -------
    dict
        ``atoms_list``, ``energies``, optional ``reaction_coordinate``,
        ``relative_energy``, ``parallel_force``, ``neb_bead``, ``frame_index``.
    """
    if not frames:
        msg = "neb_path_arrays: empty frame sequence"
        raise ValueError(msg)
    atoms_list = atoms_list_from_frames(frames)
    energies = energies_from_frames(frames)
    result: dict[str, Any] = {
        "atoms_list": atoms_list,
        "energies": energies,
        "n_frames": len(frames),
    }
    for key in (
        "reaction_coordinate",
        "relative_energy",
        "parallel_force",
        "neb_bead",
        "frame_index",
        "lowest_eigenvalue",
    ):
        vals = []
        ok = True
        for frame in frames:
            v = metadata_value(frame, key)
            if v is None and key in (
                "reaction_coordinate",
                "relative_energy",
                "parallel_force",
            ):
                # optional NEB stamps
                ok = False
                break
            vals.append(v)
        if ok and any(v is not None for v in vals):
            result[key] = np.array(
                [np.nan if v is None else float(v) for v in vals],
                dtype=float,
            )
    return result


def min_trajectory_from_frames(frames: Sequence[Any]) -> MinTrajectoryData:
    """Build :class:`MinTrajectoryData` from stamped minimization ConFrames."""
    if not frames:
        msg = "min_trajectory_from_frames: empty frame sequence"
        raise ValueError(msg)
    atoms_list = atoms_list_from_frames(frames)
    dat_df = table_from_min_metadata(list(frames))
    if dat_df.is_empty():
        # Minimal table from energy + sequential iteration
        energies = energies_from_frames(frames)
        dat_df = pl.DataFrame(
            {
                "iteration": list(range(len(frames))),
                "step_size": [0.0] * len(frames),
                "convergence": [0.0] * len(frames),
                "energy": energies.tolist(),
            }
        )
    return MinTrajectoryData(
        atoms_list=atoms_list,
        dat_df=dat_df,
        initial_atoms=atoms_list[0],
        final_atoms=atoms_list[-1],
    )


def dimer_trajectory_from_frames(frames: Sequence[Any]) -> DimerTrajectoryData:
    """Build :class:`DimerTrajectoryData` from stamped climb ConFrames."""
    if not frames:
        msg = "dimer_trajectory_from_frames: empty frame sequence"
        raise ValueError(msg)
    atoms_list = atoms_list_from_frames(frames)
    dat_df = table_from_dimer_metadata(list(frames))
    if dat_df.is_empty():
        energies = energies_from_frames(frames)
        e0 = float(energies[0])
        dat_df = pl.DataFrame(
            {
                "iteration": list(range(len(frames))),
                "step_size": [0.0] * len(frames),
                "delta_e": (energies - e0).tolist(),
                "convergence": [0.0] * len(frames),
                "eigenvalue": [0.0] * len(frames),
                "torque": [0.0] * len(frames),
                "angle": [0.0] * len(frames),
                "rotations": [0] * len(frames),
            }
        )
    return DimerTrajectoryData(
        atoms_list=atoms_list,
        dat_df=dat_df,
        initial_atoms=atoms_list[0],
        saddle_atoms=atoms_list[-1],
        mode_vector=None,
    )
