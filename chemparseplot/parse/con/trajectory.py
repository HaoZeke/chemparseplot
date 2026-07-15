# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Load general CON trajectories into typed tables + ASE atoms.

Works for any readcon-core CON/convel (eOn movies, standalone structures,
multi-image NEB paths). Does **not** require a job directory or sidecar
``.dat`` files — metadata-native energies/forces on the frames are used
when present.

.. versionadded:: 1.9.12
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from chemparseplot.parse.con.io import read_con_as_ase, read_con_frames

__all__ = [
    "ConTrajectory",
    "frames_to_table",
    "load_con_trajectory",
]

# Columns we always try to surface from ConFrame first-class fields.
_FRAME_FIELDS = (
    "frame_index",
    "energy",
    "neb_bead",
    "neb_band",
    "time",
    "timestep",
    "potential_type",
    "spec_version",
)


def _frame_field(frame: Any, name: str) -> Any:
    return getattr(frame, name, None)


def _metadata_dict(frame: Any) -> dict[str, Any]:
    meta = getattr(frame, "metadata", None)
    if meta is None:
        return {}
    if hasattr(meta, "items"):
        return {str(k): v for k, v in meta.items()}
    try:
        return dict(meta)
    except Exception:
        return {}


def _force_stats(frame: Any) -> tuple[float | None, float | None]:
    if not getattr(frame, "has_forces", False):
        return None, None
    forces = getattr(frame, "forces_array", None)
    if forces is None:
        return None, None
    arr = np.asarray(forces, dtype=float)
    if arr.size == 0:
        return None, None
    norms = np.linalg.norm(arr.reshape(-1, 3), axis=1)
    return float(np.max(norms)), float(np.sqrt(np.mean(norms**2)))


def frames_to_table(frames: list[Any]):
    """Build a Polars table from readcon ``ConFrame`` objects.

    Columns always include ``frame_index`` (fallback to enumeration) and
    ``n_atoms``. Energy / force stats / NEB ids / time are filled when the
    frame carries them. Extra string metadata keys are promoted to columns
    when present on any frame.
    """
    import polars as pl

    rows: list[dict[str, Any]] = []
    extra_keys: set[str] = set()
    for i, frame in enumerate(frames):
        meta = _metadata_dict(frame)
        extra_keys.update(meta.keys())
        fmax, frms = _force_stats(frame)
        atoms = getattr(frame, "atoms", None)
        n_atoms = len(atoms) if atoms is not None else None
        row: dict[str, Any] = {
            "frame_index": _frame_field(frame, "frame_index"),
            "energy": _frame_field(frame, "energy"),
            "neb_bead": _frame_field(frame, "neb_bead"),
            "neb_band": _frame_field(frame, "neb_band"),
            "time": _frame_field(frame, "time"),
            "timestep": _frame_field(frame, "timestep"),
            "potential_type": _frame_field(frame, "potential_type"),
            "spec_version": _frame_field(frame, "spec_version"),
            "n_atoms": n_atoms,
            "has_forces": bool(getattr(frame, "has_forces", False)),
            "has_velocities": bool(getattr(frame, "has_velocities", False)),
            "fmax": fmax,
            "frms": frms,
        }
        if row["frame_index"] is None:
            row["frame_index"] = i
        # Prefer first-class energy over metadata string duplicate.
        if row["energy"] is None and "energy" in meta:
            try:
                row["energy"] = float(meta["energy"])
            except (TypeError, ValueError):
                pass
        for key, val in meta.items():
            if key in row and row[key] is not None:
                continue
            row[key] = val
        rows.append(row)

    if not rows:
        return pl.DataFrame()

    # Stable column order: core fields first, then sorted extras.
    core = [
        "frame_index",
        "energy",
        "fmax",
        "frms",
        "n_atoms",
        "neb_bead",
        "neb_band",
        "time",
        "timestep",
        "has_forces",
        "has_velocities",
        "potential_type",
        "spec_version",
    ]
    extras = sorted(k for k in extra_keys if k not in core)
    # Keep only columns that appear
    present = [c for c in core if any(c in r for r in rows)]
    present.extend(extras)
    # Fill missing keys for polars
    for r in rows:
        for c in present:
            r.setdefault(c, None)
    return pl.DataFrame(rows).select(present)


@dataclass
class ConTrajectory:
    """In-memory CON trajectory with table + ASE views.

    Attributes
    ----------
    path:
        Source path when loaded from disk (else ``None``).
    frames:
        Raw ``readcon.ConFrame`` objects.
    atoms_list:
        ASE ``Atoms`` per frame (via ``ConFrame.to_ase``).
    table:
        Polars metrics table from :func:`frames_to_table`.
    """

    path: Path | None
    frames: list[Any]
    atoms_list: list[Any]
    table: Any  # pl.DataFrame
    source: str = "readcon"

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def energies(self) -> np.ndarray:
        """Energy column as float array (NaN where missing)."""
        if self.table is None or getattr(self.table, "is_empty", lambda: True)():
            return np.full(self.n_frames, np.nan)
        if "energy" not in self.table.columns:
            return np.full(self.n_frames, np.nan)
        return np.asarray(self.table["energy"].to_numpy(), dtype=float)

    def with_energies(self, energies: list[float] | np.ndarray) -> ConTrajectory:
        """Return a copy with per-frame energies set on frames and table."""
        from chemparseplot.parse.con.io import frame_with_energy

        if len(energies) != len(self.frames):
            msg = f"energies length {len(energies)} != n_frames {len(self.frames)}"
            raise ValueError(msg)
        new_frames = [
            frame_with_energy(fr, float(e)) for fr, e in zip(self.frames, energies)
        ]
        return ConTrajectory(
            path=self.path,
            frames=new_frames,
            atoms_list=self.atoms_list,
            table=frames_to_table(new_frames),
            source=self.source,
        )


def load_con_trajectory(path: str | Path) -> ConTrajectory:
    """Load a CON/convel file into a :class:`ConTrajectory`.

    Parameters
    ----------
    path:
        Path to ``.con`` or ``.convel`` (multi-frame movies supported).
    """
    path = Path(path)
    frames = read_con_frames(path)
    atoms_list = [frame.to_ase() for frame in frames]
    table = frames_to_table(frames)
    return ConTrajectory(
        path=path.resolve(),
        frames=frames,
        atoms_list=atoms_list,
        table=table,
        source="readcon",
    )
