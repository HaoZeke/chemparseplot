# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Shared typed result objects for parser outputs.

These records preserve mapping-style access for compatibility with existing
callers while giving parser APIs explicit, named return types.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ParserAttrs(Mapping[str, Any]):
    """Named mapping for metadata-style parser records."""

    data: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


@dataclass(frozen=True, slots=True)
class ArrayGroup(Mapping[str, np.ndarray]):
    """Named mapping of arrays loaded from parser backends."""

    data: dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


class DataclassMapping(Mapping[str, Any]):
    """Mixin exposing dataclass fields through the mapping protocol."""

    def __getitem__(self, key: str) -> Any:
        for field_info in fields(self):
            if field_info.name == key:
                return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (field_info.name for field_info in fields(self))

    def __len__(self) -> int:
        return len(fields(self))


@dataclass(frozen=True, slots=True)
class OrcaNebResult(DataclassMapping):
    """Structured ORCA NEB result with mapping-style compatibility."""

    energies: np.ndarray
    rmsd_r: np.ndarray | None = None
    rmsd_p: np.ndarray | None = None
    grad_r: np.ndarray | None = None
    grad_p: np.ndarray | None = None
    forces: list[np.ndarray | None] | None = None
    converged: bool = False
    n_images: int | None = None
    barrier_forward: float | None = None
    barrier_reverse: float | None = None
    source: str = "unknown"
    orca_version: str = "unknown"

    def __post_init__(self) -> None:
        if self.n_images is None:
            object.__setattr__(self, "n_images", len(self.energies))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> OrcaNebResult:
        """Coerce a mapping-like ORCA payload into a typed result."""

        def _maybe_array(key: str) -> np.ndarray | None:
            values = data.get(key)
            if values is None:
                return None
            return np.asarray(values)

        forces = data.get("forces")
        return cls(
            energies=np.asarray(data.get("energies", [])),
            rmsd_r=_maybe_array("rmsd_r"),
            rmsd_p=_maybe_array("rmsd_p"),
            grad_r=_maybe_array("grad_r"),
            grad_p=_maybe_array("grad_p"),
            forces=list(forces) if forces is not None else None,
            converged=bool(data.get("converged", False)),
            n_images=data.get("n_images"),
            barrier_forward=data.get("barrier_forward"),
            barrier_reverse=data.get("barrier_reverse"),
            source=str(data.get("source", "unknown")),
            orca_version=str(data.get("orca_version", "unknown")),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryNebPath(DataclassMapping):
    """Structured ChemGP trajectory path arrays."""

    images: np.ndarray
    energies: np.ndarray
    gradients: np.ndarray
    f_para: np.ndarray
    rxn_coord: np.ndarray


@dataclass(frozen=True, slots=True)
class TrajectoryNebResult(DataclassMapping):
    """Structured ChemGP trajectory NEB result."""

    path: TrajectoryNebPath
    convergence: ArrayGroup
    metadata: ParserAttrs


@dataclass(frozen=True, slots=True)
class PlumedFesResult(DataclassMapping):
    """Structured PLUMED free-energy-surface result."""

    fes: np.ndarray
    hills: np.ndarray
    rows: int
    dimension: int
    per: list[bool] | tuple[bool, ...]
    x: np.ndarray
    y: np.ndarray | None = None
    pcv1: list[float] | tuple[float, ...] | None = None
    pcv2: list[float] | tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class PlumedMinimaResult(DataclassMapping):
    """Structured PLUMED minima result."""

    minima: Any
    fes_result: PlumedFesResult


LANDFOLD_FES_SCHEMA = "landfold.fes.v1"
ENERGY_SCHEMA = "chemparseplot.energy.v1"
_ENERGY_FRAMES = ("plane", "rmsd", "progress", "landfold")


@dataclass(frozen=True, slots=True)
class LandfoldFesResult(DataclassMapping):
    """Structured landfold free-energy-surface result.

    Matches the ``landfold.fes.v1`` Python binding: ``x`` and ``y`` are
    bin centres, ``free_energy`` and ``density`` are ``(ny, nx)``.
    """

    x: np.ndarray
    y: np.ndarray
    free_energy: np.ndarray
    density: np.ndarray
    kt: float = 1.0
    schema: str = LANDFOLD_FES_SCHEMA
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> LandfoldFesResult:
        """Coerce a ``landfold.fes.v1`` dict (or a FES CSV load) into a typed result."""
        schema = str(data.get("schema", LANDFOLD_FES_SCHEMA))
        if schema != LANDFOLD_FES_SCHEMA:
            msg = f"expected schema {LANDFOLD_FES_SCHEMA}, got {schema!r}"
            raise ValueError(msg)
        energy = data.get("free_energy", data.get("fes"))
        if energy is None:
            msg = "landfold FES mapping needs free_energy (or fes)"
            raise ValueError(msg)
        x = np.asarray(data["x"], dtype=float)
        y = np.asarray(data["y"], dtype=float)
        fes = np.asarray(energy, dtype=float)
        if x.ndim != 1 or y.ndim != 1:
            msg = "landfold FES x and y must be 1-D bin centres"
            raise ValueError(msg)
        if fes.shape != (y.size, x.size):
            msg = (
                f"free_energy shape {fes.shape} does not match "
                f"(len(y), len(x)) = {(y.size, x.size)}"
            )
            raise ValueError(msg)
        density = data.get("density")
        if density is None:
            rho = np.zeros_like(fes)
        else:
            rho = np.asarray(density, dtype=float)
            if rho.shape != fes.shape:
                msg = f"density shape {rho.shape} does not match {fes.shape}"
                raise ValueError(msg)
        kt = float(data.get("kt", 1.0))
        if not np.isfinite(kt) or kt <= 0.0:
            msg = "landfold FES kt must be finite and > 0"
            raise ValueError(msg)
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            msg = "landfold FES metadata must be a dict"
            raise TypeError(msg)
        return cls(
            x=x,
            y=y,
            free_energy=fes,
            density=rho,
            kt=kt,
            schema=schema,
            metadata=dict(metadata),
        )


@dataclass(frozen=True, slots=True)
class EnergyRepresentation(DataclassMapping):
    """Metric 2D plane plus a physical energy field.

    The plane is RMSD ``(r, p)``, the rotated ``(s, d)`` frame, or a
    landfold ``(s1, s2)`` map. ``energy`` is potential energy, not
    occupancy invert. Optional ``f_para`` builds synthetic gradients
    along the path tangent (MethodsX).
    """

    x: np.ndarray
    y: np.ndarray
    energy: np.ndarray
    grad_x: np.ndarray | None = None
    grad_y: np.ndarray | None = None
    f_para: np.ndarray | None = None
    step: np.ndarray | None = None
    frame: str = "plane"
    xlabel: str = r"$s_1$"
    ylabel: str = r"$s_2$"
    schema: str = ENERGY_SCHEMA
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> EnergyRepresentation:
        """Coerce a ``chemparseplot.energy.v1`` mapping."""
        schema = str(data.get("schema", ENERGY_SCHEMA))
        if schema != ENERGY_SCHEMA:
            msg = f"expected schema {ENERGY_SCHEMA}, got {schema!r}"
            raise ValueError(msg)
        if "energy" not in data:
            msg = "energy representation needs energy"
            raise ValueError(msg)
        x = np.asarray(data["x"], dtype=float).reshape(-1)
        y = np.asarray(data["y"], dtype=float).reshape(-1)
        energy = np.asarray(data["energy"], dtype=float).reshape(-1)
        if x.shape != y.shape or x.shape != energy.shape:
            msg = "energy representation x, y, and energy must have the same length"
            raise ValueError(msg)
        frame = str(data.get("frame", "plane"))
        if frame not in _ENERGY_FRAMES:
            msg = f"frame must be one of {_ENERGY_FRAMES}, got {frame!r}"
            raise ValueError(msg)

        def _opt(key: str) -> np.ndarray | None:
            values = data.get(key)
            if values is None:
                return None
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.shape != x.shape:
                msg = f"{key} must match x/y/energy"
                raise ValueError(msg)
            return arr

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            msg = "energy representation metadata must be a dict"
            raise TypeError(msg)
        xlabel = str(data.get("xlabel", r"$s_1$"))
        ylabel = str(data.get("ylabel", r"$s_2$"))
        if frame == "rmsd":
            xlabel = str(data.get("xlabel", r"RMSD-R"))
            ylabel = str(data.get("ylabel", r"RMSD-P"))
        elif frame == "progress":
            xlabel = str(data.get("xlabel", r"$s$"))
            ylabel = str(data.get("ylabel", r"$d$"))
        return cls(
            x=x,
            y=y,
            energy=energy,
            grad_x=_opt("grad_x"),
            grad_y=_opt("grad_y"),
            f_para=_opt("f_para"),
            step=_opt("step"),
            frame=frame,
            xlabel=xlabel,
            ylabel=ylabel,
            schema=schema,
            metadata=dict(metadata),
        )
