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
            object.__setattr__(self, "n_images", int(len(self.energies)))

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
class TrajectoryNebResult(DataclassMapping):
    """Structured ChemGP trajectory NEB result."""

    path: ArrayGroup
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
