# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Parsers for ChemGP JSONL output formats.

ChemGP Rust examples produce JSONL files with method comparison data,
GP quality grids, and RFF approximation benchmarks. This module provides
structured parsing into typed containers for downstream plotting.

.. versionadded:: 1.5.0
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from chemparseplot.parse.types import ParserAttrs


@dataclass(frozen=True, slots=True)
class ComparisonRecord:
    """One optimizer-step record from a comparison JSONL."""

    method: str
    oracle_calls: int
    step: int | None = None
    energy: float | None = None
    force: float | None = None
    max_force: float | None = None

    @classmethod
    def from_mapping(cls, rec: ParserAttrs) -> ComparisonRecord:
        return cls(
            method=str(rec["method"]),
            oracle_calls=int(rec["oracle_calls"]),
            step=int(rec["step"]) if "step" in rec else None,
            energy=float(rec["energy"]) if "energy" in rec else None,
            force=float(rec["force"]) if "force" in rec else None,
            max_force=float(rec["max_force"]) if "max_force" in rec else None,
        )


@dataclass(slots=True)
class OptimizerTrace:
    """Single optimizer trace from a comparison JSONL.

    Attributes
    ----------
    method : str
        Optimizer name (e.g. ``"gp_minimize"``, ``"neb"``, ``"otgpd"``).
    steps : list[int]
        Step indices.
    oracle_calls : list[int]
        Cumulative oracle call counts.
    energies : list[float] | None
        Energy at each step (minimize, dimer).
    forces : list[float] | None
        Force norm at each step (dimer: ``force``, NEB: ``max_force``).
    """

    method: str
    steps: list[int] = field(default_factory=list)
    oracle_calls: list[int] = field(default_factory=list)
    energies: list[float] | None = None
    forces: list[float] | None = None

    def add_record(self, rec: ComparisonRecord) -> None:
        """Accumulate one typed optimizer record into the trace."""

        self.steps.append(rec.step if rec.step is not None else len(self.steps))
        self.oracle_calls.append(rec.oracle_calls)
        if rec.energy is not None:
            if self.energies is None:
                self.energies = []
            self.energies.append(rec.energy)
        force_value = rec.force if rec.force is not None else rec.max_force
        if force_value is not None:
            if self.forces is None:
                self.forces = []
            self.forces.append(force_value)


@dataclass(slots=True)
class ComparisonData:
    """Parsed optimizer comparison from a single JSONL file.

    Attributes
    ----------
    traces : dict[str, OptimizerTrace]
        Keyed by method name.
    summary : ParserAttrs | None
        Summary record if present.
    """

    traces: dict[str, OptimizerTrace] = field(default_factory=dict)
    summary: ParserAttrs | None = None

    def ensure_trace(self, method: str) -> OptimizerTrace:
        """Return the named optimizer trace, creating it if needed."""

        if method not in self.traces:
            self.traces[method] = OptimizerTrace(method=method)
        return self.traces[method]


def parse_comparison_jsonl(path: str | Path) -> ComparisonData:
    """Parse a ChemGP optimizer comparison JSONL file.

    Handles minimize, dimer, and NEB comparison formats. Each line is a
    JSON object with a ``method`` field (or ``summary: true``).

    Parameters
    ----------
    path
        Path to the JSONL file.

    Returns
    -------
    ComparisonData
        Parsed traces keyed by method name.
    """
    data = ComparisonData()
    with open(path) as f:
        for line in f:
            rec = ParserAttrs(data=json.loads(line.strip()))
            if rec.get("summary"):
                data.summary = rec
                continue
            record = ComparisonRecord.from_mapping(rec)
            trace = data.ensure_trace(record.method)
            trace.add_record(record)
    return data


@dataclass(frozen=True, slots=True)
class RFFExactRecord:
    """Exact-GP benchmark reference record."""

    energy_mae: float
    gradient_mae: float

    @classmethod
    def from_mapping(cls, rec: ParserAttrs) -> RFFExactRecord:
        return cls(
            energy_mae=float(rec["energy_mae"]),
            gradient_mae=float(rec["gradient_mae"]),
        )


@dataclass(frozen=True, slots=True)
class RFFApproxRecord:
    """One random-feature approximation benchmark record."""

    d_rff: int
    energy_mae_vs_true: float
    gradient_mae_vs_true: float
    energy_mae_vs_gp: float
    gradient_mae_vs_gp: float

    @classmethod
    def from_mapping(cls, rec: ParserAttrs) -> RFFApproxRecord:
        return cls(
            d_rff=int(rec["d_rff"]),
            energy_mae_vs_true=float(rec["energy_mae_vs_true"]),
            gradient_mae_vs_true=float(rec["gradient_mae_vs_true"]),
            energy_mae_vs_gp=float(rec["energy_mae_vs_gp"]),
            gradient_mae_vs_gp=float(rec["gradient_mae_vs_gp"]),
        )


@dataclass(slots=True)
class RFFQualityData:
    """Parsed RFF approximation quality data.

    Attributes
    ----------
    exact_energy_mae : float
        Exact GP energy MAE vs true surface.
    exact_gradient_mae : float
        Exact GP gradient MAE vs true surface.
    d_rff_values : list[int]
        RFF feature counts tested.
    energy_mae_vs_true : list[float]
        RFF energy MAE vs true surface.
    gradient_mae_vs_true : list[float]
        RFF gradient MAE vs true surface.
    energy_mae_vs_gp : list[float]
        RFF energy MAE vs exact GP.
    gradient_mae_vs_gp : list[float]
        RFF gradient MAE vs exact GP.
    """

    exact_energy_mae: float = 0.0
    exact_gradient_mae: float = 0.0
    d_rff_values: list[int] = field(default_factory=list)
    energy_mae_vs_true: list[float] = field(default_factory=list)
    gradient_mae_vs_true: list[float] = field(default_factory=list)
    energy_mae_vs_gp: list[float] = field(default_factory=list)
    gradient_mae_vs_gp: list[float] = field(default_factory=list)

    def add_exact_gp(self, rec: RFFExactRecord) -> None:
        """Store the exact-GP reference metrics."""

        self.exact_energy_mae = rec.energy_mae
        self.exact_gradient_mae = rec.gradient_mae

    def add_rff(self, rec: RFFApproxRecord) -> None:
        """Store one RFF approximation record."""

        self.d_rff_values.append(rec.d_rff)
        self.energy_mae_vs_true.append(rec.energy_mae_vs_true)
        self.gradient_mae_vs_true.append(rec.gradient_mae_vs_true)
        self.energy_mae_vs_gp.append(rec.energy_mae_vs_gp)
        self.gradient_mae_vs_gp.append(rec.gradient_mae_vs_gp)


def parse_rff_quality_jsonl(path: str | Path) -> RFFQualityData:
    """Parse a ChemGP RFF quality JSONL file.

    Parameters
    ----------
    path
        Path to the JSONL file.

    Returns
    -------
    RFFQualityData
        Parsed exact GP and RFF metrics.
    """
    data = RFFQualityData()
    with open(path) as f:
        for line in f:
            rec = ParserAttrs(data=json.loads(line.strip()))
            if rec["type"] == "exact_gp":
                data.add_exact_gp(RFFExactRecord.from_mapping(rec))
            elif rec["type"] == "rff":
                data.add_rff(RFFApproxRecord.from_mapping(rec))
    return data


@dataclass(slots=True)
class GPQualityGrid:
    """GP quality grid data for a single training set size.

    Attributes
    ----------
    n_train : int
        Number of training points.
    nx : int
        Grid x resolution.
    ny : int
        Grid y resolution.
    x : list[list[float]]
        Grid x coordinates (ny x nx).
    y : list[list[float]]
        Grid y coordinates (ny x nx).
    true_e : list[list[float]]
        True energy on grid.
    gp_e : list[list[float]]
        GP predicted energy on grid.
    gp_var : list[list[float]]
        GP variance on grid.
    train_x : list[float]
        Training point x coordinates.
    train_y : list[float]
        Training point y coordinates.
    train_e : list[float]
        Training point energies.
    """

    n_train: int = 0
    nx: int = 0
    ny: int = 0
    x: list[list[float]] = field(default_factory=list)
    y: list[list[float]] = field(default_factory=list)
    true_e: list[list[float]] = field(default_factory=list)
    gp_e: list[list[float]] = field(default_factory=list)
    gp_var: list[list[float]] = field(default_factory=list)
    train_x: list[float] = field(default_factory=list)
    train_y: list[float] = field(default_factory=list)
    train_e: list[float] = field(default_factory=list)

    @classmethod
    def from_records(
        cls,
        *,
        n_train: int,
        meta: ParserAttrs,
        records: list[GPGridRecord],
        train_points: TrainingPointSet | None = None,
    ) -> GPQualityGrid:
        """Build a typed grid from JSONL records and parsed metadata."""

        nx = int(meta["nx"]) if "nx" in meta else 0
        ny = int(meta["ny"]) if "ny" in meta else 0
        grid = cls(n_train=n_train, nx=nx, ny=ny)
        grid.x = [[0.0] * nx for _ in range(ny)]
        grid.y = [[0.0] * nx for _ in range(ny)]
        grid.true_e = [[0.0] * nx for _ in range(ny)]
        grid.gp_e = [[0.0] * nx for _ in range(ny)]
        grid.gp_var = [[0.0] * nx for _ in range(ny)]

        for rec in records:
            grid.x[rec.iy][rec.ix] = rec.x
            grid.y[rec.iy][rec.ix] = rec.y
            grid.true_e[rec.iy][rec.ix] = rec.true_e
            grid.gp_e[rec.iy][rec.ix] = rec.gp_e
            grid.gp_var[rec.iy][rec.ix] = rec.gp_var

        if train_points is not None:
            grid.train_x = list(train_points.x)
            grid.train_y = list(train_points.y)
            grid.train_e = list(train_points.e)
        return grid


@dataclass(frozen=True, slots=True)
class GPGridRecord:
    """One grid-sample record from the GP-quality JSONL."""

    ix: int
    iy: int
    x: float
    y: float
    true_e: float
    gp_e: float
    gp_var: float

    @classmethod
    def from_mapping(cls, rec: ParserAttrs) -> GPGridRecord:
        return cls(
            ix=int(rec["ix"]),
            iy=int(rec["iy"]),
            x=float(rec["x"]),
            y=float(rec["y"]),
            true_e=float(rec["true_e"]),
            gp_e=float(rec["gp_e"]),
            gp_var=float(rec["gp_var"]),
        )


@dataclass(slots=True)
class TrainingPointSet:
    """Accumulated training points for a single ``n_train`` value."""

    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    e: list[float] = field(default_factory=list)

    def append(self, *, x: float, y: float, energy: float) -> None:
        self.x.append(x)
        self.y.append(y)
        self.e.append(energy)


@dataclass(frozen=True, slots=True)
class StationaryPoint:
    """A stationary point (minimum or saddle) on the PES."""

    kind: str  # "minimum" or "saddle"
    id: int
    x: float
    y: float
    energy: float


@dataclass(slots=True)
class GPQualityData:
    """Complete GP quality data from mb_gp_quality.jsonl.

    Attributes
    ----------
    meta : ParserAttrs
        Grid metadata (nx, ny, x_min, x_max, y_min, y_max).
    stationary : list[StationaryPoint]
        Minima and saddle points.
    grids : dict[int, GPQualityGrid]
        Grid data keyed by n_train.
    """

    meta: ParserAttrs = field(default_factory=ParserAttrs)
    stationary: list[StationaryPoint] = field(default_factory=list)
    grids: dict[int, GPQualityGrid] = field(default_factory=dict)


def parse_gp_quality_jsonl(path: str | Path) -> GPQualityData:
    """Parse a ChemGP GP quality JSONL file.

    Parameters
    ----------
    path
        Path to the JSONL file (e.g. ``mb_gp_quality.jsonl``).

    Returns
    -------
    GPQualityData
        Structured grid data with metadata and stationary points.
    """
    data = GPQualityData()
    train_points = defaultdict(TrainingPointSet)
    grid_records = defaultdict(list)

    with open(path) as f:
        for line in f:
            rec = ParserAttrs(data=json.loads(line.strip()))
            t = rec["type"]
            if t == "grid_meta":
                data.meta = rec
            elif t in ("minimum", "saddle"):
                data.stationary.append(
                    StationaryPoint(
                        kind=t,
                        id=int(rec["id"]),
                        x=float(rec["x"]),
                        y=float(rec["y"]),
                        energy=float(rec["energy"]),
                    )
                )
            elif t == "train_point":
                n = int(rec["n_train"])
                train_points[n].append(
                    x=float(rec["x"]),
                    y=float(rec["y"]),
                    energy=float(rec["energy"]),
                )
            elif t == "grid":
                grid_records[int(rec["n_train"])].append(GPGridRecord.from_mapping(rec))

    for n_train, records in grid_records.items():
        data.grids[n_train] = GPQualityGrid.from_records(
            n_train=n_train,
            meta=data.meta,
            records=records,
            train_points=train_points.get(n_train),
        )

    return data
