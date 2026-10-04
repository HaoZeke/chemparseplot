# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Producer-independent records of surrogate-assisted saddle searches.

A surrogate-assisted search spends calls on an expensive *oracle* (the true
potential) and answers the rest from a fitted surrogate. The classes here
hold what a plot needs, whichever code produced it. NaN marks a quantity the
producer did not record.

Units: energy in eV, force in eV/A, length in A, curvature in eV/A^2,
time in s.

.. versionadded:: 1.10.0
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


@dataclass
class BandSnapshot:
    """A path of images at one outer iteration.

    ``coordinate`` is the path coordinate in A (0 at the reactant);
    ``sigma`` and ``true_*`` hold NaN where not recorded or not evaluated.
    """

    coordinate: np.ndarray
    energy: np.ndarray
    sigma: np.ndarray | None = None
    true_energy: np.ndarray | None = None
    true_force: np.ndarray | None = None
    climbing: int | None = None
    outer: int = -1
    oracle_calls: int = -1
    positions: np.ndarray | None = None

    def __post_init__(self):
        self.coordinate = _arr(self.coordinate)
        self.energy = _arr(self.energy)
        n = len(self.coordinate)
        for name in ("sigma", "true_energy", "true_force"):
            v = getattr(self, name)
            if v is not None:
                v = _arr(v)
                if v.shape != (n,):
                    msg = f"{name} has shape {v.shape}, expected ({n},)"
                    raise ValueError(msg)
                setattr(self, name, v)
        if self.energy.shape != (n,):
            msg = "energy and coordinate differ in length"
            raise ValueError(msg)

    @property
    def n_images(self) -> int:
        return len(self.coordinate)

    @property
    def evaluated(self) -> np.ndarray:
        if self.true_energy is None:
            return np.zeros(self.n_images, dtype=bool)
        return np.isfinite(self.true_energy)


@dataclass
class EvaluatedPoints:
    """Oracle observations in acquisition order."""

    energy: np.ndarray
    max_force: np.ndarray | None = None
    positions: np.ndarray | None = None
    coordinate: np.ndarray | None = None
    # Cartesian distance (A) of each observation from the final path.
    distance: np.ndarray | None = None
    # Cartesian energy gradients (eV/A), one row per observation.
    gradients: np.ndarray | None = None
    n_calls: int | None = None

    def __post_init__(self):
        self.energy = _arr(self.energy)

    def __len__(self) -> int:
        return len(self.energy)


@dataclass
class AcquisitionEvent:
    """One decision to call the oracle."""

    oracle_calls: int
    kind: str
    outer: int = -1
    image: int = -1
    detail: str = ""


@dataclass
class BandHistory:
    """Two-ended search: path snapshots, observations and decisions."""

    final: BandSnapshot
    snapshots: list[BandSnapshot] = field(default_factory=list)
    points: EvaluatedPoints | None = None
    reference: BandSnapshot | None = None
    events: list[AcquisitionEvent] = field(default_factory=list)
    # Geometry the search reports as its saddle (flat Cartesian, A) and whether
    # it passed the producer's acceptance checks.
    saddle: np.ndarray | None = None
    saddle_certified: bool | None = None
    # Atomic numbers of the atoms (same order as the positions), where they came
    # from, and the files looked at when none was found.
    numbers: np.ndarray | None = None
    numbers_source: str = ""
    numbers_looked_for: list[str] = field(default_factory=list)


@dataclass
class SearchHistory:
    """Per-outer (or per-call) series of a search.

    ``series`` keys use the standard names ``oracle_calls``, ``outer``,
    ``true_force``, ``surrogate_force``, ``surrogate_force_sigma``, ``energy``,
    ``n_train``, ``magnitude_sigma2``, ``noise_sigma2``, ``length_scale_min``,
    ``length_scale_max``, ``time_refit``, ``time_predict``, ``time_inner``,
    ``rss_mb``. Missing quantities are absent, never zero-filled.
    """

    series: dict[str, np.ndarray]
    force_tolerance: float | None = None
    converged: bool | None = None
    events: list[AcquisitionEvent] = field(default_factory=list)
    retained_cap: int | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.series = {k: _arr(v) for k, v in self.series.items()}

    def __getitem__(self, key: str) -> np.ndarray:
        return self.series[key]

    def get(self, key: str) -> np.ndarray | None:
        return self.series.get(key)

    @property
    def oracle_calls(self) -> np.ndarray:
        return self.series["oracle_calls"]

    def to_frame(self, method: str = "search"):
        """Long-form pandas frame (columns ``oracle_calls``, ``max_force``,
        ``kind``, ``method``) for
        :func:`chemparseplot.plot.chemgp.plot_convergence_curve`.
        """
        import pandas as pd

        rows = []
        for key, kind in (("true_force", "true"), ("surrogate_force", "surrogate")):
            v = self.series.get(key)
            if v is None:
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "oracle_calls": self.oracle_calls,
                        "max_fatom": v,
                        "kind": kind,
                        "method": f"{method} ({kind})",
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)


@dataclass
class SingleEndedHistory:
    """Dimer-type search against oracle calls."""

    oracle_calls: np.ndarray
    force: np.ndarray
    curvature: np.ndarray
    curvature_measured: np.ndarray | None = None
    curvature_surrogate: np.ndarray | None = None
    rotation_angle: np.ndarray | None = None
    escape: list[AcquisitionEvent] = field(default_factory=list)
    force_tolerance: float | None = None
    converged: bool | None = None

    def __post_init__(self):
        self.oracle_calls = _arr(self.oracle_calls)
        self.force = _arr(self.force)
        self.curvature = _arr(self.curvature)
        for name in ("curvature_measured", "curvature_surrogate", "rotation_angle"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, _arr(v))


@dataclass
class CellRecord:
    """One reaction of a campaign."""

    label: str
    set: str = ""
    method: str = ""
    status: str = ""
    converged: bool | None = None
    passed: bool | None = None
    index: int | None = None
    search_calls: int | None = None
    validation_calls: int | None = None
    total_calls: int | None = None
    wall: dict[str, float] = field(default_factory=dict)
    energy_delta: float | None = None
    ledger: dict[str, int] = field(default_factory=dict)
    extras: dict = field(default_factory=dict)


@dataclass
class CampaignTable:
    """One row per reaction, plus calls of comparison methods."""

    name: str
    cells: list[CellRecord]
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    tolerance: float | None = None

    def by_label(self) -> dict[str, CellRecord]:
        return {c.label: c for c in self.cells}

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "label": c.label,
                    "set": c.set,
                    "method": c.method,
                    "status": c.status,
                    "converged": c.converged,
                    "passed": c.passed,
                    "index": c.index,
                    "search_calls": c.search_calls,
                    "validation_calls": c.validation_calls,
                    "total_calls": c.total_calls,
                    **{f"wall_{k}": v for k, v in c.wall.items()},
                }
                for c in self.cells
            ]
        )


@dataclass
class ScalingTable:
    """Time against worker count for several series.

    ``reference`` names the series whose first point defines speedup 1 for all
    series; without it each series is measured against its own first point.
    """

    series: dict[str, tuple[np.ndarray, np.ndarray]]
    reference: str | None = None
    # Optional per-point extras, keyed like ``series``: slowest/fastest
    # repetition (s), oracle calls of the run, and runs that hit a cap.
    time_min: dict[str, np.ndarray] = field(default_factory=dict)
    time_max: dict[str, np.ndarray] = field(default_factory=dict)
    calls: dict[str, np.ndarray] = field(default_factory=dict)
    capped: dict[str, np.ndarray] = field(default_factory=dict)
    # ``ranks x threads`` label of each point, keyed like ``series``.
    layouts: dict[str, list[str]] = field(default_factory=dict)

    def speedup(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Speedup of ``name`` against its own first point.

        With ``reference`` set, every series is measured against the first
        point of that series instead, so series that split one total (search
        and validation of a pipeline) share a baseline.
        """
        w, t = self.series[name]
        w, t = _arr(w), _arr(t)
        _, rt = self.series[self.reference or name]
        return w, _arr(rt)[0] / t

    def speedup_range(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """(low, high) speedup from the slowest and fastest repetition."""
        if name not in self.time_min or name not in self.time_max:
            return None
        _, rt = self.series[self.reference or name]
        r0 = _arr(rt)[0]
        return r0 / _arr(self.time_max[name]), r0 / _arr(self.time_min[name])


@dataclass
class SurrogateSearch:
    """All records of one search, with provenance."""

    label: str
    producer: str
    band: BandHistory | None = None
    search: SearchHistory | None = None
    single_ended: SingleEndedHistory | None = None
    cell: CellRecord | None = None
    provenance: dict = field(default_factory=dict)
