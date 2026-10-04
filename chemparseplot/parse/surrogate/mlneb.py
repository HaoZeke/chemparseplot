# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Parser for ASE/CatLearn-style ML-NEB output.

The producer writes into a run directory

- ``predicted.traj``: the surrogate band of every iteration, one frame per
  image, ``info["results"]["uncertainty"]`` the predictive standard deviation;
- ``evaluated.traj``: the geometries sent to the oracle (reactant, product,
  then one per step) with true energies and forces;
- ``ml_summary.txt`` / ``ml_time.txt``: pipe tables per step.

Needs ``ase``.

.. versionadded:: 1.10.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chemparseplot.parse.surrogate.gpr_optim import project_on_path, sha256_file
from chemparseplot.parse.surrogate.model import (
    AcquisitionEvent,
    BandHistory,
    BandSnapshot,
    CellRecord,
    EvaluatedPoints,
    SearchHistory,
    SurrogateSearch,
)

# Largest coordinate difference (A) for an evaluated geometry to sit on an image.
MATCH_TOLERANCE = 0.05


def read_pipe_table(path: str | Path) -> dict[str, np.ndarray]:
    """Read a ``| a | b |`` table into columns; ``nan`` stays NaN and a
    ``Date`` column is kept as text."""
    rows = [
        [c.strip() for c in ln.strip().strip("|").split("|")]
        for ln in Path(path).read_text().splitlines()
        if ln.strip().startswith("|")
    ]
    head, body = rows[0], rows[1:]
    cols: dict[str, np.ndarray] = {}
    for j, name in enumerate(head):
        raw = [r[j] for r in body]
        cols[name] = (
            np.array(raw) if name == "Date" else np.array([float(x) for x in raw])
        )
    return cols


def _band_length(frames) -> int:
    first = frames[0].positions
    for i in range(1, len(frames)):
        if np.array_equal(frames[i].positions, first):
            return i
    return len(frames)


def _fmax(forces: np.ndarray) -> float:
    return float(np.linalg.norm(forces, axis=1).max())


def parse_mlneb_run(
    run_dir: str | Path,
    *,
    label: str | None = None,
    force_tolerance: float | None = None,
) -> SurrogateSearch:
    """Parse an ML-NEB run directory into a :class:`SurrogateSearch`.

    Parameters
    ----------
    run_dir
        Directory with ``predicted.traj``, ``evaluated.traj`` and
        ``ml_summary.txt`` (``ml_time.txt`` is optional).
    force_tolerance
        Convergence threshold in eV/A; the files do not record it.
    """
    from ase.io import read

    run = Path(run_dir)
    pred = read(run / "predicted.traj", ":")
    ev = read(run / "evaluated.traj", ":")
    summary = read_pipe_table(run / "ml_summary.txt")
    n = _band_length(pred)
    if len(pred) % n:
        msg = f"{len(pred)} predicted frames do not divide into bands of {n}"
        raise ValueError(msg)
    ev_pos = np.array([a.positions.ravel() for a in ev])
    ev_e = np.array([a.get_potential_energy() for a in ev])
    ev_f = np.array([_fmax(a.get_forces()) for a in ev])
    ev_g = np.array([a.get_forces().ravel() for a in ev])
    snapshots: list[BandSnapshot] = []
    events: list[AcquisitionEvent] = []
    surrogate_force, picked = [], []
    for k in range(len(pred) // n):
        frames = pred[k * n : (k + 1) * n]
        pos = np.array([a.positions.ravel() for a in frames])
        coord = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
        )
        energy = np.array([a.get_potential_energy() for a in frames])
        sigma = np.array(
            [a.info.get("results", {}).get("uncertainty", np.nan) for a in frames]
        )
        known = min(k + 3, len(ev))  # reactant, product and k + 1 steps trained on
        true_e = np.full(n, np.nan)
        true_f = np.full(n, np.nan)
        for j in range(known):
            d = np.abs(pos - ev_pos[j]).max(axis=1)
            i = int(d.argmin())
            if d[i] < MATCH_TOLERANCE:
                true_e[i], true_f[i] = ev_e[j], ev_f[j]
        interior = energy[1:-1]
        ci = int(interior.argmax()) + 1 if len(interior) else None
        snapshots.append(
            BandSnapshot(
                coord,
                energy,
                sigma=sigma,
                true_energy=true_e,
                true_force=true_f,
                climbing=ci,
                outer=k,
                oracle_calls=known,
                positions=pos,
            )
        )
        if known < len(ev):
            d = np.abs(pos - ev_pos[known]).max(axis=1)
            i = int(d.argmin())
            image = i if d[i] < MATCH_TOLERANCE else -1
            events.append(AcquisitionEvent(known + 1, "band acquisition", k, image))
            picked.append(known)
            surrogate_force.append(
                _fmax(frames[i].get_forces()) if image >= 0 else np.nan
            )
    steps = summary["Step"].astype(int)
    calls = steps + 2  # two endpoint evaluations precede step 1
    series = {
        "oracle_calls": calls.astype(float),
        "outer": steps.astype(float),
        "true_force": summary["True fmax/[eV/Å]"],
        "energy": summary["True energy/[eV]"],
        "energy_sigma": summary["Uncertainty/[eV]"],
        "energy_error": summary["True error/[eV]"],
        "n_train": calls.astype(float),
    }
    sf = np.full(len(steps), np.nan)
    for p, v in zip(picked, surrogate_force, strict=True):
        if p - 2 < len(sf):
            sf[p - 2] = v
    if np.isfinite(sf).any():
        series["surrogate_force"] = sf
    if (run / "ml_time.txt").is_file():
        t = read_pipe_table(run / "ml_time.txt")
        series["time_refit"] = t["ML training/[s]"]
        series["time_predict"] = t["ML run/[s]"]
        series["time_oracle"] = t["Evaluation/[s]"]
    final = snapshots[-1]
    pc, pd = project_on_path(ev_pos, final.positions, final.coordinate)
    points = EvaluatedPoints(
        energy=ev_e,
        max_force=ev_f,
        positions=ev_pos,
        coordinate=pc,
        distance=pd,
        gradients=-ev_g,
        n_calls=len(ev),
    )
    fmax_last = float(series["true_force"][-1])
    search = SearchHistory(
        series=series,
        force_tolerance=force_tolerance,
        converged=None if force_tolerance is None else fmax_last <= force_tolerance,
        events=events,
    )
    label = label or run.name
    cell = CellRecord(
        label=label,
        method="ml-neb",
        converged=search.converged,
        search_calls=len(ev),
        total_calls=len(ev),
        ledger={"endpoints": 2, "outer": len(ev) - 2, "total": len(ev)},
    )
    return SurrogateSearch(
        label=label,
        producer="ml-neb",
        band=BandHistory(
            final=final,
            snapshots=snapshots,
            points=points,
            events=events,
            numbers=np.asarray(ev[0].numbers, dtype=int),
            numbers_source="evaluated.traj",
        ),
        search=search,
        cell=cell,
        provenance={
            f: sha256_file(run / f)
            for f in ("predicted.traj", "evaluated.traj", "ml_summary.txt")
        },
    )
