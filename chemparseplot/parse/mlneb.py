"""Retained CatLearn ML-NEB progress tables and measured force histories."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

PROGRESS_FIELDS = 10
PRINTED_FORCE_TOLERANCE = 5.01e-7

VALUE_COLUMNS = (
    "barrier_eV",
    "reverse_barrier_eV",
    "max_uncertainty_eV",
    "mean_uncertainty_eV",
    "sample_fmax_eV_A",
)


def read_mlneb_history(path):
    """Read each step once, using its first printed timestamp.

    ML-NEB reprints its entire table with the current time at every step.
    The force belongs to the newly evaluated geometry; uncertainty belongs
    to the preceding surrogate path. Neither is a final saddle certificate.
    """
    steps = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip().startswith("| ML-NEB |"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != PROGRESS_FIELDS:
            msg = "Incomplete ML-NEB progress row"
            raise ValueError(msg)
        step = int(parts[2])
        # The source has local wall-clock fields; only elapsed durations are used.
        stamp = datetime.strptime(parts[3], "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007
        values = dict(zip(VALUE_COLUMNS, map(float, parts[4:9]), strict=True))
        if not all(math.isfinite(value) for value in values.values()):
            msg = "Nonfinite ML-NEB history value"
            raise ValueError(msg)
        if step in steps:
            if any(steps[step][key] != value for key, value in values.items()):
                msg = "Conflicting repeated ML-NEB step"
                raise ValueError(msg)
            continue
        steps[step] = {"step": step, "timestamp": stamp, **values}
    if not steps or list(steps) != list(range(len(steps))):
        msg = "ML-NEB history must contain consecutive steps from zero"
        raise ValueError(msg)
    start = steps[0]["timestamp"]
    result = []
    for row in steps.values():
        elapsed = (row["timestamp"] - start).total_seconds()
        if result and elapsed < result[-1]["elapsed_s"]:
            msg = "ML-NEB timestamps are not monotone"
            raise ValueError(msg)
        result.append(
            {
                "step": row["step"],
                "observations": row["step"] + 2,
                "elapsed_s": elapsed,
                **{key: row[key] for key in VALUE_COLUMNS},
            }
        )
    return result


def check_mlneb_forces(history, frames):
    """Match printed force norms to stored measured ASE trajectory frames.

    Step zero reports the second endpoint. Each following step adds one
    observation. The tolerance is half the last printed decimal place.
    """
    if len(frames) != history[-1]["observations"]:
        msg = "History and measured trajectory lengths differ"
        raise ValueError(msg)
    largest = 0.0
    for row in history:
        frame = frames[row["observations"] - 1]
        forces = frame.calc.results["forces"]
        maximum = max(math.sqrt(sum(float(x) ** 2 for x in atom)) for atom in forces)
        error = abs(maximum - row["sample_fmax_eV_A"])
        if error > PRINTED_FORCE_TOLERANCE:
            msg = "Printed force does not match the measured trajectory"
            raise ValueError(msg)
        largest = max(largest, error)
    return largest
