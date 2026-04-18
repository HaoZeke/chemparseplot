import datetime
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import ase
import numpy as np
from ase.io import read

try:
    from rgpycrumbs.basetypes import SaddleMeasure, SpinID
    from rgpycrumbs.time.helpers import one_day_tdelta
except ImportError:
    SaddleMeasure = SpinID = None
    one_day_tdelta = None

log = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class SellaLog:
    """Typed record for a single Sella geometry-step log entry.

```{versionadded} 0.0.3
```
"""

    step_id: int
    time_float: float
    energy: float
    fmax: float
    cmax: float
    rtrust: float
    rho: float
    trj_id: int


def _sella_loglist(log_f: str | Path) -> list[SellaLog]:
    _txt = np.loadtxt(log_f, skiprows=1, dtype=str)
    records = []
    for tline in np.atleast_2d(_txt):
        records.append(
            SellaLog(
                step_id=int(tline[1]),
                time_float=time.mktime(
                    datetime.datetime.strptime(tline[2], "%H:%M:%S")
                    .replace(tzinfo=datetime.UTC)
                    .timetuple()
                ),
                energy=float(tline[3]),
                fmax=float(tline[4]),
                cmax=float(tline[5]),
                rtrust=float(tline[6]),
                rho=float(tline[7]),
                trj_id=int(tline[8]),
            )
        )
    return records


@dataclass(frozen=True, slots=True)
class LogStart:
    """Start-of-log record for a Sella calculation.

    ```{versionadded} 0.0.3
    ```
    """

    tstart: str
    init_energy: float


@dataclass(frozen=True, slots=True)
class LogEnd:
    """End-of-log record for a Sella calculation.

    ```{versionadded} 0.0.3
    ```
    """

    iter_steps: int
    tend: str
    saddle_energy: float
    saddle_fmax: float


# def parse_sella_saddle(eresp: Path, rloc: SpinID):
#     respth = eresp / "npes.txt"
#     if respth.exists():
#         rdat = respth.read_text()
#     else:
#         return SaddleMeasure()  # default is a failure state
#     npes = int(respth.read_text().split()[-1])
#     logdat = (
#         [f for f in eresp.iterdir() if "log" in str(f)][0]
#         .read_text()
#         .strip()
#         .split("\n")
#     )
#     _, _, tstart, init_energy, _, _, _, _ = logdat[1].split()
#     _, iter_steps, tend, saddle_energy, saddle_fmax, _, _, _ = logdat[-1].split()

#     return SaddleMeasure(
#         pes_calls=npes,
#         iter_steps=int(iter_steps),
#         tot_time=one_day_tdelta(tend, tstart),
#         saddle_energy=float(saddle_energy),
#         saddle_fmax=float(saddle_fmax),
#         init_energy=float(init_energy),
#         barrier=float(saddle_energy) - float(init_energy),
#         success=True,
#         method="Sella",
#         mol_id=rloc.mol_id,
#         spin=rloc.spin,
#     )


def parse_log_line(line: str, line_type: str) -> LogStart | LogEnd | None:
    """Parses a log line based on its type.

    ```{versionadded} 0.0.3
    ```

    Args:
        line: The log line string.
        line_type: Either "start" or "end" indicating the line type.

    Returns:
        A LogStart or LogEnd instance, or None if parsing fails.
    """
    try:
        parts = line.split()
        if line_type == "start":
            return LogStart(tstart=parts[2], init_energy=float(parts[3]))
        elif line_type == "end":
            return LogEnd(
                iter_steps=int(parts[1]),
                tend=parts[2],
                saddle_energy=float(parts[3]),
                saddle_fmax=float(parts[4]),
            )
        else:
            return None
    except (IndexError, ValueError):
        log.warning("Could not parse %s log line: %s", line_type, line)
        return None


def _default_saddle_measure() -> SaddleMeasure:
    if SaddleMeasure is None:
        msg = "rgpycrumbs is required to construct Sella saddle measurements"
        raise ImportError(msg)
    return SaddleMeasure()


def parse_sella_saddle(eresp: Path, rloc: SpinID) -> SaddleMeasure:
    """Parses Sella saddle point calculation results.

    ```{versionadded} 0.0.3
    ```

    Args:
        eresp: Path to the directory containing the results.
        rloc: SpinID object containing molecule ID and spin.

    Returns:
        A SaddleMeasure object containing the parsed results.
    """
    traj_files = list(eresp.glob("*.traj"))

    if traj_files:
        traj_file = traj_files[0]  # Assume only one .traj file
        try:
            # Use ase.io.read with index=":" to read all images
            traj = read(traj_file, index=":")
            npes = len(traj)
        except Exception as e:
            log.warning("Could not read or process trajectory file %s: %s", traj_file, e)
            return _default_saddle_measure()
    else:
        log.warning("No .traj file found in %s. Using npes.txt or default.", eresp)
        respth = eresp / "npes.txt"
        if respth.exists():
            try:
                npes = int(respth.read_text().split()[-1])
            except (IndexError, ValueError):
                log.warning("Could not parse npes from %s", respth)
                return _default_saddle_measure()
        else:
            return _default_saddle_measure()

    log_files = [f for f in eresp.iterdir() if "log" in str(f)]
    if not log_files:
        log.warning("No log file found in %s", eresp)
        return _default_saddle_measure()

    try:
        logdat = log_files[0].read_text().strip().split("\n")
        start_data = parse_log_line(logdat[1], "start")
        end_data = parse_log_line(logdat[-1], "end")

        if start_data is None or end_data is None:
            return _default_saddle_measure()

        tot_time = one_day_tdelta(end_data.tend, start_data.tstart)

        return SaddleMeasure(
            pes_calls=npes,
            iter_steps=end_data.iter_steps,
            tot_time=tot_time,
            saddle_energy=end_data.saddle_energy,
            saddle_fmax=end_data.saddle_fmax,
            init_energy=start_data.init_energy,
            barrier=end_data.saddle_energy - start_data.init_energy,
            success=True,
            method="Sella",
            mol_id=rloc.mol_id,
            spin=rloc.spin,
        )

    except (IndexError, ValueError) as e:
        log.warning("Error parsing log data: %s", e)
        return _default_saddle_measure()


def _no_ghost(atm: ase.Atoms):
    # Sella writes out X symbol'd atoms for ghost atoms
    del atm[[atom.index for atom in atm if atom.symbol == "X"]]
    return atm


def _get_ghosts(traj_f):
    traj = ase.io.read(traj_f, ":")
    n_ghosts = 0
    for atm in traj:
        n_ghosts += Counter(atm.symbols).get("X", 0)
    return n_ghosts


def get_unique_frames(traj, sloglist: list[SellaLog], nround: int = 2):
    """Filter trajectory to unique frames by matching rounded fmax values.

    ```{versionadded} 0.0.3
    ```
    """
    # XXX: This is fairly idiotic, but it turns out empirically filtering by
    # rounding the fmax to 2 seems to give the same number as the number of
    # steps, so those frames are the "unique" ones.
    #
    #
    # However, this is ONLY for S000 so one needs to play with the nround for
    # other trajectories...
    #
    # I guess it could still be used if the trajectories were generated without
    # the newer patch
    unique_frames = []
    used_indices = set()

    for sella_entry in sloglist:
        target_fmax = round(sella_entry.fmax, nround)
        for i, atoms in enumerate(traj):
            if i not in used_indices:
                current_fmax = round(
                    np.max(np.linalg.norm(atoms.get_forces(), axis=1)), nround
                )
                if current_fmax == target_fmax:  # Exact comparison after rounding
                    unique_frames.append((i, atoms))
                    used_indices.add(i)
                    break
    return unique_frames


def make_geom_traj(traj_f, log_f, out_f="geom_step.traj"):
    """Write a geometry-step trajectory from a Sella trajectory and log file.

    ```{versionadded} 0.0.3
    ```
    """
    # XXX: This uses a logfile which has the patch for writing the trajectory ID
    traj = ase.io.read(traj_f, ":")
    sellalog = _sella_loglist(log_f)

    # Create a new trajectory with the unique frames
    unique_traj = [traj[x.trj_id] for x in sellalog]
    ase.io.write(out_f, unique_traj, format="traj")
