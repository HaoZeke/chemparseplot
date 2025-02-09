from pathlib import Path
from dataclasses import dataclass
from chemparseplot.basetypes import SaddleMeasure, SpinID
from rgpycrumbs.time.helpers import one_day_tdelta
from typing import Optional
from ase.io import read, Trajectory

@dataclass
class LogStart:
    tstart: str
    init_energy: float


@dataclass
class LogEnd:
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


def parse_log_line(line: str, line_type: str) -> Optional[LogStart | LogEnd]:
    """Parses a log line based on its type.

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
        print(f"Warning: Could not parse {line_type} log line: {line}")
        return None


def parse_sella_saddle(eresp: Path, rloc: SpinID) -> SaddleMeasure:
    """Parses Sella saddle point calculation results.

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
            print(
                f"Warning: Could not read or process trajectory file {traj_file}: {e}"
            )
            return SaddleMeasure()
    else:
        print(f"Warning: No .traj file found in {eresp}. Using npes.txt or default.")
        respth = eresp / "npes.txt"
        if respth.exists():
            try:
                npes = int(respth.read_text().split()[-1])
            except (IndexError, ValueError):
                print(f"Warning: Could not parse npes from {respth}")
                return SaddleMeasure()
        else:
            return SaddleMeasure()

    log_files = [f for f in eresp.iterdir() if "log" in str(f)]
    if not log_files:
        print(f"Warning: No log file found in {eresp}")
        return SaddleMeasure()

    try:
        logdat = log_files[0].read_text().strip().split("\n")
        start_data = parse_log_line(logdat[1], "start")
        end_data = parse_log_line(logdat[-1], "end")

        if start_data is None or end_data is None:
            return SaddleMeasure()

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
        print(f"Warning: Error parsing log data: {e}")
        return SaddleMeasure()
