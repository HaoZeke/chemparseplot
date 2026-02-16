import configparser
import gzip
import os
import re
from pathlib import Path
from enum import Enum

import numpy as np
from rgpycrumbs.parsers.bless import BLESS_LOG
from rgpycrumbs.parsers.common import _NUM
from rgpycrumbs.search.helpers import tail

from rgpycrumbs.basetypes import DimerOpt, MolGeom, SaddleMeasure, SpinID


class EONSaddleStatus(Enum):
    # See SaddleSearchJob.cpp for the ordering
    # (numerical_status, descriptive_string)
    GOOD = 0, "Success"
    INIT = 1, "Initial"  # Should never show up, before run
    BAD_NO_CONVEX = 2, "Initial displacement unable to reach convex region"
    BAD_HIGH_ENERGY = 3, "Barrier too high"
    BAD_MAX_CONCAVE_ITERATIONS = 4, "Too many iterations in concave region"
    BAD_MAX_ITERATIONS = 5, "Too many iterations"
    BAD_NOT_CONNECTED = 6, "Saddle is not connected to initial state"
    BAD_PREFACTOR = 7, "Prefactors not within window"
    BAD_HIGH_BARRIER = 8, "Energy barrier not within window"
    BAD_MINIMA = 9, "Minimizations from saddle did not converge"
    FAILED_PREFACTOR = 10, "Hessian calculation failed"
    POTENTIAL_FAILED = 11, "Potential failed"
    NONNEGATIVE_ABORT = 12, "Nonnegative initial mode, aborting"
    NONLOCAL_ABORT = 13, "Nonlocal abort"
    NEGATIVE_BARRIER = 14, "Negative barrier detected"
    BAD_MD_TRAJECTORY_TOO_SHORT = 15, "No reaction found during MD trajectory"
    BAD_NO_NEGATIVE_MODE_AT_SADDLE = (
        16,
        "Converged to stationary point with zero negative modes",
    )
    BAD_NO_BARRIER = 17, "No forward barrier was found along minimized band"
    ZEROMODE_ABORT = 18, "Zero mode abort."
    OPTIMIZER_ERROR = 19, "Optimizer error."
    UNKNOWN = -1, "Unknown status"

    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj

    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return cls.UNKNOWN


def extract_saddle_gprd(log: list[str]):
    logdata = [BLESS_LOG.search(x).group("logdata").strip() for x in log]
    numdat = []
    for line in logdata:
        if line.isdigit() or line.startswith("-") and line[1:].isdigit():
            numdat.append(float(line))
        if _NUM.match(line):
            numdat.append(np.array(_NUM.findall(line), dtype=float))
    # TODO(rg): There's got to be a cleaner way..
    # assert len(numdat) == 3, f"numdat: {numdat} for log:\n{log}"
    return MolGeom(pos=numdat[0], energy=numdat[1], forces=numdat[2])


def _read_results_dat(eresp: Path) -> dict:
    """Reads and parses the results.dat file.

    Args:
        eresp: Path to the eOn results directory.

    Returns:
        A dictionary containing the parsed data from results.dat, or None if the file
        does not exist or the termination reason is not 0.
    """
    respth = eresp / "results.dat"
    if not respth.exists():
        return None

    rdat = respth.read_text()
    termination_int = int(re.search(r"(\d+) termination_reason", rdat).group(1))
    termination_reason = EONSaddleStatus.from_value(termination_int)
    term_dict = {
        "termination_status": str(termination_reason).split(".")[-1],
    }
    if termination_reason != EONSaddleStatus.GOOD:
        return term_dict

    results_data = {
        "pes_calls": int(re.search(r"(\d+) total_force_calls", rdat).group(1)),
        "iter_steps": int(re.search(r"(\d+) iterations", rdat).group(1)),
        "saddle_energy": float(
            re.search(r"(-?\d+\.\d+) potential_energy_saddle", rdat).group(1)
        ),
    }
    results_data |= term_dict
    return results_data


def _find_log_file(eresp: Path) -> Path | None:
    """Finds the most recent, valid log file within the eOn results directory.

    Args:
        eresp: Path to the eOn results directory.

    Returns:
        Path to the chosen log file, or None if no suitable log file is found.
    """
    log_files = list(eresp.glob("*.log.gz"))
    if not log_files:
        print(f"Warning: *.log.gz not found for {eresp}")
        return None

    # Sort log files by modification time (most recent first)
    log_files.sort(key=os.path.getmtime, reverse=True)

    for f in log_files:
        last_lines = tail(f, 5)  # Get the last 5 lines
        if not last_lines:  # if tail fails, move onto next file.
            continue
        fcheck = [x for x in last_lines if "Fail" in x]
        if not fcheck:  # If "Fail" is NOT in any of the last 5 lines
            return f

    print(f"No valid log files found in {eresp}")
    return None


def _extract_total_time(log_data: list[str]) -> float:
    """Extracts the total run time from the log data.

    Args:
        log_data: A list of strings representing the lines of the log file.

    Returns:
        The total run time in seconds as a float, or 0.0 if not found.
    """
    try:
        tot_time = float(
            re.search(
                r"INFO] real\s*(?P<realtime>\d*\.+\d*)",
                " ".join(log_data[-5:]),
            ).group("realtime")
        )
    except (AttributeError, TypeError):
        tot_time = 0.0
    return tot_time


def _extract_initial_energy(log_data: list[str]) -> float:
    """Extracts the initial energy from the log data.

    Args:
        log_data: A list of strings representing the lines of the log file.

    Returns:
        The initial energy as a float, or None if not found.
    """
    start_saddle_line = next(
        (line for line in log_data if "Saddle point search started from" in line), None
    )
    if start_saddle_line:
        init_energy = float(
            re.search(
                r"Saddle point search started from reactant with energy (-?\d+\.\d+) eV.",
                start_saddle_line,
            ).group(1)
        )
        return init_energy
    return None


def _extract_saddle_info(
    log_data: list[str], eresp: Path, *, is_gprd: bool
) -> tuple[float, str]:
    """Extracts saddle point information (saddle_fmax and method) from the log data.

    Args:
        log_data: A list of strings representing the lines of the log file.
        eresp: Path to the eOn results directory.
        is_gprd: Boolean flag indicating whether the GPRD method was used.

    Returns:
        A tuple containing:
        - saddle_fmax: The maximum force at the saddle point.
    """
    start_line = next(
        (i + 1 for i, line in enumerate(log_data) if "Elapsed time" in line), None
    )
    end_line = next(
        (
            i - 1
            for i, line in enumerate(log_data)
            if "timing information" in line.strip()
        ),
        None,
    )

    if is_gprd and start_line is not None and end_line is not None:
        saddle = extract_saddle_gprd(log_data[start_line:end_line])
        saddle_fmax = np.abs(np.max(saddle.forces))
    elif not is_gprd:
        try:
            # Expected header: Step, Step Size, Delta E, ||Force||
            # ||Force|| is the 5th element (index 4)
            saddle_fmax = float(
                (eresp / "client_spdlog.log")
                .read_text()
                .strip()
                .split("\n")[-5:][0]
                .split()[4]
            )
        except (FileNotFoundError, IndexError):
            saddle_fmax = 0.0
    else:
        saddle_fmax = 0.0

    return saddle_fmax


def _get_methods(eresp: Path) -> DimerOpt:
    _conf = configparser.ConfigParser()
    _conf.read(eresp / "config.ini")
    if _conf["Saddle Search"]["min_mode_method"] == "gprdimer":
        return DimerOpt(
            saddle="GPRD",
            rot=_conf["GPR Dimer"]["rotation_opt_method"],
            trans=_conf["GPR Dimer"]["translation_opt_method"],
        )
    elif _conf["Saddle Search"]["min_mode_method"] == "dimer":
        return DimerOpt(
            saddle="IDimer",
            # XXX: This is basically the default
            rot=_conf["Dimer"].get("opt_method", "cg"),
            trans=_conf["Optimizer"]["opt_method"],
        )
    else:
        errmsg = "Clearly wrong.."
        raise ValueError(errmsg)


def parse_eon_saddle(eresp: Path, rloc: "SpinID") -> "SaddleMeasure":
    """Parses eOn saddle point search results from a directory.

    Args:
        eresp: Path to the directory containing eOn results.
        rloc: A SpinID object.

    Returns:
        A SaddleMeasure object.
    """
    meth = _get_methods(eresp)
    is_gprd = meth.saddle.lower() == "gprd"
    # 1. Read results.dat
    results_data = _read_results_dat(eresp)
    if results_data is None:
        return SaddleMeasure(
            mol_id=getattr(rloc, "mol_id", None),
            spin=getattr(rloc, "spin", None),
            success=False,
            method=meth.saddle,
            dimer_rot=meth.rot,
            dimer_trans=meth.trans,
        )

    if list(results_data.keys()) == ['termination_status']:
        return SaddleMeasure(
            mol_id=getattr(rloc, "mol_id", None),
            spin=getattr(rloc, "spin", None),
            success=False,
            method=meth.saddle,
            dimer_rot=meth.rot,
            dimer_trans=meth.trans,
            termination_status=results_data["termination_status"],
        )

    # 2. Find the log file
    log_file = _find_log_file(eresp)
    if log_file is None:
        return SaddleMeasure(
            pes_calls=results_data["pes_calls"],
            iter_steps=results_data["iter_steps"],
            saddle_energy=results_data["saddle_energy"],
            tot_time=0.0,
            saddle_fmax=0.0,
            success=False,
            method=meth.saddle,
            dimer_rot=meth.rot,
            dimer_trans=meth.trans,
            mol_id=getattr(rloc, "mol_id", None),
            spin=getattr(rloc, "spin", None),
            termination_status=results_data["termination_status"],
        )

    # 3. Parse the log file
    with gzip.open(log_file, "rt") as f:
        log_data = f.readlines()

    # 4. Extract data from the log file
    tot_time = _extract_total_time(log_data)
    init_energy = _extract_initial_energy(log_data)
    saddle_fmax = _extract_saddle_info(log_data, eresp, is_gprd=is_gprd)

    # 5. Construct and return SaddleMeasure
    if tot_time <= 0.0 and meth.saddle == "IDimer":
        return SaddleMeasure(
            mol_id=getattr(rloc, "mol_id", None),
            spin=getattr(rloc, "spin", None),
            success=False,
            method=meth.saddle,
            dimer_rot=meth.rot,
            dimer_trans=meth.trans,
            pes_calls=results_data["pes_calls"],
            iter_steps=results_data["iter_steps"],
            saddle_energy=results_data["saddle_energy"],
            termination_status=results_data["termination_status"],
        )

    return SaddleMeasure(
        pes_calls=results_data["pes_calls"],
        iter_steps=results_data["iter_steps"],
        tot_time=tot_time,
        saddle_energy=results_data["saddle_energy"],
        saddle_fmax=saddle_fmax,
        success=True,
        method=meth.saddle,
        dimer_rot=meth.rot,
        dimer_trans=meth.trans,
        init_energy=init_energy,
        barrier=(
            results_data["saddle_energy"] - init_energy
            if init_energy is not None
            else None
        ),
        mol_id=getattr(rloc, "mol_id", None),
        spin=getattr(rloc, "spin", None),
        termination_status=results_data["termination_status"],
    )
