import re
import gzip
from pathlib import Path
from enum import StrEnum, auto

from rgpycrumbs.parsers.bless import BLESS_LOG, BLESS_TIME
from rgpycrumbs.parsers.common import _NUM
from rgpycrumbs.search.helpers import tail, head_search

from chemparseplot.basetypes import MolGeom, SaddleMeasure, SpinID

def SaddleSearchMethod(StrEnum):
    UNKNOWN = auto()
    GPRD = auto()
    IDLBFGSX2 = "idimer_lbfgsrot_lbfgs"
    IDLBFGSCG = "idimer_cgrot_lbfgs"


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
        eresp: Path to the EON results directory.

    Returns:
        A dictionary containing the parsed data from results.dat, or None if the file
        does not exist or the termination reason is not 0.
    """
    respth = eresp / "results.dat"
    if not respth.exists():
        return None

    rdat = respth.read_text()
    termination_reason = int(re.search(r"(\d+) termination_reason", rdat).group(1))
    if termination_reason != 0:
        return None

    results_data = {
        "pes_calls": int(re.search(r"(\d+) total_force_calls", rdat).group(1)),
        "iter_steps": int(re.search(r"(\d+) iterations", rdat).group(1)),
        "saddle_energy": float(
            re.search(r"(-?\d+\.\d+) potential_energy_saddle", rdat).group(1)
        ),
    }
    return results_data


def _find_log_file(eresp: Path) -> Path:
    """Finds the correct log file within the EON results directory.

    Args:
        eresp: Path to the EON results directory.

    Returns:
        Path to the chosen log file, or None if no suitable log file is found.
    """
    log_files = list(eresp.glob("*.log.gz"))
    if not log_files:
        print(f"Warning: *.log.gz not found for {eresp}")
        return None

    if len(log_files) != 1:
        for f in log_files:
            try:
                last_line = tail(f, 1)[0]
                if "pixi run eonclient" in last_line and head_search(
                    f, "scf_thresh: 1e-8", 60
                ):
                    print(f"Found the desired log file: {f}")
                    return f
            except IndexError:
                pass  # Ignore empty or invalid log files

        print(f"No valid log files found")
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
    log_data: list[str], eresp: Path, is_gprd: bool
) -> tuple[float, str]:
    """Extracts saddle point information (saddle_fmax and method) from the log data.

    Args:
        log_data: A list of strings representing the lines of the log file.
        eresp: Path to the EON results directory.
        is_gprd: Boolean flag indicating whether the GPRD method was used.

    Returns:
        A tuple containing:
        - saddle_fmax: The maximum force at the saddle point.
        - method: A string indicating the method used ("GPRD", "IDimer", or "Unknown").
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
        method = "GPRD"
    elif not is_gprd:
        try:
            saddle_fmax = float(
                (eresp / "client_spdlog.log")
                .read_text()
                .strip()
                .split("\n")[-5:][0]
                .split()[5]
            )
            method = "IDimer"
        except (FileNotFoundError, IndexError):
            saddle_fmax = 0.0
            method = (
                "Unknown"  # Or handle the error in a way that suits your application
            )
    else:
        saddle_fmax = 0.0
        method = "Unknown"

    return saddle_fmax, method


def parse_eon_saddle(
    eresp: Path, rloc: "SpinID", meth: SaddleSearchMethod
) -> "SaddleMeasure":
    """Parses EON saddle point search results from a directory.

    Args:
        eresp: Path to the directory containing EON results.
        rloc: A SpinID object.
        is_gprd: Boolean flag for GPRD method.

    Returns:
        A SaddleMeasure object.
    """

    # 1. Read results.dat
    results_data = _read_results_dat(eresp)
    if results_data is None:
        return SaddleMeasure(
            mol_id=getattr(rloc, "mol_id", None), spin=getattr(rloc, "spin", None)
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
            mol_id=getattr(rloc, "mol_id", None),
            spin=getattr(rloc, "spin", None),
        )

    # 3. Parse the log file
    with gzip.open(log_file, "rt") as f:
        log_data = f.readlines()

    # 4. Extract data from the log file
    tot_time = _extract_total_time(log_data)
    init_energy = _extract_initial_energy(log_data)
    saddle_fmax, method = _extract_saddle_info(log_data, eresp, is_gprd)

    # 5. Construct and return SaddleMeasure
    if tot_time <= 0.0 and method == "IDimer":
        return SaddleMeasure(
            mol_id=getattr(rloc, "mol_id", None), spin=getattr(rloc, "spin", None)
        )

    return SaddleMeasure(
        pes_calls=results_data["pes_calls"],
        iter_steps=results_data["iter_steps"],
        tot_time=tot_time,
        saddle_energy=results_data["saddle_energy"],
        saddle_fmax=saddle_fmax,
        success=True,
        method=method,
        init_energy=init_energy,
        barrier=(
            results_data["saddle_energy"] - init_energy
            if init_energy is not None
            else None
        ),
        mol_id=getattr(rloc, "mol_id", None),
        spin=getattr(rloc, "spin", None),
    )
