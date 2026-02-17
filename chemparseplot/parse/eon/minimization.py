#!/usr/bin/env python3

import re
from pathlib import Path


def min_e_result(eresp: Path) -> dict:
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
    termination_reason = re.search(r"(\w+) termination_reason", rdat).group(1)
    if termination_reason != "GOOD":
        return None

    min_energy = float(re.search(r"(-?\d+\.\d+) potential_energy", rdat).group(1))
    return min_energy
