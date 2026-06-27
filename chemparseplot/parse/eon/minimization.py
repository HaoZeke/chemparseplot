#!/usr/bin/env python3

import re
from pathlib import Path


def min_e_result(eresp: Path) -> float | None:
    """Reads and parses the results.dat file.

    ```{versionadded} 0.0.3
    ```

    Args:
        eresp: Path to the eOn results directory.

    Returns:
        The minimized energy from ``results.dat``, or ``None`` if the file does not
        exist or the run did not terminate successfully.
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
