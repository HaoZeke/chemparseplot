# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Bond length at each step of an ORCA relaxed surface scan.

```{versionadded} 1.14.0
```

ORCA opens every step of a ``%geom Scan`` over a bond with a banner::

    *               RELAXED SURFACE SCAN STEP   1               *
    *                                                           *
    *                 Bond (  1,   0)  :   7.55890395           *

The number after the colon is the bond length the step is held at, in
angstrom.
"""

import re

import numpy as np

from chemparseplot.units import Q_

STEP_BANNER = re.compile(
    r"RELAXED SURFACE SCAN STEP\s+(?P<step>\d+)"
    r"(?:[^\n]*\n){1,3}?"
    r"\s*\*\s*Bond \(\s*(?P<i>\d+),\s*(?P<j>\d+)\)\s*:\s*(?P<value>-?\d+\.\d+)"
)


def extract_scan_steps(text: str) -> tuple[tuple[int, int], Q_]:
    """Scanned atom pair and the bond length of every step of an ORCA bond scan.

    ```{versionadded} 1.14.0
    ```

    Parameters
    ----------
    text : str
        The text of the ORCA output.

    Returns
    -------
    tuple[tuple[int, int], Q_]
        The atom pair, numbered from 0 as in the ORCA input, and the bond
        length of each step in angstrom, in step order.

    Raises
    ------
    ValueError
        When the output has no bond scan banner, when the banners name more
        than one atom pair, or when they do not number the steps 1, 2, 3, ...
    """
    found = list(STEP_BANNER.finditer(text))
    if not found:
        msg = "no 'RELAXED SURFACE SCAN STEP' banner with a Bond line"
        raise ValueError(msg)
    pairs = {(int(m["i"]), int(m["j"])) for m in found}
    if len(pairs) != 1:
        msg = f"the banners scan more than one bond: {sorted(pairs)}"
        raise ValueError(msg)
    steps = [int(m["step"]) for m in found]
    if steps != list(range(1, len(steps) + 1)):
        msg = f"the steps are not numbered 1 to {len(steps)}: {steps}"
        raise ValueError(msg)
    lengths = np.array([float(m["value"]) for m in found])
    return pairs.pop(), Q_(lengths, "angstrom")
