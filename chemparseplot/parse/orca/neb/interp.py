"""
For parsing .interp files from inputs like:
!B3LYP def2-SVP NEB-CI
%neb
nimages = 7
Product "prod.xyz"
end
*xyzfile 0 1 react.xyz
"""
import re
from collections import namedtuple

import chemparseplot.parse.converter as conv
import chemparseplot.parse.patterns as pat
from chemparseplot.units import Q_

# namedtuple for storing NEB iteration data
nebiter = namedtuple("nebiter", ["iteration", "nebpath"])
"""
A namedtuple representing an iteration of a Nudged Elastic Band (NEB) calculation.

Parameters
----------
iteration : int
    The iteration number of the NEB calculation.
nebpath : nebpath namedtuple
    The data for the NEB path at this iteration.

See Also
--------
nebpath : Stores the normalized arclength, actual arclength, and energy data for
    the NEB path.
"""

# namedtuple for storing the NEB path data
nebpath = namedtuple("nebpath", ["norm_dist", "arc_dist", "energy"])
"""
A namedtuple representing the NEB path data.

Parameters
----------
norm_dist : float
    Normalized Arclength (0 to 1), representing the progression along the reaction path.
    Calculated as xcoord2 = arcS[img] / arcS[nim-1].
arc_dist : float
    Actual Arclength at each point along the reaction path. Calculated as
    xcoord = arcS[img] + dx(ii).
energy : float
    Interpolated Energy at each point, calculated using cubic polynomial
    interpolation.  The energy is calculated using the formula:
    p = a*pow(dx(ii), 3.0) + b*pow(dx(ii), 2.0) + c*dx(ii) + d,
    where a, b, c, and d are coefficients of the cubic polynomial.

Notes
-----
The `nebpath` namedtuple is used within the `nebiter` namedtuple to store
detailed path information for each NEB iteration.
"""

# fmt: off
INTERP_PAT = (
    r"Iteration:\s*(?P<iteration>\d+)\s*\n"  # Capture iteration number
    r"Images: Distance\s+\(Bohr\), Energy \(Eh\)\s*\n"  # Match 'Images:' line
    + pat.THREE_COL_NUM
)
# fmt: on


def extract_interp_points(text: str) -> list[int, Q_, Q_]:
    data = []
    for match in re.finditer(INTERP_PAT, text, re.DOTALL):
        iteration = int(match.group("iteration"))
        energytxt = match.group("threecolnum")
        ixydat = conv.np_txt(energytxt)
        nxdu = Q_(ixydat[:, 0], "dimensionless")
        xdu = Q_(ixydat[:, 1], "bohr")
        ydu = Q_(ixydat[:, 2], "hartree")
        tnp = nebpath(norm_dist=nxdu, arc_dist=xdu, energy=ydu)
        data.append(nebiter(iteration=iteration, nebpath=tnp))
    return data
