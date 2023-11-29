# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
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

import chemparseplot.parse.converter as conv
import chemparseplot.parse.patterns as pat
from chemparseplot.basetypes import nebiter, nebpath
from chemparseplot.units import Q_

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
