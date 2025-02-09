# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
from collections import namedtuple

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
    interpolation. The energy is calculated using the formula:
    p = a*pow(dx(ii), 3.0) + b*pow(dx(ii), 2.0) + c*dx(ii) + d,
    where a, b, c, and d are coefficients of the cubic polynomial.

Notes
-----
The `nebpath` namedtuple is used within the `nebiter` namedtuple to store
detailed path information for each NEB iteration.
"""
