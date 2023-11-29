"""
For parsing outputs from input files like this:
!OPT UHF def2-SVP
%geom Scan
  B 0 1 = 7.5589039543, 0.2116708996, 33
 end
end
*xyzfile 0 1 h2_base.xyz
"""
import re

import numpy as np

import chemparseplot.parse.patterns as pat
from chemparseplot.units import Q_


def extract_energy_data(data: str, energy_type: str) -> tuple[Q_, Q_]:
    """
    Extracts and converts the energy data for a specified energy type.

    This function assumes the input data is a blob of text. It searches for
    'Calculated Surface' followed by the specified energy type ('Actual' or 'SCF')
    and extracts the two-column data (distance and energy values) following it.
    Energies are returned in Hartree and distances in Bohr, as these are the default
    units used in ORCA.

    Parameters
    ----------
    data : str
        The blob of text containing energy data.
    energy_type : str
        The type of energy to search for ('Actual' or 'SCF').

    Returns
    -------
    tuple[Q_, Q_]
        A tuple containing two `Quantity` objects from the `pint` library.
        The first element is an array of distances in Bohr, and the second
        element is an array of energies in Hartree.

    """
    # Regular expression to find the energy type and the two-column data following it
    # https://regex101.com/r/RF6b4V/2
    # fmt: off
    pattern = (
        r".*? Calculated Surface.*?"
        rf"{energy_type}.*?"
    ) + pat.TWO_COL_NUM
    matchr = re.search(pattern, data, re.MULTILINE)
    # fmt: on

    if not matchr:
        return Q_(np.array([]), "bohr"), Q_(np.array([]), "hartree")

    # Extract and convert the data
    energy_data = matchr.group("twocolnum")
    x_values, y_values = [], []
    for line in energy_data.split("\n"):
        x, y = map(float, line.split())
        x_values.append(x)
        y_values.append(y)

    return Q_(np.array(x_values), "bohr"), Q_(np.array(y_values), "hartree")
