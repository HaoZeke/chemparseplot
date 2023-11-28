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
from pathlib import Path
from chemparseplot.units import Q_

def extract_energy_data(data, energy_type):
    """
    Extracts and converts the energy data for the specified energy type ('Actual' or 'SCF').
    Energies are converted from Hartree to eV, distances from Bohr to Angstrom.
    Assumes the input data is a blob of text.
    """
    # Regular expression to find the energy type and the two-column data following it
    pattern = rf".*? Calculated Surface.*?{energy_type}.*?.*?((?:\s+\d+\.\d+\s+-?\d+\.\d+)+)"
    matchr = re.search(pattern, data, re.MULTILINE)

    if not matchr:
        return np.array([]), np.array([])

    # Extract and convert the data
    energy_data = matchr.group(0)
    x_values, y_values = [], []
    for line in energy_data.split('\n')[1:]:  # Skip the first
        x, y = map(float, line.split())
        x_values.append(x)
        y_values.append(y)

    return Q_(np.as_array(x_values), 'bohr'), Q_(np.as_array(y_values), 'hartree')
