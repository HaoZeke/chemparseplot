# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

# Unit support slows the entire code down by around a second
# import pint

# ureg = pint.UnitRegistry()
# ureg.define("kcal_mol = kcal / 6.02214076e+23 = kcm")
# Q_ = ureg.Quantity

# # Silence NEP 18 warning
# import warnings

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     Q_([])

# Conversion factors
HARTREE_TO_EV = 27.211407953
BOHR_TO_ANGSTROM = 0.529177249
