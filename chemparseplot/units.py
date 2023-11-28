# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
import warnings

import pint

ureg = pint.UnitRegistry(cache_folder=":auto:")
ureg.define("kcal_mol = kcal / 6.02214076e+23 = kcm")
Q_ = ureg.Quantity

# Silence NEP 18 warning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Q_([])
