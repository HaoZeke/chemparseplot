# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
import pint

ureg = pint.UnitRegistry()
ureg.define("kcal_mol = kcal / 6.02214076e+23 = kcm")
Q_ = ureg.Quantity

# Silence NEP 18 warning
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Q_([])