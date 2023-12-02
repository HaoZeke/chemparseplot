# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

from chemparseplot.plot import geomscan, structs
from chemparseplot.units import ureg

ureg.setup_matplotlib(True)
ureg.mpl_formatter = "{:~P}"
