# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

from chemparseplot.plot import geomscan, structs, theme
from chemparseplot.plot.theme import (
    RUHI_COLORS,
    RUHI_THEME,
    PlotTheme,
    get_theme,
    setup_global_theme,
    setup_publication_theme,
)
from chemparseplot.units import ureg

ureg.setup_matplotlib(True)
ureg.mpl_formatter = "{:~P}"
