# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

from chemparseplot.plot.theme import (
    RUHI_COLORS,
    RUHI_THEME,
    PlotTheme,
    get_theme,
    setup_global_theme,
    setup_publication_theme,
)

# Lazy imports for submodules with heavy deps (cmcrameri, pint, etc.)
def __getattr__(name):
    if name == "geomscan":
        from chemparseplot.plot import geomscan as _mod
        return _mod
    if name == "structs":
        from chemparseplot.plot import structs as _mod
        return _mod
    if name == "chemgp":
        from chemparseplot.plot import chemgp as _mod
        return _mod
    if name == "ureg":
        from chemparseplot.units import ureg as _ureg
        _ureg.setup_matplotlib(True)
        _ureg.mpl_formatter = "{:~P}"
        return _ureg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
