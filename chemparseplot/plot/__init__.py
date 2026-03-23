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
    import importlib

    _LAZY_SUBMODULES = {"geomscan", "structs", "chemgp", "optimization"}
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name == "ureg":
        from chemparseplot.units import ureg as _ureg

        _ureg.setup_matplotlib(True)
        _ureg.mpl_formatter = "{:~P}"
        return _ureg
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
