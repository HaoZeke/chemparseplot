# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Regression checks for the Ruhi theme colour maps."""

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
from matplotlib.colors import to_hex

from chemparseplot.plot.palette import RUHI_SEQUENTIAL
from chemparseplot.plot.theme import RUHI_THEME

# The linear blend registered as ruhi_diverging, teal through sunshine.
_DIVERGING_START = "#004D40"
_DIVERGING_END = "#F1DB4B"


def test_ruhi_theme_uses_the_diverging_blend():
    assert RUHI_THEME.cmap_profile == RUHI_THEME.cmap_landscape == "ruhi_diverging"
    cmap = mpl.colormaps[RUHI_THEME.cmap_profile]
    assert to_hex(cmap(0.0)).upper() == _DIVERGING_START
    assert to_hex(cmap(1.0)).upper() == _DIVERGING_END
    assert RUHI_SEQUENTIAL in mpl.colormaps
