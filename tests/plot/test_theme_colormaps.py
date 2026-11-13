"""Registered colormaps and their anchors."""

import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import to_rgb

from chemparseplot.plot.theme import RUHI_COLORS, build_cmap


def test_bond_order_map_separates_contacts_from_bonds():
    cmap = colormaps["ruhi_bond_order"]
    assert np.allclose(cmap(0.0)[:3], to_rgb(RUHI_COLORS["coral"]), atol=0.01)
    assert np.allclose(cmap(0.25)[:3], to_rgb(RUHI_COLORS["sunshine"]), atol=0.02)
    assert np.allclose(cmap(1.0)[:3], to_rgb(RUHI_COLORS["teal"]), atol=0.01)
    # single to triple bonds darken monotonically
    lightness = [sum(cmap(order / 3.0)[:3]) for order in (1.0, 1.5, 2.0, 3.0)]
    assert lightness == sorted(lightness, reverse=True)


def test_build_cmap_places_colours_at_positions():
    cmap = build_cmap(
        ["#000000", "#ff0000", "#ffffff"], "test_positioned", positions=[0.0, 0.2, 1.0]
    )
    assert np.allclose(cmap(0.2)[:3], (1.0, 0.0, 0.0), atol=0.02)
    even = build_cmap(["#000000", "#ff0000", "#ffffff"], "test_even")
    assert np.allclose(even(0.5)[:3], (1.0, 0.0, 0.0), atol=0.02)
    with pytest.raises(ValueError):
        build_cmap(["#000000", "#ffffff"], "test_short", positions=[0.0])
