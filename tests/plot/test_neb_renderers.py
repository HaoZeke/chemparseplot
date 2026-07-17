# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
import shutil
from unittest.mock import patch

import numpy as np
import pytest

from tests.conftest import skip_if_not_env

skip_if_not_env("neb")

import matplotlib.pyplot as plt
from ase import Atoms

from chemparseplot.plot.neb import (
    _check_xyzrender,
    _render_xyzrender,
    plot_structure_strip,
    render_structure_to_image,
)

pytestmark = pytest.mark.neb


@pytest.fixture()
def water():
    return Atoms(
        "H2O",
        positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]],
    )


class TestRenderStructureToImage:
    """ASE renderer contract."""

    def test_returns_rgba(self, water):
        img = render_structure_to_image(water, zoom=0.3, rotation="0x,90y,0z")
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 4
        assert img.dtype in (np.float32, np.float64)


class TestCheckXyzrender:
    """Package importability (ensure_import / site-packages), not PATH binary."""

    def test_missing_package_raises(self):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "xyzrender" or name.startswith("xyzrender."):
                raise ImportError("blocked")
            if name == "rgpycrumbs" or name.startswith("rgpycrumbs."):
                raise ImportError("blocked")
            return real_import(name, *a, **k)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            with pytest.raises(RuntimeError, match="xyzrender is required"):
                _check_xyzrender()

    def test_present_package_passes(self):
        pytest.importorskip("xyzrender")
        _check_xyzrender()


class TestRenderXyzrender:
    """xyzrender Python API renderer."""

    def test_produces_image(self, water):
        pytest.importorskip("xyzrender")
        img = _render_xyzrender(water, canvas_size=200)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] in (3, 4)


class TestPlotStructureStripRendererParam:
    """Renderer dispatch in plot_structure_strip."""

    def test_ase_renderer_called(self, water):
        _, ax = plt.subplots()
        atoms_list = [water, water]
        labels = ["A", "B"]

        with patch("chemparseplot.plot.neb.render_structure_to_image") as mock_render:
            mock_render.return_value = np.zeros((10, 10, 4), dtype=np.float32)
            plot_structure_strip(ax, atoms_list, labels, renderer="ase")
            assert mock_render.call_count == len(atoms_list)

        plt.close("all")
