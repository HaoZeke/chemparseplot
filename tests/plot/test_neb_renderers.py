# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
import shutil
from unittest.mock import patch

import numpy as np
import pytest

from tests.conftest import skip_if_not_env

skip_if_not_env("neb")

from ase import Atoms  # noqa: E402

from chemparseplot.plot.neb import (  # noqa: E402
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
        assert img.dtype == np.float32 or img.dtype == np.float64


class TestCheckXyzrender:
    """Binary availability check."""

    def test_missing_binary_raises(self):
        with patch.object(shutil, "which", return_value=None):
            with pytest.raises(RuntimeError, match="xyzrender binary not found"):
                _check_xyzrender()

    def test_present_binary_passes(self):
        with patch.object(shutil, "which", return_value="/usr/bin/xyzrender"):
            _check_xyzrender()


class TestRenderXyzrender:
    """xyzrender subprocess renderer."""

    @pytest.mark.skipif(
        shutil.which("xyzrender") is None,
        reason="xyzrender not installed",
    )
    def test_produces_image(self, water):
        img = _render_xyzrender(water, canvas_size=200)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 4


class TestPlotStructureStripRendererParam:
    """Renderer dispatch in plot_structure_strip."""

    def test_ase_renderer_called(self, water):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        atoms_list = [water, water]
        labels = ["A", "B"]

        with patch(
            "chemparseplot.plot.neb.render_structure_to_image"
        ) as mock_render:
            mock_render.return_value = np.zeros((10, 10, 4), dtype=np.float32)
            plot_structure_strip(
                ax, atoms_list, labels, renderer="ase"
            )
            assert mock_render.call_count == len(atoms_list)

        plt.close("all")
