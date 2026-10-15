# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Public CON trajectory load and plot helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

readcon = pytest.importorskip("readcon")
pytest.importorskip("ase")
pytest.importorskip("matplotlib")
pytest.importorskip("polars")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from chemparseplot.api import load_trajectory, plot_con_overlay, plot_con_overview
from chemparseplot.parse.eon.con_io import write_con_frames
from tests.parse._con_fixtures import write_band


def _with_forces(path: Path) -> None:
    frames = list(readcon.read_con(str(path)))
    for frame in frames:
        for atom in frame.atoms:
            atom.fx = 0.0
            atom.fy = 0.0
            atom.fz = 0.4
    write_con_frames(path, frames)


def test_load_trajectory_con_fmax_from_frc(tmp_path: Path):
    src = tmp_path / "movie.con"
    write_band(src, [(1.5, 0.0), (2.5, 1.0)])
    frames = readcon.read_con(str(src))
    for frame in frames:
        for atom in frame.atoms:
            atom.fx = 0.3
            atom.fy = 0.0
            atom.fz = -0.4
    write_con_frames(src, frames)

    loaded = load_trajectory(src)
    assert loaded.source == "con"
    assert loaded.n_frames == 2
    assert loaded.energies == pytest.approx([1.5, 2.5])
    assert loaded.table["fmax"].to_list() == pytest.approx([0.5, 0.5])
    assert loaded.frames[0].frc is not None


def test_load_trajectory_xyz_uses_ase(tmp_path: Path):
    xyz = tmp_path / "mol.xyz"
    xyz.write_text("1\nenergy=-1.25\nH 0 0 0\n")
    loaded = load_trajectory(xyz)
    assert loaded.source == "xyz"
    assert loaded.n_frames == 1
    assert loaded.table["fmax"].null_count() == 1


def test_plot_con_overview_and_overlay(tmp_path: Path):
    left = tmp_path / "a.con"
    right = tmp_path / "b.con"
    write_band(left, [(0.0, 0.0), (1.0, 0.2)])
    write_band(right, [(5.0, 0.0), (5.5, 0.2), (6.0, 0.4)])
    _with_forces(left)
    trajs = [load_trajectory(left), load_trajectory(right)]
    fig = plot_con_overview(
        trajs[0],
        relative=True,
        show_forces=True,
        structures="endpoints",
        strip_renderer="ase",
        title="one",
    )
    assert len(fig.axes) >= 2
    fig.savefig(tmp_path / "overview.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    returned = plot_con_overlay(
        trajs,
        labels=["a", "b"],
        ax=ax,
        energy_unit="kcal/mol",
        relative=True,
        title="two",
    )
    assert returned is fig
    assert ax.get_title() == "two"
    assert len(ax.lines) == 2
    plt.close(fig)


def test_plot_con_overlay_label_count():
    traj = type(
        "T",
        (),
        {
            "energies": [1.0, np.nan],
            "n_frames": 2,
            "table": None,
            "frames": [],
            "source": "con",
        },
    )()
    with pytest.raises(ValueError, match="labels"):
        plot_con_overlay([traj, traj], labels=["only-one"])
