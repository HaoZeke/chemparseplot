# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Image comparisons of the NEB profile, energy landscape and convergence figures.

The figures are drawn from the Au/Al(100) EMT runs in data/au_al100.json
(written by data/au_al100.py) and compared with the PNG baselines in
data/figures/. Text is set in DejaVu Sans, which matplotlib ships, with
matplotlib's own test font settings, so the images do not depend on the
fonts installed. Glyph rasters depend on the FreeType build, so the
comparisons are skipped under a FreeType other than the one recorded with
the baselines.

To redraw the baselines after an intended change:

    CHEMPARSEPLOT_REGENERATE_BASELINES=1 pytest tests/plot/test_figure_images.py
"""

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.testing
import numpy as np
import pytest
from matplotlib import ft2font
from matplotlib.testing.compare import compare_images

pl = pytest.importorskip("polars")
pytest.importorskip("scipy")
pytest.importorskip("ase")

from chemparseplot.plot import journal as jr
from chemparseplot.plot.chemgp import plot_surface_contour
from chemparseplot.plot.neb import plot_neb_profile
from chemparseplot.plot.optimization import (
    plot_relative_error,
    plot_single_ended_convergence,
)

DATA = Path(__file__).parent / "data"
BASELINES = DATA / "figures"
VERSIONS = BASELINES / "versions.json"
REGENERATE = os.environ.get("CHEMPARSEPLOT_REGENERATE_BASELINES") == "1"
FACE = "DejaVu Sans"
# Root-mean-square difference allowed, in 8-bit code values: half of one
# quantisation step. Rounding changes of one step pass on up to a quarter
# of the values; one pixel flipped between black and white fails on the
# single-column baselines, and two fail on the double-column one.
RMS_TOL = 0.5

_recorded = json.loads(VERSIONS.read_text()) if VERSIONS.exists() else {}
pytestmark = pytest.mark.skipif(
    not REGENERATE and ft2font.__freetype_version__ != _recorded.get("freetype"),
    reason=(
        f"baselines were drawn with FreeType {_recorded.get('freetype')}, "
        f"this matplotlib uses {ft2font.__freetype_version__}"
    ),
)


@pytest.fixture(scope="module")
def run():
    return json.loads((DATA / "au_al100.json").read_text())


@pytest.fixture
def style(monkeypatch):
    monkeypatch.setattr(jr, "resolve_font", lambda candidates: FACE)
    with mpl.rc_context():
        mpl.rcdefaults()
        matplotlib.testing.set_font_settings_for_testing()
        yield jr.figure_style(jr.DEFAULT_JOURNAL)
    plt.close("all")


def _dpi(style):
    """Resolution at which the narrowest line is one pixel wide."""
    return jr.BP_PER_INCH / style.narrow


def _check(path, name):
    baseline = BASELINES / f"{name}.png"
    if REGENERATE:
        BASELINES.mkdir(exist_ok=True)
        shutil.copyfile(path, baseline)
        VERSIONS.write_text(
            json.dumps(
                {
                    "matplotlib": mpl.__version__,
                    "freetype": ft2font.__freetype_version__,
                },
                indent=1,
            )
            + "\n"
        )
        return
    result = compare_images(str(baseline), str(path), tol=RMS_TOL)
    assert result is None, (
        f"{result} (baselines drawn with matplotlib {_recorded.get('matplotlib')})"
    )


def _save(fig, tmp_path, name, style):
    return jr.save_figure(fig, tmp_path / f"{name}.png", dpi=_dpi(style))


def _band(row, e0):
    return np.array(row["s"]), np.array(row["energies"]) - e0, np.array(row["f_para"])


def _final_band(run, optimizer="BFGS"):
    return _band(run["neb"][optimizer]["history"][-1], run["e_initial"])


def _bridge_barrier(run):
    return run["saddle"]["energy"] - run["e_initial"]


def test_neb_profile(style, run, tmp_path):
    fig = plot_neb_profile(*_final_band(run))
    assert fig.get_figwidth() == jr.ACS.width("single")
    _check(_save(fig, tmp_path, "neb_profile", style), "neb_profile")


def test_neb_profile_history(style, run, tmp_path):
    rows = run["neb"]["BFGS"]["history"]
    fig = plot_neb_profile(
        *_final_band(run),
        history=[_band(r, run["e_initial"]) for r in rows[:-1]],
        reference=_bridge_barrier(run),
        reference_label="Bridge saddle",
    )
    _check(_save(fig, tmp_path, "neb_profile_history", style), "neb_profile_history")


def test_energy_landscape(style, run, tmp_path):
    grid = run["map"]
    x, y = np.meshgrid(grid["x"], grid["y"])
    z = np.array(grid["energy"])
    band = np.array(run["neb"]["BFGS"]["history"][-1]["au_xy"])
    start, end = run["au_initial_xyz"], run["au_final_xyz"]
    saddle = run["saddle"]["au_xyz"]
    fig = plot_surface_contour(
        x,
        y,
        z - z.min(),
        paths={"NEB": (band[:, 0], band[:, 1])},
        points={
            "minima": ([start[0], end[0]], [start[1], end[1]]),
            "saddles": ([saddle[0]], [saddle[1]]),
        },
        xlabel="$x$ (Å)",
        ylabel="$y$ (Å)",
        zlabel=r"$\Delta E$ (eV)",
    )
    assert fig.get_figwidth() == jr.ACS.width("single")
    _check(_save(fig, tmp_path, "energy_landscape", style), "energy_landscape")


def test_convergence(style, run, tmp_path):
    trajs, labels = [], []
    for name, neb in run["neb"].items():
        rows = neb["history"]
        frame = pl.DataFrame(
            {
                "iteration": [r["iteration"] for r in rows],
                "convergence": [r["fmax"] for r in rows],
                "step_size": [r["step"] for r in rows],
            }
        )
        trajs.append(SimpleNamespace(dat_df=frame))
        labels.append(name)
    out = tmp_path / "convergence.png"
    plot_single_ended_convergence(
        trajs,
        labels,
        out,
        _dpi(style),
        threshold=run["meta"]["fmax"],
        force_label=r"$F_{\max}$ (eV/Å)",
        step_label="Step (Å)",
    )
    _check(out, "convergence")


def test_relative_error(style, run, tmp_path):
    jr.apply_style()
    fig, ax = jr.journal_figure(style=style)
    reference = _bridge_barrier(run)
    for index, (name, neb) in enumerate(run["neb"].items()):
        rows = neb["history"]
        iterations = [r["iteration"] for r in rows]
        barriers = [max(r["energies"]) - run["e_initial"] for r in rows]
        plot_relative_error(ax, iterations, barriers, reference, label=name, index=index)
    jr.label_lines(ax)
    ax.set_ylabel("Barrier relative error")
    _check(_save(fig, tmp_path, "relative_error", style), "relative_error")
