#!/usr/bin/env python
"""Tutorial smoke tests for chemparseplot (pytest-collectable).

Also runnable as a script: python tests/tutorials/test_chemparseplot.py
"""

from __future__ import annotations

import gzip
import sys
import tempfile
from pathlib import Path

import pytest

from tests.conftest import skip_if_not_env

skip_if_not_env("neb")


def test_tutorial_imports():
    from chemparseplot.parse.eon.saddle_search import parse_eon_saddle
    from chemparseplot.parse.orca import geomscan
    from chemparseplot.plot.neb import plot_energy_path, plot_landscape_surface

    assert callable(parse_eon_saddle)
    assert geomscan is not None
    assert callable(plot_energy_path)
    assert callable(plot_landscape_surface)


def test_tutorial_orca_geomscan_parsing():
    from chemparseplot.parse.orca import geomscan

    sample_output = """
          *****************************************************
          *               ORCA 5.0.4                          *
          *               OPTIMIZATION RUN                    *
          *****************************************************

Geometry scan step 1
Energy:    -123.456789 Eh

Geometry scan step 2
Energy:    -123.450000 Eh

Geometry scan step 3
Energy:    -123.445000 Eh
"""
    energy_data = geomscan.extract_energy_data(sample_output, "Actual")
    assert len(energy_data) > 0


def test_tutorial_eon_saddle_parse():
    pytest.importorskip("rgpycrumbs")
    from rgpycrumbs.basetypes import SpinID

    from chemparseplot.parse.eon.saddle_search import parse_eon_saddle

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "results.dat").write_text(
            """
0 termination_reason
100 total_force_calls
50 iterations
-123.450000 potential_energy_saddle
"""
        )
        (tmpdir / "config.ini").write_text(
            """
[Saddle Search]
min_mode_method = dimer

[Dimer]
opt_method = cg

[Optimizer]
opt_method = lbfgs
"""
        )
        with gzip.open(tmpdir / "test.log.gz", "wb") as f:
            f.write(
                b"""
[INFO] Saddle point search started from reactant with energy -123.500000 eV.
[INFO] real    10.5
[INFO] Saddle found
"""
            )
        (tmpdir / "client_spdlog.log").write_text(
            """
Step  Step_Size  Delta_E   ||Force||
1     0.1        -0.01     0.05
"""
        )
        result = parse_eon_saddle(tmpdir, SpinID(mol_id="test", spin=0))
        assert result.success
        assert result.pes_calls == 100
        assert result.iter_steps == 50


def test_tutorial_neb_plot_energy_path():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from chemparseplot.plot.neb import plot_energy_path

    rc = np.array([0.0, 1.0, 2.0, 3.0])
    energy = np.array([-123.5, -123.4, -123.3, -123.5])
    f_para = np.array([0.1, 0.05, -0.05, -0.1])
    fig, ax = plt.subplots()
    plot_energy_path(ax, rc, energy, f_para, "blue", alpha=0.8, zorder=10)
    assert len(ax.lines) > 0
    plt.close(fig)


if __name__ == "__main__":
    # Preserve script entry for CI tutorial runners
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(f"Running {name}...")
                fn()
                print(f"  ok {name}")
            except Exception as exc:
                failed += 1
                print(f"  FAIL {name}: {exc}", file=sys.stderr)
    sys.exit(failed)
