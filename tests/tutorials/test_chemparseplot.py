#!/usr/bin/env python
"""
Test script for chemparseplot tutorials.
Verifies that tutorial code examples work correctly.

Run with: cd chemparseplot && uv run python tests/tutorials/test_chemparseplot.py
"""

import pytest

from tests.conftest import skip_if_not_env

skip_if_not_env("neb")

import sys
from pathlib import Path

# Test 1: Import tutorial modules
print("Test 1: Importing chemparseplot modules...")
try:
    from chemparseplot.parse.eon.saddle_search import parse_eon_saddle
    from chemparseplot.parse.orca import geomscan
    from chemparseplot.plot.neb import plot_energy_path, plot_landscape_surface

    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Test geomscan parsing (from existing tutorial)
print("\nTest 2: Testing ORCA geomscan parsing...")
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

try:
    energy_data = geomscan.extract_energy_data(sample_output, "Actual")
    assert len(energy_data) > 0, "Expected energy data"
    print(f"  Extracted {len(energy_data)} energy points")
    print("✓ ORCA geomscan parsing works")
except Exception as e:
    print(f"✗ Geomscan parsing failed: {e}")
    sys.exit(1)

# Test 3: Test eOn saddle search parsing
print("\nTest 3: Testing eOn saddle search parsing...")
import gzip

# Create a minimal mock eOn results directory
import tempfile

from rgpycrumbs.basetypes import SpinID

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    # Create minimal results.dat (termination_reason=0 means GOOD)
    results_dat = """
0 termination_reason
100 total_force_calls
50 iterations
-123.450000 potential_energy_saddle
"""
    (tmpdir / "results.dat").write_text(results_dat)

    # Create minimal config.ini
    config_ini = """
[Saddle Search]
min_mode_method = dimer

[Dimer]
opt_method = cg

[Optimizer]
opt_method = lbfgs
"""
    (tmpdir / "config.ini").write_text(config_ini)

    # Create minimal log file
    log_content = b"""
[INFO] Saddle point search started from reactant with energy -123.500000 eV.
[INFO] real    10.5
[INFO] Saddle found
"""
    with gzip.open(tmpdir / "test.log.gz", "wb") as f:
        f.write(log_content)

    # Create client_spdlog.log with force data
    client_log = """
Step  Step_Size  Delta_E   ||Force||
1     0.1        -0.01     0.05
"""
    (tmpdir / "client_spdlog.log").write_text(client_log)

    try:
        rloc = SpinID(mol_id="test", spin=0)
        result = parse_eon_saddle(tmpdir, rloc)

        assert result.success, "Expected successful parse"
        assert result.pes_calls == 100
        assert result.iter_steps == 50
        print(f"  Success: {result.success}")
        print(f"  PES calls: {result.pes_calls}")
        print(f"  Iterations: {result.iter_steps}")
        print(f"  Method: {result.method}")
        print("✓ eOn saddle search parsing works")
    except Exception as e:
        print(f"✗ eOn parsing failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

# Test 4: Test plotting functions (headless)
print("\nTest 4: Testing NEB plotting functions...")
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

try:
    # Create sample data
    rc = np.array([0.0, 1.0, 2.0, 3.0])
    energy = np.array([-123.5, -123.4, -123.3, -123.5])
    f_para = np.array([0.1, 0.05, -0.05, -0.1])

    fig, ax = plt.subplots()
    plot_energy_path(ax, rc, energy, f_para, "blue", alpha=0.8, zorder=10)

    # Verify plot was created
    assert len(ax.lines) > 0, "Expected plot lines"
    print(f"  Created plot with {len(ax.lines)} lines")
    print("✓ NEB plotting functions work")
    plt.close(fig)
except Exception as e:
    print(f"✗ Plotting failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All chemparseplot tutorial tests passed! ✓")
print("=" * 60)
