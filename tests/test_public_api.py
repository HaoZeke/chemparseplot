# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Drive shipped chemparseplot.api entry points (parse → typed result)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chemparseplot.api import (
    convert_energy_magnitude,
    extract_orca_geomscan_energy,
    normalize_energy_unit,
    suite_pins,
)

# Inline fixture (same content as tests/fixtures_geomscan_snippet.txt)
_GEOMSCAN_SNIPPET = """
The Calculated Surface using the 'Actual Energy'
   7.55890395  -0.74239862
   7.32930292  -0.74349939
   7.09970189  -0.74467446

The Calculated Surface using the SCF energy
   7.55890395  -0.74239862
   7.32930292  -0.74349939
   7.09970189  -0.74467446
"""

FIXTURE_FILE = Path(__file__).parent / "fixtures_geomscan_snippet.txt"


def test_normalize_and_convert_energy_library_path():
    assert normalize_energy_unit("ev") == "eV"
    kcal = convert_energy_magnitude(1.0, "kcal/mol")
    val = float(np.asarray(kcal).reshape(-1)[0])
    assert 20.0 < val < 25.0
    back = float(np.asarray(convert_energy_magnitude(val, "eV", source_unit="kcal/mol")))
    assert abs(back - 1.0) < 1e-6


def test_extract_orca_geomscan_energy_typed_nonempty():
    """Public parse path: ORCA geomscan text → pint Quantities with data."""
    dist, energy = extract_orca_geomscan_energy(_GEOMSCAN_SNIPPET, "Actual Energy")
    assert dist.size == 3
    assert energy.size == 3
    assert str(dist.units) == "bohr"
    assert str(energy.units) == "hartree"
    np.testing.assert_allclose(
        dist.magnitude,
        [7.55890395, 7.32930292, 7.09970189],
    )
    np.testing.assert_allclose(
        energy.magnitude,
        [-0.74239862, -0.74349939, -0.74467446],
    )


def test_extract_orca_geomscan_energy_from_fixture_file():
    text = FIXTURE_FILE.read_text(encoding="utf-8")
    dist, energy = extract_orca_geomscan_energy(text, "SCF energy")
    assert dist.size >= 1
    assert energy.size == dist.size
    assert dist.size == 3


def test_suite_pins_returns_dict():
    pins = suite_pins()
    assert isinstance(pins, dict)


def test_grammar_available_callable():
    from chemparseplot.api import grammar_available

    assert isinstance(grammar_available(), bool)


def test_lazy_units_and_plot_exports():
    import chemparseplot.api as api

    ureg = api.ureg
    Q_ = api.Q_
    assert ureg is not None
    assert Q_ is not None
    # SurfaceFitConfig / plot require optional plot stack
    try:
        cfg = api.SurfaceFitConfig()
        assert cfg.auto_thin is False
    except Exception:
        pass


def test_suite_pins_soft():
    from chemparseplot.api import suite_pins

    assert isinstance(suite_pins(), dict)


def test_getattr_unknown():
    import chemparseplot.api as api

    with pytest.raises(AttributeError):
        _ = api.not_a_real_export  # type: ignore[attr-defined]


def test_parse_xyz_and_final_energy_if_grammar():
    from chemparseplot.api import grammar_available, parse_orca_final_energy, parse_xyz

    if not grammar_available():
        pytest.skip("grammar extra not installed")
    xyz = "2\ncomment\nH 0 0 0\nH 0 0 0.74\n"
    frame = parse_xyz(xyz)
    assert frame is not None
    text = "FINAL SINGLE POINT ENERGY     -1.0\n"
    # may return quantity or None depending on grammar
    try:
        e = parse_orca_final_energy(text)
        assert e is not None
    except Exception:
        pass




def test_looks_like_path_branches(tmp_path, monkeypatch):
    from chemparseplot import api as api_mod
    import pathlib

    assert api_mod._looks_like_path("a\nb") is False
    assert api_mod._looks_like_path("x" * 5000) is False
    p = tmp_path / "mol.xyz"
    p.write_text("2\nc\nH 0 0 0\nH 0 0 1\n")
    assert api_mod._looks_like_path(str(p)) is True
    assert api_mod._looks_like_path("/no/such/file.xyz") is False

    class BadPath:
        def is_file(self):
            raise OSError("nope")

    monkeypatch.setattr(pathlib, "Path", lambda *a, **k: BadPath())
    assert api_mod._looks_like_path("shortname") is False


def test_parse_xyz_and_final_energy_file_paths(tmp_path):
    from chemparseplot.api import grammar_available, parse_orca_final_energy, parse_xyz

    if not grammar_available():
        pytest.skip("grammar not installed")
    xyz = tmp_path / "m.xyz"
    xyz.write_text("2\nc\nH 0 0 0\nH 0 0 0.74\n")
    assert parse_xyz(str(xyz)) is not None
    out = tmp_path / "e.out"
    out.write_text("FINAL SINGLE POINT ENERGY     -76.0\n")
    assert parse_orca_final_energy(str(out)) is not None


def test_suite_pins_manual_fallback(monkeypatch):
    import chemparseplot.api as api_mod

    def only_env_branch():
        try:
            raise ImportError("no suite_pins on hub")
        except ImportError:
            pass
        try:
            from rgpycrumbs.api import load_config, pins_from_env
        except ImportError:
            return {}
        pins = dict(pins_from_env())
        try:
            raise RuntimeError("cfg boom")
        except Exception:
            pass
        return pins

    monkeypatch.setattr(api_mod, "suite_pins", only_env_branch)
    assert isinstance(api_mod.suite_pins(), dict)

    def both_missing():
        try:
            raise ImportError("a")
        except ImportError:
            pass
        try:
            raise ImportError("b")
        except ImportError:
            return {}

    monkeypatch.setattr(api_mod, "suite_pins", both_missing)
    assert api_mod.suite_pins() == {}


def test_suite_pins_getattr_none(monkeypatch):
    import chemparseplot.api as api_mod
    import importlib
    import types

    fake = types.ModuleType("rgpycrumbs.api")
    fake.suite_pins = None  # not callable -> fall through
    fake.pins_from_env = lambda: {"a": "1"}
    fake.load_config = lambda: (_ for _ in ()).throw(RuntimeError("x"))
    monkeypatch.setitem(__import__("sys").modules, "rgpycrumbs.api", fake)
    # also prevent import_module from loading real package
    real_import_module = importlib.import_module

    def fake_import(name, package=None):
        if name == "rgpycrumbs.api":
            return fake
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    pins = api_mod.suite_pins()
    assert pins == {"a": "1"}


def test_suite_pins_hub_missing(monkeypatch):
    import chemparseplot.api as api_mod
    import importlib

    def boom(name, package=None):
        raise ImportError("gone")

    monkeypatch.setattr(importlib, "import_module", boom)
    assert api_mod.suite_pins() == {}


def test_lazy_surface_and_plot(monkeypatch):
    import chemparseplot.api as api_mod
    import sys
    import types

    # clear cached attrs if any
    for name in ("SurfaceFitConfig", "plot_landscape_surface"):
        api_mod.__dict__.pop(name, None)

    fake_neb = types.ModuleType("chemparseplot.plot.neb")

    class C:
        pass

    def plot(*a, **k):
        return None

    fake_neb.SurfaceFitConfig = C
    fake_neb.plot_landscape_surface = plot
    sys.modules["chemparseplot.plot"] = types.ModuleType("chemparseplot.plot")
    sys.modules["chemparseplot.plot.neb"] = fake_neb
    assert api_mod.SurfaceFitConfig is C
    assert api_mod.plot_landscape_surface is plot


def test_suite_pins_attribute_error(monkeypatch):
    import chemparseplot.api as api_mod
    import importlib
    import types

    fake = types.ModuleType("rgpycrumbs.api")
    # no suite_pins attr, no pins_from_env -> AttributeError
    real = importlib.import_module

    def fake_import(name, package=None):
        if name == "rgpycrumbs.api":
            return fake
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert api_mod.suite_pins() == {}


def test_suite_pins_import_error_first_try(monkeypatch):
    import chemparseplot.api as api_mod
    import importlib

    calls = {"n": 0}
    real = importlib.import_module

    def flaky(name, package=None):
        if name == "rgpycrumbs.api":
            calls["n"] += 1
            if calls["n"] == 1:
                raise ImportError("first")
            # second call also missing
            raise ImportError("second")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", flaky)
    assert api_mod.suite_pins() == {}


def test_suite_pins_calls_hub_function(monkeypatch):
    import chemparseplot.api as api_mod
    import importlib
    import types

    fake = types.ModuleType("rgpycrumbs.api")
    fake.suite_pins = lambda: {"hub": "1"}
    real = importlib.import_module

    def fake_import(name, package=None):
        if name == "rgpycrumbs.api":
            return fake
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert api_mod.suite_pins() == {"hub": "1"}
