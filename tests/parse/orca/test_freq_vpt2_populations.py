# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""Real ORCA IR / VPT2 / population parsers (text + OPI IrMode)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chemparseplot.api import (
    parse_orca_ir_spectrum,
    parse_orca_populations,
    parse_orca_vibrational_frequencies,
    parse_orca_vpt2_fundamentals,
)
from chemparseplot.parse.orca import freq as freq_mod
from chemparseplot.parse.orca.freq import (
    IrSpectrumMode,
    VibMode,
    parse_orca_ir_spectrum as parse_ir,
    parse_orca_vibrational_frequencies as pv,
)
from chemparseplot.parse.orca.migration import (
    MIGRATION_CHECKLIST,
    parse_orca_ir_frequencies,
)
from chemparseplot.parse.orca.populations import parse_orca_populations as pp
from chemparseplot.parse.orca.vpt2 import parse_orca_vpt2_fundamentals as pvpt2

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "orca"


def test_migration_checklist_marks_done():
    assert MIGRATION_CHECKLIST["ir_spectrum"].startswith("done")
    assert MIGRATION_CHECKLIST["vpt2_fundamentals"].startswith("done")
    assert MIGRATION_CHECKLIST["populations"].startswith("done")


def test_vibrational_frequencies_h2o():
    modes = parse_orca_vibrational_frequencies(FIX / "vib_freq_h2o.out")
    assert len(modes) >= 6
    assert all(isinstance(m, VibMode) for m in modes)
    assert modes[0].freq_cm1 == 0.0
    assert not modes[0].imaginary
    assert any(m.freq_cm1 > 1000 for m in modes)


def test_vibrational_frequencies_from_path_str(tmp_path):
    p = tmp_path / "v.out"
    p.write_text(
        "VIBRATIONAL FREQUENCIES\n"
        "-----------------------\n"
        "   0:         0.00 cm**-1\n"
        "   1:       100.00 cm**-1\n"
    )
    modes = pv(str(p))
    assert len(modes) == 2
    assert modes[1].freq_cm1 == 100.0


def test_vibrational_frequencies_imaginary_from_text():
    text = """
VIBRATIONAL FREQUENCIES
-----------------------
   0:         0.00 cm**-1
   6:       -21.38 cm**-1 ***imaginary mode***
   7:        52.96 cm**-1
"""
    modes = pv(text)
    assert len(modes) == 3
    assert modes[1].imaginary
    assert modes[1].freq_cm1 == pytest.approx(-21.38)


def test_vibrational_empty_when_no_header():
    assert pv("nothing") == []


def test_ir_spectrum_text():
    modes = parse_orca_ir_spectrum(FIX / "ir_spectrum_h2o.out", backend="text")
    assert len(modes) >= 1
    m0 = modes[0]
    assert m0.mode >= 0
    assert m0.freq_cm1 > 0
    assert m0.intensity_km_mol >= 0
    assert len(m0.dipole) == 3


def test_ir_spectrum_manual_line_fallback(monkeypatch):
    # Force IrMode.from_string to fail so the cleaned-token path runs
    class Boom:
        @staticmethod
        def from_string(line):
            raise ValueError("force fallback")

    monkeypatch.setitem(
        __import__("sys").modules,
        "opi.output.ir_mode",
        SimpleNamespace(IrMode=Boom),
    )
    # also patch import path used inside _parse_ir_text
    import opi.output.ir_mode as imod

    monkeypatch.setattr(imod, "IrMode", Boom)
    text = """
IR SPECTRUM
-----------
 Mode    freq (cm**-1)   T**2         TX         TY         TZ
-------------------------------------------------------------------
   6:      1639.47   57.55  ( -0.1  0.2  -0.3)
"""
    modes = freq_mod._parse_ir_text(text)
    assert len(modes) == 1
    assert modes[0].freq_cm1 == pytest.approx(1639.47)
    assert modes[0].dipole == pytest.approx((-0.1, 0.2, -0.3))


def test_ir_opi_backend_mock(monkeypatch, tmp_path):
    class FakeIr:
        def __init__(self):
            self.wavenumber = 100.0
            self.intensity = 1.5
            self.dipole = (0.0, 0.1, 0.0)
            self.eps = 0.01

    class FakeOut:
        def __init__(self, basename, working_dir=None):
            self.basename = basename

        def get_ir(self):
            return {6: FakeIr()}

    monkeypatch.setattr(freq_mod, "opi_available", lambda: True)
    monkeypatch.setattr(
        "chemparseplot.parse.orca._opi.get_opi_output_class", lambda: FakeOut
    )
    modes = parse_ir("ignored", backend="opi", basename="job", working_dir=tmp_path)
    assert len(modes) == 1
    assert modes[0].mode == 6
    assert modes[0].eps == 0.01


def test_ir_opi_empty_dict(monkeypatch, tmp_path):
    class FakeOut:
        def __init__(self, *a, **k):
            pass

        def get_ir(self):
            return None

    monkeypatch.setattr(freq_mod, "opi_available", lambda: True)
    monkeypatch.setattr(
        "chemparseplot.parse.orca._opi.get_opi_output_class", lambda: FakeOut
    )
    assert parse_ir("x", backend="opi", basename="job", working_dir=tmp_path) == []


def test_ir_opi_required_raises_without_opi(monkeypatch):
    monkeypatch.setattr(freq_mod, "opi_available", lambda: False)

    def _boom():
        raise ImportError("no opi")

    monkeypatch.setattr(
        "chemparseplot.parse.orca._opi.get_opi_output_class", _boom
    )
    with pytest.raises(ImportError):
        parse_ir("x", backend="opi", basename="job")


def test_vpt2_fundamentals():
    rows = parse_orca_vpt2_fundamentals(FIX / "vpt2_fundamentals.out")
    assert len(rows) == 3
    assert rows[0].mode == 1
    assert rows[0].harmonic_cm1 == pytest.approx(1638.78)


def test_vpt2_from_path_str(tmp_path):
    p = tmp_path / "v.out"
    p.write_text(
        "Fundamental transition\n"
        " ---------------------------------------------\n"
        " Mode[i] w[i] [1/cm]  v[i] [1/cm]  Diff [1/cm]\n"
        " ---------------------------------------------\n"
        "    1      100.0      90.0      -10.0\n"
        " ---------------------------------------------\n"
    )
    rows = pvpt2(str(p))
    assert len(rows) == 1
    assert rows[0].diff_cm1 == -10.0


def test_vpt2_missing_raises():
    with pytest.raises(ValueError, match="VPT2"):
        pvpt2("no fundamentals here")


def test_populations_mulliken_and_loewdin():
    text = (FIX / "mulliken_loewdin.out").read_text()
    mull = parse_orca_populations(text, kind="Mulliken")
    assert len(mull) >= 2
    loe = pp(text, kind="Loewdin")
    assert len(loe) >= 2


def test_populations_from_file_path():
    mull = parse_orca_populations(FIX / "mulliken_loewdin.out", kind="Mulliken")
    assert mull


def test_populations_missing_raises():
    with pytest.raises(ValueError, match="Mulliken"):
        parse_orca_populations("no charges", kind="Mulliken")


def test_populations_bad_kind():
    with pytest.raises(ValueError, match="Unsupported"):
        parse_orca_populations("x", kind="NotAMethod")  # type: ignore[arg-type]


def test_populations_opi_mock(monkeypatch, tmp_path):
    from chemparseplot.parse.orca import populations as pop_mod

    class Block:
        atomiccharges = [[0.1, -0.1]]

    class FakeOut:
        def __init__(self, *a, **k):
            pass

        def get_mulliken(self):
            return [Block()]

        def get_loewdin(self):
            return [Block()]

    monkeypatch.setattr(pop_mod, "opi_available", lambda: True)
    monkeypatch.setattr(
        "chemparseplot.parse.orca._opi.get_opi_output_class", lambda: FakeOut
    )
    rows = pp("ignored", kind="Mulliken", backend="opi", basename="j", working_dir=tmp_path)
    assert len(rows) == 2
    assert rows[0].charge == 0.1


def test_populations_opi_empty_falls_to_text(monkeypatch):
    from chemparseplot.parse.orca import populations as pop_mod

    class FakeOut:
        def __init__(self, *a, **k):
            pass

        def get_mulliken(self):
            return None

    monkeypatch.setattr(pop_mod, "opi_available", lambda: True)
    monkeypatch.setattr(
        "chemparseplot.parse.orca._opi.get_opi_output_class", lambda: FakeOut
    )
    text = (FIX / "mulliken_loewdin.out").read_text()
    rows = pp(text, kind="Mulliken", backend="auto", basename="j")
    assert rows


def test_ir_frequencies_alias():
    modes = parse_orca_ir_frequencies(FIX / "vib_freq_h2o.out")
    assert len(modes) >= 1


def test_api_lazy_orcas():
    import chemparseplot.api as api

    assert callable(api.parse_orca_vibrational_frequencies)
    assert callable(api.parse_orca_ir_spectrum)
    assert callable(api.parse_orca_vpt2_fundamentals)
    assert callable(api.parse_orca_populations)


def test_basetypes_import_from_chemparseplot():
    from chemparseplot.basetypes import SaddleMeasure, nebiter, nebpath

    p = nebpath(0.0, 0.0, 0.0)
    assert nebiter(1, p).iteration == 1
    assert SaddleMeasure().success is False


def test_as_text_oserror(monkeypatch):
    class P:
        def __init__(self, *a, **k):
            pass

        def is_file(self):
            raise OSError("x")

        def read_text(self, **k):
            raise OSError("x")

    monkeypatch.setattr(freq_mod, "Path", P)
    # string without newline triggers path probe
    assert freq_mod._as_text("shortpath") == "shortpath"


def test_vib_header_without_modes_within_window():
    text = "VIBRATIONAL FREQUENCIES\n" + ("junk\n" * 25)
    assert pv(text) == []


def test_ir_header_without_data_timeout():
    text = "IR SPECTRUM\n" + ("junk\n" * 35)
    assert freq_mod._parse_ir_text(text) == []


def test_ir_text_break_on_short_fallback_line(monkeypatch):
    class Boom:
        @staticmethod
        def from_string(line):
            raise ValueError("x")

    import opi.output.ir_mode as imod

    monkeypatch.setattr(imod, "IrMode", Boom)
    text = """
IR SPECTRUM
 Mode freq T2 TX TY TZ
---
   1: 10.0 1.0
"""
    # short cleaned line should break
    assert freq_mod._parse_ir_text(text) == []


def test_ir_auto_without_opi_uses_text(monkeypatch):
    monkeypatch.setattr(freq_mod, "opi_available", lambda: False)
    modes = parse_ir(FIX / "ir_spectrum_h2o.out", backend="auto", basename="job")
    assert modes


def test_vpt2_value_error_row():
    text = """
Fundamental transition
---
 Mode w v d
---
    not a row of numbers
---
"""
    with pytest.raises(ValueError):
        pvpt2(text)


def test_populations_loewdin_opi(monkeypatch, tmp_path):
    from chemparseplot.parse.orca import populations as pop_mod

    class Block:
        atomiccharges = [[0.2]]

    class FakeOut:
        def __init__(self, *a, **k):
            pass

        def get_loewdin(self):
            return [Block()]

        def get_mulliken(self):
            raise RuntimeError("fail")

    monkeypatch.setattr(pop_mod, "opi_available", lambda: True)
    monkeypatch.setattr(
        "chemparseplot.parse.orca._opi.get_opi_output_class", lambda: FakeOut
    )
    rows = pp("x", kind="Loewdin", backend="opi", basename="j", working_dir=tmp_path)
    assert rows[0].charge == 0.2


def test_populations_opi_exception_returns_empty(monkeypatch, tmp_path):
    from chemparseplot.parse.orca import populations as pop_mod

    class FakeOut:
        def __init__(self, *a, **k):
            pass

        def get_mulliken(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(pop_mod, "opi_available", lambda: True)
    monkeypatch.setattr(
        "chemparseplot.parse.orca._opi.get_opi_output_class", lambda: FakeOut
    )
    assert pop_mod._parse_opi("j", tmp_path, "Mulliken") == []


def test_populations_opi_backend_empty_returns_empty(monkeypatch, tmp_path):
    from chemparseplot.parse.orca import populations as pop_mod

    class FakeOut:
        def __init__(self, *a, **k):
            pass

        def get_mulliken(self):
            return []

    monkeypatch.setattr(pop_mod, "opi_available", lambda: True)
    monkeypatch.setattr(
        "chemparseplot.parse.orca._opi.get_opi_output_class", lambda: FakeOut
    )
    assert pp("x", kind="Mulliken", backend="opi", basename="j", working_dir=tmp_path) == []


def test_populations_bad_first_token():
    text = """
MULLIKEN ATOMIC CHARGES
-----------------------
   H H : 0.0
"""
    with pytest.raises(ValueError):
        pp(text, kind="Mulliken")


def test_vpt2_as_text_oserror(monkeypatch):
    from chemparseplot.parse.orca import vpt2 as vpt2_mod

    class P:
        def __init__(self, *a, **k):
            pass

        def is_file(self):
            raise OSError("x")

    monkeypatch.setattr(vpt2_mod, "Path", P)
    # when path_or_text is Path instance it goes other branch — use str
    assert "Fundamental" not in vpt2_mod._as_text("nopath")
    with pytest.raises(ValueError):
        pvpt2("nopath")


def test_vpt2_short_parts_break():
    text = """
Fundamental transition
---
 Mode
---
    1  only_two
---
"""
    with pytest.raises(ValueError):
        pvpt2(text)


def test_populations_no_numeric_tokens():
    text = """
MULLIKEN ATOMIC CHARGES
-----------------------
   0 H :
"""
    with pytest.raises(ValueError, match="Mulliken"):
        pp(text, kind="Mulliken")


def test_populations_opi_backend_empty_list_explicit(monkeypatch, tmp_path):
    from chemparseplot.parse.orca import populations as pop_mod

    class FakeOut:
        def __init__(self, *a, **k):
            pass

        def get_mulliken(self):
            return []

    monkeypatch.setattr(pop_mod, "opi_available", lambda: True)
    monkeypatch.setattr(
        "chemparseplot.parse.orca._opi.get_opi_output_class", lambda: FakeOut
    )
    # backend opi + empty -> return [] without text fallback
    assert (
        pp("no text charges", kind="Mulliken", backend="opi", basename="j", working_dir=tmp_path)
        == []
    )


def test_ir_opi_backend_when_auto_and_empty_falls_to_text(monkeypatch, tmp_path):
    class FakeOut:
        def __init__(self, *a, **k):
            pass

        def get_ir(self):
            return {}

    monkeypatch.setattr(freq_mod, "opi_available", lambda: True)
    monkeypatch.setattr(
        "chemparseplot.parse.orca._opi.get_opi_output_class", lambda: FakeOut
    )
    # empty dict is falsy -> fall through to text for auto
    modes = parse_ir(FIX / "ir_spectrum_h2o.out", backend="auto", basename="j", working_dir=tmp_path)
    assert modes


def test_ir_opi_backend_returns_empty_without_fallback(monkeypatch, tmp_path):
    class FakeOut:
        def __init__(self, *a, **k):
            pass

        def get_ir(self):
            return None

    monkeypatch.setattr(freq_mod, "opi_available", lambda: True)
    monkeypatch.setattr(
        "chemparseplot.parse.orca._opi.get_opi_output_class", lambda: FakeOut
    )
    assert parse_ir("textignored", backend="opi", basename="j", working_dir=tmp_path) == []


def test_ir_text_successful_irmode_path():
    # real OPI IrMode parser on a well-formed line
    text = """
IR SPECTRUM
-----------

 Mode    freq (cm**-1)   T**2         TX         TY         TZ
-------------------------------------------------------------------
   6:   1535.92   0.012167   61.49  ( 0.028738 -0.018467 -0.036127)
"""
    modes = freq_mod._parse_ir_text(text)
    assert len(modes) == 1
    assert modes[0].mode == 6
    assert modes[0].eps == pytest.approx(0.012167)


def test_ir_backend_opi_when_unavailable_raises_then_empty(monkeypatch):
    monkeypatch.setattr(freq_mod, "opi_available", lambda: False)

    def boom():
        raise ImportError("missing opi")

    monkeypatch.setattr(
        "chemparseplot.parse.orca._opi.get_opi_output_class", boom
    )
    with pytest.raises(ImportError):
        parse_ir("x", backend="opi", basename="job")


def test_ir_opi_backend_without_basename_returns_empty_list():
    # backend opi but no basename -> skip first block, hit bare return []
    assert parse_ir("whatever text", backend="opi") == []


def test_populations_backend_opi_without_opi_available(monkeypatch):
    from chemparseplot.parse.orca import populations as pop_mod

    monkeypatch.setattr(pop_mod, "opi_available", lambda: False)
    # basename set but opi unavailable -> fall through; backend opi returns []
    assert (
        pp("no charges here", kind="Mulliken", backend="opi", basename="j") == []
    )


def test_populations_eof_without_sum_footer():
    text = "MULLIKEN ATOMIC CHARGES\n---------------\n   0 H :   -0.1\n   1 H :    0.1\n"
    rows = pp(text, kind="Mulliken")
    assert len(rows) == 2


def test_vpt2_eof_without_dash_footer():
    text = (
        "Fundamental transition\n"
        " Mode[i] w[i] v[i] Diff\n"
        "    1      100.0      90.0      -10.0\n"
    )
    rows = pvpt2(text)
    assert len(rows) == 1
