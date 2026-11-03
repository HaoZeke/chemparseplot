# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""The journal figure tutorial runs as written and gives what it states.

Its Python blocks carry ``:eval no`` because the doctest job installs only
the core dependencies; here they run in order in one namespace, as a reader
runs them in one session.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import skip_if_not_env

skip_if_not_env("neb")

REPO = Path(__file__).resolve().parents[2]
TUTORIAL = REPO / "docs/orgmode/tutorials/journal_figure.org"
BLOCK = re.compile(r"#\+begin_src python[^\n]*\n(.*?)#\+end_src", re.DOTALL)


@pytest.mark.skipif(not TUTORIAL.exists(), reason="the docs are not in this tree")
def test_journal_figure_tutorial(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from chemparseplot.plot.journal import get_preset

    monkeypatch.chdir(tmp_path)
    namespace = {}
    with matplotlib.rc_context():
        for code in BLOCK.findall(TUTORIAL.read_text()):
            exec(compile(code, str(TUTORIAL), "exec"), namespace)  # noqa: S102
    plt.close("all")

    acs, rsc = get_preset("jctc"), get_preset("pccp")
    assert namespace["converged"]
    assert round(float(namespace["energy"].max()), 2) == 0.37
    assert namespace["width"] == pytest.approx(acs.width(), abs=1 / acs.raster_dpi)
    assert namespace["fig"].get_size_inches()[0] == pytest.approx(rsc.width())
    assert namespace["report"].passes_graphics_contrast() == ("teal", "sky", "magenta")
    for name in ("profile.pdf", "profile.png", "profile-pccp.pdf"):
        assert (tmp_path / name).stat().st_size > 0
