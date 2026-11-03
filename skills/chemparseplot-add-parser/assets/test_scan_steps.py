# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
import gzip
from pathlib import Path

import numpy as np
import pytest

from chemparseplot.parse.orca.geomscan import extract_energy_data
from chemparseplot.parse.orca.scan_steps import extract_scan_steps

pytestmark = pytest.mark.pure

BANNER = """
         *************************************************************
         *               RELAXED SURFACE SCAN STEP {step:>3}               *
         *                                                           *
         *                 Bond (  1,   0)  : {length:>12.8f}           *
         *************************************************************
"""
TUTORIAL_RUN = (
    Path(__file__).resolve().parents[3] / "doc/source/tutorials/orca/data/scanuhf.gz"
)


def banners(lengths, first=1):
    return "".join(BANNER.format(step=k, length=v) for k, v in enumerate(lengths, first))


def test_steps_in_order():
    pair, length = extract_scan_steps(banners([7.55890395, 7.32930292]))
    assert pair == (1, 0)
    assert str(length.units) == "angstrom"
    assert np.allclose(length.magnitude, [7.55890395, 7.32930292])


def test_no_banner_raises():
    with pytest.raises(ValueError, match="no 'RELAXED SURFACE SCAN STEP'"):
        extract_scan_steps("FINAL SINGLE POINT ENERGY      -1.124039184")


def test_skipped_step_raises():
    text = banners([7.55890395]) + banners([7.09970189], first=3)
    with pytest.raises(ValueError, match="not numbered"):
        extract_scan_steps(text)


@pytest.mark.skipif(not TUTORIAL_RUN.is_file(), reason="needs the ORCA tutorial run")
def test_tutorial_run_matches_the_surface_table():
    with gzip.open(TUTORIAL_RUN, "rt") as fh:
        text = fh.read()
    pair, length = extract_scan_steps(text)
    distance, _ = extract_energy_data(text, "Actual")
    assert pair == (1, 0)
    assert len(length) == 33
    assert np.allclose(length.magnitude, distance.magnitude)
