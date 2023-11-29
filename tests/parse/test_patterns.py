# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
import re

import pytest

from chemparseplot.parse import patterns as pat


def test_num_cols_less_than_one():
    with pytest.raises(ValueError):
        pat.create_multicol_pattern(0)


def test_two_col_pattern():
    pattern = pat.create_multicol_pattern(2)
    regex = re.compile(pattern)
    assert regex.search("  1.23  -4.56")
    assert not regex.search("1.23")


def test_three_col_pattern():
    pattern = pat.create_multicol_pattern(3)
    regex = re.compile(pattern)
    assert regex.search(" 1.23 -4.56  7.89")
    assert not regex.search("1.23 -4.56")


def test_custom_pattern_name():
    pattern = pat.create_multicol_pattern(2, "customname")
    assert "(?P<customname>" in pattern
