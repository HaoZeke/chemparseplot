# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
import re

import numpy as np

from chemparseplot.parse import converter as conv
from chemparseplot.parse import patterns as pat


def test_numeric_from_match():
    # Create a sample string that matches the TWO_COL_NUM pattern
    sample_data = " 1.23 4.56\n7.89 10.11"

    # Generate the regex pattern for two columns
    pattern = pat.create_multicol_pattern(2, "twocolnum")
    regex = re.compile(pattern)

    match = regex.search(sample_data)
    assert match is not None

    matched_data = match.group("twocolnum")
    result_array = conv.np_txt(matched_data)
    assert isinstance(result_array, np.ndarray)
    assert np.array_equal(result_array, np.array([[1.23, 4.56], [7.89, 10.11]]))
