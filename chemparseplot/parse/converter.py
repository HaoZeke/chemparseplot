# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
from io import StringIO

import numpy as np


def np_txt(matched_data):
    datio = StringIO(matched_data)
    return np.loadtxt(datio)
