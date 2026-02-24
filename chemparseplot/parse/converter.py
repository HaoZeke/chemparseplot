# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
from io import StringIO

import numpy as np


def np_txt(matched_data):
    """Convert a matched text block to a numpy array via ``np.loadtxt``.

    ```{versionadded} 0.0.2
    ```
    """
    datio = StringIO(matched_data)
    return np.loadtxt(datio)
