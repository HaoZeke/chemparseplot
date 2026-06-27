# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Unit registry and quantity helpers via pint.

```{versionadded} 0.0.2
```
"""

import warnings

import pint

# ``:auto:`` lets flexcache persist environment-specific definition paths, which
# breaks as soon as another Pixi env imports the same cached registry.
ureg = pint.UnitRegistry(cache_folder=None)
ureg.define("kcal_mol = kcal / 6.02214076e+23 = kcm")
Q_ = ureg.Quantity

# Silence NEP 18 warning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Q_([])
