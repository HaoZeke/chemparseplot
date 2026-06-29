# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Unit registry and quantity helpers via pint.

```{versionadded} 0.0.2
```

Pint is imported on first access to ``ureg`` / ``Q_`` so importing this module
for type checking or attribute existence does not force registry construction
until a quantity is actually needed (callers typically ``from chemparseplot.units import ureg``).
"""

from __future__ import annotations

import warnings
from typing import Any

__all__ = ["Q_", "ureg"]

_ureg = None
_Q = None


def _ensure_registry():
    global _ureg, _Q
    if _ureg is not None:
        return _ureg, _Q
    import pint

    # ``:auto:`` lets flexcache persist environment-specific definition paths, which
    # breaks as soon as another Pixi env imports the same cached registry.
    _ureg = pint.UnitRegistry(cache_folder=None)
    _ureg.define("kcal_mol = kcal / 6.02214076e+23 = kcm")
    _Q = _ureg.Quantity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _Q([])
    return _ureg, _Q


def __getattr__(name: str) -> Any:
    if name == "ureg":
        ureg, _ = _ensure_registry()
        return ureg
    if name == "Q_":
        _, Q = _ensure_registry()
        return Q
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
