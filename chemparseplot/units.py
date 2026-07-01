# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Unit registry and quantity helpers via pint.

```{versionadded} 0.0.2
```

Pint is imported on first access so a bare package import stays light. Energy
presentation units (``eV``, ``kcal/mol``, ``kJ/mol``) are defined on a small
chemical-energy dimension so conversions used by plot/parse helpers go through
``Quantity.to`` instead of ad-hoc floats.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

__all__ = [
    "ENERGY_UNITS",
    "Q_",
    "as_energy_quantity",
    "convert_energy_magnitude",
    "energy_quantity",
    "normalize_energy_unit",
    "ureg",
]

# Canonical presentation names used across CLIs and plot APIs.
ENERGY_UNITS = ("eV", "kcal/mol", "kJ/mol")

# Pint names for the chemical-energy dimension (plot/presentation, not SI molar).
_ENERGY_PINT_NAME = {
    "eV": "chem_eV",
    "kcal/mol": "chem_kcal_mol",
    "kJ/mol": "chem_kJ_mol",
}

# Factors relative to eV (ASE / chemical convention).
_ENERGY_TO_EV = {
    "eV": 1.0,
    "kcal/mol": 1.0 / 23.06054783061903,
    "kJ/mol": 1.0 / 96.48533212331002,
}

_ureg = None
_Q = None


def normalize_energy_unit(unit: str) -> str:
    """Map aliases to a canonical energy presentation unit name."""
    key = unit.strip()
    aliases = {
        "ev": "eV",
        "eV": "eV",
        "kcal/mol": "kcal/mol",
        "kcal_mol": "kcal/mol",
        "kcm": "kcal/mol",
        "kj/mol": "kJ/mol",
        "kJ/mol": "kJ/mol",
        "kJ_mol": "kJ/mol",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported energy unit {unit!r}; expected one of {ENERGY_UNITS}"
        ) from exc


def _ensure_registry():
    global _ureg, _Q
    if _ureg is not None:
        return _ureg, _Q
    import pint

    # Avoid flexcache path bleed across Pixi envs.
    _ureg = pint.UnitRegistry(cache_folder=None)
    # Historical alias used in older docs/examples.
    _ureg.define("kcal_mol = kcal / 6.02214076e+23 = kcm")
    # Chemical energy presentation dimension (plot axes / CLI --energy-unit).
    _ureg.define("chem_eV = [chem_energy]")
    _ureg.define("chem_kcal_mol = chem_eV / 23.06054783061903")
    _ureg.define("chem_kJ_mol = chem_eV / 96.48533212331002")
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


def as_energy_quantity(values: Any, unit: str = "eV"):
    """Wrap *values* as a pint ``Quantity`` in a chemical-energy presentation unit."""
    ureg, Q = _ensure_registry()
    canon = normalize_energy_unit(unit)
    arr = np.asarray(values, dtype=float)
    return Q(arr, _ENERGY_PINT_NAME[canon])


def energy_quantity(values: Any, unit: str = "eV"):
    """Alias of :func:`as_energy_quantity`."""
    return as_energy_quantity(values, unit)


def convert_energy_magnitude(
    values: Any, unit: str, *, source_unit: str = "eV"
) -> np.ndarray:
    """Convert energy magnitudes between presentation units via pint."""
    ureg, _ = _ensure_registry()
    src = normalize_energy_unit(source_unit)
    dst = normalize_energy_unit(unit)
    if src == dst:
        return np.asarray(values, dtype=float)
    q = as_energy_quantity(values, src)
    out = q.to(_ENERGY_PINT_NAME[dst])
    return np.asarray(out.magnitude, dtype=float)
