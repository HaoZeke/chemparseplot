# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Stable public library entry points for chemparseplot.

Prefer this module for new consumers (cookbooks, wailord). Heavy plot/parse
stacks still live under ``chemparseplot.plot`` / ``chemparseplot.parse`` and
may require optional deps (transitional extras until suite uv design fully
covers them).

Suite pins/config: use ``rgpycrumbs.api`` (shared rgpkgs TOML) — this package
does **not** invent ``~/.config/chemparseplot``.

.. versionadded:: 1.9.8
"""

from __future__ import annotations

from chemparseplot.units import (
    ENERGY_UNITS,
    convert_energy_magnitude,
    energy_quantity,
    normalize_energy_unit,
)

__all__ = [
    "ENERGY_UNITS",
    "convert_energy_magnitude",
    "energy_quantity",
    "normalize_energy_unit",
    "suite_pins",
]


def suite_pins() -> dict[str, str]:
    """Return suite package pins from rgpkgs config + env (soft on old hub).

    Empty dict if rgpycrumbs is missing or too old.
    """
    try:
        from rgpycrumbs.api import load_config, pins_from_env
    except ImportError:
        return {}
    pins = dict(pins_from_env())
    try:
        cfg = load_config()
        pins.update(cfg.merged_package_pins_normalized())
    except Exception:  # noqa: BLE001 — soft fail for consumers
        pass
    return pins
