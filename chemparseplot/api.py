# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Stable public library entry points for chemparseplot.

Prefer this module for new consumers (cookbooks, wailord, notebooks).

**Representative in-process path (parse → typed result):**

.. code-block:: python

    from chemparseplot.api import extract_orca_geomscan_energy

    dist_bohr, energy_Eh = extract_orca_geomscan_energy(orca_out_text, "Actual Energy")
    # dist_bohr, energy_Eh are pint Quantities (bohr, hartree)

Heavy plot stacks are re-exported lazily so bare installs stay light.

Suite pins/config: use ``rgpycrumbs.api`` (shared rgpkgs TOML) — this package
does **not** invent ``~/.config/chemparseplot``.

.. versionadded:: 1.9.8
.. versionchanged:: 1.9.9
   Expose ORCA geomscan energy parse as the stable library path.
.. versionchanged:: 1.9.11
   Expand re-exports (SurfaceFitConfig, plot helpers, basetypes); hub suite_pins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chemparseplot.parse.orca.geomscan import extract_energy_data
from chemparseplot.units import (
    ENERGY_UNITS,
    convert_energy_magnitude,
    energy_quantity,
    normalize_energy_unit,
)

if TYPE_CHECKING:
    from chemparseplot.units import Q_

__all__ = [
    "ENERGY_UNITS",
    "Q_",
    "SurfaceFitConfig",
    "convert_energy_magnitude",
    "energy_quantity",
    "extract_orca_geomscan_energy",
    "grammar_available",
    "normalize_energy_unit",
    "parse_orca_final_energy",
    "parse_xyz",
    "plot_landscape_surface",
    "suite_pins",
    "ureg",
]


def grammar_available() -> bool:
    """True if the optional parsimonious grammar track is importable."""
    from chemparseplot.parse.grammar import grammar_available as _ga

    return _ga()


def _looks_like_path(s: str) -> bool:
    """True for short path-like strings (avoid Path on multi-line blobs)."""
    if "\n" in s or len(s) > 4096:
        return False
    from pathlib import Path

    try:
        return Path(s).is_file()
    except OSError:
        return False


def parse_xyz(path_or_text: str):
    """Parse XYZ via grammar track (file path if exists, else text).

    Requires ``chemparseplot[grammar]``.
    """
    from chemparseplot.parse.grammar.xyz import parse_xyz_file, parse_xyz_text

    if _looks_like_path(path_or_text):
        return parse_xyz_file(path_or_text)
    return parse_xyz_text(path_or_text)


def parse_orca_final_energy(path_or_text: str):
    """Last ``FINAL SINGLE POINT ENERGY`` as a hartree Quantity.

    Grammar track; requires ``chemparseplot[grammar]``.
    """
    from chemparseplot.parse.grammar.orca_text import (
        parse_orca_text_file,
        parse_orca_text_summary,
    )

    if _looks_like_path(path_or_text):
        summary = parse_orca_text_file(path_or_text)
    else:
        summary = parse_orca_text_summary(path_or_text)
    return summary.final_energy


def extract_orca_geomscan_energy(
    data: str, energy_type: str = "Actual Energy"
) -> tuple[Q_, Q_]:
    """Parse ORCA geometry-scan surface energies from output text.

    In-process library path: text → typed ``(distance, energy)`` pint Quantities
    in Bohr and Hartree (ORCA defaults).
    """
    return extract_energy_data(data, energy_type)


def suite_pins() -> dict[str, str]:
    """Return suite package pins from rgpkgs config + env (soft on old hub)."""
    try:
        from rgpycrumbs.api import suite_pins as _suite_pins

        return _suite_pins()
    except ImportError:
        pass
    try:
        from rgpycrumbs.api import load_config, pins_from_env
    except ImportError:
        return {}
    pins = dict(pins_from_env())
    try:
        pins.update(load_config().merged_package_pins_normalized())
    except Exception:  # noqa: BLE001
        pass
    return pins


def __getattr__(name: str) -> Any:
    """Lazy optional re-exports (plot stack / units)."""
    if name in {"ureg", "Q_"}:
        from chemparseplot import units as _units

        return getattr(_units, name)
    if name == "SurfaceFitConfig":
        from chemparseplot.plot.neb import SurfaceFitConfig

        return SurfaceFitConfig
    if name == "plot_landscape_surface":
        from chemparseplot.plot.neb import plot_landscape_surface

        return plot_landscape_surface
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
