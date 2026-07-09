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

Heavy plot stacks still live under ``chemparseplot.plot`` (optional deps /
transitional extras until suite uv design fully covers them).

Suite pins/config: use ``rgpycrumbs.api`` (shared rgpkgs TOML) — this package
does **not** invent ``~/.config/chemparseplot``.

.. versionadded:: 1.9.8
.. versionchanged:: 1.9.9
   Expose ORCA geomscan energy parse as the stable library path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    "convert_energy_magnitude",
    "energy_quantity",
    "extract_orca_geomscan_energy",
    "grammar_available",
    "normalize_energy_unit",
    "parse_orca_final_energy",
    "parse_xyz",
    "suite_pins",
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

    Parameters
    ----------
    data:
        Blob of ORCA output containing ``Calculated Surface`` blocks.
    energy_type:
        ``\"Actual Energy\"`` or ``\"SCF energy\"`` (substring match in the block).

    Returns
    -------
    tuple[Q_, Q_]
        Distances (bohr) and energies (hartree). Empty quantities if no match.
    """
    return extract_energy_data(data, energy_type)


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
