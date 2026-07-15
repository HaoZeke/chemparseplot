# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Public migration surface from wailord.io.orca into chemparseplot.

Real implementations live in sibling modules; this module re-exports them and
keeps a status checklist for docs / inventory.

.. versionadded:: 1.9.11
"""

from __future__ import annotations

from chemparseplot.parse.orca.freq import (
    IrSpectrumMode,
    VibMode,
    parse_orca_ir_spectrum,
    parse_orca_vibrational_frequencies,
)
from chemparseplot.parse.orca.populations import (
    AtomicCharge,
    parse_orca_populations,
)
from chemparseplot.parse.orca.vpt2 import (
    Vpt2Fundamental,
    parse_orca_vpt2_fundamentals,
)

__all__ = [
    "AtomicCharge",
    "IrSpectrumMode",
    "MIGRATION_CHECKLIST",
    "VibMode",
    "Vpt2Fundamental",
    "parse_orca_ir_frequencies",
    "parse_orca_ir_spectrum",
    "parse_orca_populations",
    "parse_orca_vibrational_frequencies",
    "parse_orca_vpt2_fundamentals",
]

MIGRATION_CHECKLIST: dict[str, str] = {
    "geomscan_surface": "done:chemparseplot.api.extract_orca_geomscan_energy",
    "final_single_point": "done:chemparseplot.api.parse_orca_final_energy",
    "xyz": "done:chemparseplot.api.parse_xyz",
    "vibrational_frequencies": "done:chemparseplot.parse.orca.freq.parse_orca_vibrational_frequencies",
    "ir_spectrum": "done:chemparseplot.parse.orca.freq.parse_orca_ir_spectrum (text + OPI Output.get_ir)",
    "vpt2_fundamentals": "done:chemparseplot.parse.orca.vpt2.parse_orca_vpt2_fundamentals",
    "populations": "done:chemparseplot.parse.orca.populations.parse_orca_populations (text + OPI get_mulliken/get_loewdin)",
}


def parse_orca_ir_frequencies(path_or_text: str):
    """Alias: vibrational frequency table (not IR intensities)."""
    return parse_orca_vibrational_frequencies(path_or_text)
