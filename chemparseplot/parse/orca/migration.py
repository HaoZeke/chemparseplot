# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tracked ports from wailord.io.orca into chemparseplot.

These symbols mark the public migration surface. Implementations land over
time; until then, callers should keep using wailord experiment helpers only for
orchestration, not as the long-term parse home.

Status
------
- geomscan / Calculated Surface: **done** → ``chemparseplot.api.extract_orca_geomscan_energy``
- final single-point energy: **done** → ``chemparseplot.api.parse_orca_final_energy``
- IR / vibrational frequencies: **stub** (was ``orcaVis.vib_freq``)
- VPT2 fundamentals: **stub** (was wailord VPT2 helpers)
- Mulliken / Loewdin populations: **stub**

.. versionadded:: 1.9.11
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MIGRATION_CHECKLIST",
    "parse_orca_ir_frequencies",
    "parse_orca_populations",
    "parse_orca_vpt2_fundamentals",
]

MIGRATION_CHECKLIST: dict[str, str] = {
    "geomscan_surface": "done:chemparseplot.api.extract_orca_geomscan_energy",
    "final_single_point": "done:chemparseplot.api.parse_orca_final_energy",
    "xyz": "done:chemparseplot.api.parse_xyz",
    "ir_frequencies": "stub:parse_orca_ir_frequencies",
    "vpt2_fundamentals": "stub:parse_orca_vpt2_fundamentals",
    "populations": "stub:parse_orca_populations",
}


def _not_implemented(name: str) -> None:
    raise NotImplementedError(
        f"{name} is not ported yet. Tracked in chemparseplot parse.orca.migration "
        f"(MIGRATION_CHECKLIST). Until then, use wailord.io.orca helpers only as a "
        f"temporary experiment path; do not add new library callers there."
    )


def parse_orca_ir_frequencies(path_or_text: str) -> Any:
    """Parse IR / vibrational frequencies (port of wailord ``orcaVis.vib_freq``)."""
    _not_implemented("parse_orca_ir_frequencies")


def parse_orca_vpt2_fundamentals(path_or_text: str) -> Any:
    """Parse VPT2 fundamental transitions (port of wailord VPT2 helpers)."""
    _not_implemented("parse_orca_vpt2_fundamentals")


def parse_orca_populations(path_or_text: str) -> Any:
    """Parse Mulliken / Loewdin population analysis blocks."""
    _not_implemented("parse_orca_populations")
