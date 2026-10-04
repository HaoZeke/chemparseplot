# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Parsers for surrogate-assisted saddle searches and the shared record types."""

from chemparseplot.parse.surrogate.cases import (
    CaseBoard,
    CaseRow,
    attach_baselines,
    parse_cases_csv,
)
from chemparseplot.parse.surrogate.model import (
    AcquisitionEvent,
    BandHistory,
    BandSnapshot,
    CampaignTable,
    CellRecord,
    EvaluatedPoints,
    ScalingTable,
    SearchHistory,
    SingleEndedHistory,
    SurrogateSearch,
)

__all__ = [
    "AcquisitionEvent",
    "BandHistory",
    "BandSnapshot",
    "CampaignTable",
    "CaseBoard",
    "CaseRow",
    "CellRecord",
    "EvaluatedPoints",
    "ScalingTable",
    "SearchHistory",
    "SingleEndedHistory",
    "SurrogateSearch",
    "attach_baselines",
    "parse_cases_csv",
    "read_search",
]


def read_search(path, producer: str = "gpr_optim", **kwargs) -> SurrogateSearch:
    """Parse one search directory with the parser of ``producer``.

    ``producer`` is ``"gpr_optim"`` (a campaign cell) or ``"ml-neb"`` (an ASE
    ML-NEB run directory). Optional dependencies load on use.
    """
    if producer == "gpr_optim":
        from chemparseplot.parse.surrogate.gpr_optim import parse_gpr_optim_cell

        return parse_gpr_optim_cell(path, **kwargs)
    if producer == "ml-neb":
        from chemparseplot.parse.surrogate.mlneb import parse_mlneb_run

        return parse_mlneb_run(path, **kwargs)
    msg = f"unknown producer {producer!r}; expected 'gpr_optim' or 'ml-neb'"
    raise ValueError(msg)
