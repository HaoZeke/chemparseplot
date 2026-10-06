# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Tidy per-case tables of a benchmark campaign.

Producer-independent: one row per case with the oracle calls and wall time of
the search that reached a certified saddle, grouped in boards (benchmark sets).

.. versionadded:: 1.10.0
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

_TRUE = {"true", "1", "yes", "y", "t"}


def _flag(text: str) -> bool:
    return text.strip().lower() in _TRUE


def _num(text: str | None) -> float | None:
    if text is None or text.strip() in {"", "nan", "NaN"}:
        return None
    return float(text)


@dataclass
class CaseRow:
    """One case of a board."""

    case: str
    label: str
    search_calls: float
    validation_calls: float
    wall_s: float
    certified: bool
    reference_wall_s: float | None = None
    wall_sd: float | None = None


@dataclass
class CaseBoard:
    """Cases of one benchmark set, in file order."""

    name: str
    rows: list[CaseRow] = field(default_factory=list)
    baselines: dict[str, dict[str, tuple[float, bool]]] = field(default_factory=dict)

    @property
    def n_certified(self) -> int:
        return sum(r.certified for r in self.rows)


def parse_cases_csv(path: str | Path) -> list[CaseBoard]:
    """Read ``board, case, label, search_calls, validation_calls,
    pipeline_wall_s, certified[, reference_pipeline_wall_s]
    [, pipeline_wall_sd]``. ``pipeline_wall_sd`` is the sample standard
    deviation of repeated runs of the same case.

    Boards and rows keep the order of the file.
    """
    boards: dict[str, CaseBoard] = {}
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        need = {
            "board",
            "case",
            "label",
            "search_calls",
            "validation_calls",
            "pipeline_wall_s",
            "certified",
        }
        missing = need - set(reader.fieldnames or [])
        if missing:
            msg = f"{path} lacks column(s) {sorted(missing)}"
            raise ValueError(msg)
        for r in reader:
            b = boards.setdefault(r["board"], CaseBoard(r["board"]))
            b.rows.append(
                CaseRow(
                    case=r["case"],
                    label=r["label"],
                    search_calls=float(r["search_calls"] or 0),
                    validation_calls=float(r["validation_calls"] or 0),
                    wall_s=float(r["pipeline_wall_s"]),
                    certified=_flag(r["certified"]),
                    reference_wall_s=_num(r.get("reference_pipeline_wall_s")),
                    wall_sd=_num(r.get("pipeline_wall_sd")),
                )
            )
    return list(boards.values())


def attach_baselines(boards: list[CaseBoard], path: str | Path) -> None:
    """Add ``board, case, method, calls, converged`` rows of a long CSV.

    A baseline row whose board or case the cases table does not hold is an
    error, so a mislabelled case cannot vanish silently.
    """
    by_name = {b.name: b for b in boards}
    with Path(path).open(newline="") as fh:
        for r in csv.DictReader(fh):
            board = by_name.get(r["board"])
            if board is None or r["case"] not in {x.case for x in board.rows}:
                msg = f"baseline row for unknown case {r['board']}/{r['case']}"
                raise ValueError(msg)
            board.baselines.setdefault(r["method"], {})[r["case"]] = (
                float(r["calls"]),
                _flag(r["converged"]),
            )
