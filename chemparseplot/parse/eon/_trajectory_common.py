"""Shared helpers for eOn trajectory parsers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import polars as pl
import readcon
from ase import Atoms

log = logging.getLogger(__name__)

T = TypeVar("T")


_COLUMN_CASTERS: dict[str, Callable[[Any], Any]] = {
    "frame_index": int,
    "iteration": int,
    "neb_band": int,
    "neb_bead": int,
    "rotations": int,
    "energy": float,
    "step_size": float,
    "convergence": float,
    "delta_e": float,
    "eigenvalue": float,
    "torque": float,
    "angle": float,
}


def coerce_metadata_value(column: str, value: Any) -> Any:
    """Coerce metadata strings back to the expected scalar type."""
    if value is None:
        return None
    caster = _COLUMN_CASTERS.get(column)
    if caster is None:
        return value
    if isinstance(value, caster):
        return value
    return caster(value)


def resolve_movie_file(job_dir: Path, stem: str) -> Path:
    """Resolve an eOn movie file that may omit the ``.con`` suffix."""

    for candidate in (job_dir / stem, job_dir / f"{stem}.con"):
        if candidate.exists():
            return candidate
    msg = f"No {stem} movie file found in {job_dir}"
    raise FileNotFoundError(msg)


def require_dat_file(job_dir: Path, name: str) -> Path:
    """Return a required trajectory data file or raise a helpful error."""

    path = job_dir / name
    if path.exists():
        return path
    msg = f"No {name} found in {job_dir} (was write_movies enabled?)"
    raise FileNotFoundError(msg)


def read_first_structure(path: Path) -> Atoms:
    """Read the first structure from an eOn CON file."""

    return readcon.read_con_as_ase(str(path))[0]


def read_optional_first(job_dir: Path, names: Sequence[str]) -> Atoms | None:
    """Return the first matching optional CON structure from ``job_dir``."""

    for name in names:
        candidate = job_dir / name
        if candidate.exists():
            return read_first_structure(candidate)
    return None


def load_movie_and_table(
    job_dir: Path,
    *,
    movie_stem: str,
    dat_name: str,
    parse_dat: Callable[[Path], pl.DataFrame],
    metadata_columns: Sequence[str],
    build_table_from_metadata: Callable[[Sequence[readcon.ConFrame]], pl.DataFrame],
    log_label: str,
) -> tuple[list[Atoms], pl.DataFrame]:
    """Load the shared movie/data payload for an eOn trajectory parser."""

    movie_file = resolve_movie_file(job_dir, movie_stem)
    log.info("Loading %s trajectory from %s", log_label, job_dir)
    frames = readcon.read_con(str(movie_file))
    atoms_list = [frame.to_ase() for frame in frames]

    dat_df = build_table_from_metadata(frames)
    if not dat_df.is_empty():
        log.info(
            "Using %s metrics from frame metadata (%d rows)",
            log_label,
            dat_df.height,
        )
    else:
        dat_file = job_dir / dat_name
        if dat_file.exists():
            dat_df = parse_dat(dat_file)
            log.info("Fell back to %s sidecar table %s", log_label, dat_file.name)
        else:
            msg = (
                f"No compatible frame metadata for columns {', '.join(metadata_columns)} "
                f"and no {dat_name} found in {job_dir}"
            )
            raise FileNotFoundError(msg)
    log.info("Loaded %d frames, %d data rows", len(atoms_list), dat_df.height)
    return atoms_list, dat_df


def metadata_value(frame: readcon.ConFrame, key: str) -> Any:
    """Return a typed metadata value from a readcon frame."""

    if key == "frame_index":
        return frame.frame_index
    if key == "energy":
        return frame.energy
    return frame.metadata.get(key)


def frame_rows_to_table(
    frames: Sequence[readcon.ConFrame],
    columns: Sequence[str],
    *,
    allow_leading_incomplete: bool = False,
) -> pl.DataFrame:
    """Build a table from frame metadata when sidecar TSV data is absent."""

    rows: list[dict[str, Any]] = []
    saw_complete_row = False
    for frame in frames:
        row = {column: coerce_metadata_value(column, metadata_value(frame, column)) for column in columns}
        if any(value is None for value in row.values()):
            if saw_complete_row or not allow_leading_incomplete:
                return pl.DataFrame()
            continue
        saw_complete_row = True
        rows.append(row)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).select(list(columns))


def load_optional_payload(path: Path, loader: Callable[[Path], T]) -> T | None:
    """Load an optional file payload if the path exists."""

    if not path.exists():
        return None
    return loader(path)
