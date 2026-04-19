"""Shared helpers for eOn trajectory parsers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

import polars as pl
import readcon
from ase import Atoms

log = logging.getLogger(__name__)

T = TypeVar("T")


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
    log_label: str,
) -> tuple[list[Atoms], pl.DataFrame]:
    """Load the shared movie/data payload for an eOn trajectory parser."""

    movie_file = resolve_movie_file(job_dir, movie_stem)
    dat_file = require_dat_file(job_dir, dat_name)

    log.info("Loading %s trajectory from %s", log_label, job_dir)
    atoms_list = readcon.read_con_as_ase(str(movie_file))
    dat_df = parse_dat(dat_file)
    log.info("Loaded %d frames, %d data rows", len(atoms_list), dat_df.height)
    return atoms_list, dat_df


def load_optional_payload(path: Path, loader: Callable[[Path], T]) -> T | None:
    """Load an optional file payload if the path exists."""

    if not path.exists():
        return None
    return loader(path)
