"""Minimization trajectory parser for eOn output.

Reads structured per-iteration data from the minimization ``.dat`` file
and concatenated trajectory from the movie ``.con`` file, as produced
by eOn with ``write_movies=true``.

.. versionadded:: 1.5.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import readcon
from ase import Atoms

from ._trajectory_common import (
    frame_rows_to_table,
    load_movie_and_table,
    read_first_structure,
    read_optional_first,
)

log = logging.getLogger(__name__)

MIN_METADATA_COLUMNS = ("frame_index", "step_size", "convergence", "energy")


@dataclass(frozen=True, slots=True)
class MinTrajectoryData:
    """Container for a minimization trajectory.

    Attributes
    ----------
    atoms_list
        Per-iteration structures from the movie file.
    dat_df
        Polars DataFrame with per-iteration metrics.
    initial_atoms
        Starting structure (first frame).
    final_atoms
        Final minimized structure (from ``min.con`` or last frame).
    """

    atoms_list: list[Atoms]
    dat_df: pl.DataFrame
    initial_atoms: Atoms
    final_atoms: Atoms


def parse_min_dat(path: Path) -> pl.DataFrame:
    """Read the structured minimization TSV data file.

    Parameters
    ----------
    path
        Path to the minimization ``.dat`` file.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: iteration, step_size, convergence, energy.
    """
    return pl.read_csv(path, separator="\t")


def parse_min_con(path: Path) -> list[Atoms]:
    """Read concatenated structures from the minimization movie file.

    Parameters
    ----------
    path
        Path to the movie ``.con`` file.

    Returns
    -------
    list[Atoms]
        List of ASE Atoms objects, one per iteration.
    """
    return readcon.read_con_as_ase(str(path))


def table_from_min_metadata(frames: list[readcon.ConFrame]) -> pl.DataFrame:
    """Reconstruct minimization metrics from per-frame CON metadata."""

    df = frame_rows_to_table(
        frames,
        MIN_METADATA_COLUMNS,
        allow_leading_incomplete=False,
    )
    if df.is_empty():
        return df
    return df.rename({"frame_index": "iteration"})


def load_min_trajectory(
    job_dir: Path,
    prefix: str = "minimization",
) -> MinTrajectoryData:
    """Load a complete minimization trajectory from an eOn job directory.

    Expects the job to have been run with ``write_movies=true``.

    Parameters
    ----------
    job_dir
        Path to the eOn job output directory.
    prefix
        Movie file prefix (default ``"minimization"``). The movie file is
        ``{prefix}`` and the data file is ``{prefix}.dat``.

    Returns
    -------
    MinTrajectoryData
        Combined trajectory data.

    Raises
    ------
    FileNotFoundError
        If required files are missing.
    """
    atoms_list, dat_df = load_movie_and_table(
        job_dir,
        movie_stem=prefix,
        dat_name=f"{prefix}.dat",
        parse_dat=parse_min_dat,
        metadata_columns=MIN_METADATA_COLUMNS,
        build_table_from_metadata=table_from_min_metadata,
        log_label="minimization",
    )

    # Prefer an explicit final structure matching the movie prefix, then the
    # legacy eOn ``min.con`` output, before falling back to the last movie frame.
    final = read_optional_first(job_dir, (f"{prefix}.con", "min.con")) or atoms_list[-1]

    return MinTrajectoryData(
        atoms_list=atoms_list,
        dat_df=dat_df,
        initial_atoms=atoms_list[0],
        final_atoms=final,
    )
