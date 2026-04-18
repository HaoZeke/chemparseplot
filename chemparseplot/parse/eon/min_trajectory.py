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

log = logging.getLogger(__name__)


@dataclass
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
    movie_file = job_dir / prefix
    if not movie_file.exists():
        movie_file = job_dir / f"{prefix}.con"
    if not movie_file.exists():
        msg = f"No minimization movie file ({prefix}) found in {job_dir}"
        raise FileNotFoundError(msg)

    dat_file = job_dir / f"{prefix}.dat"
    if not dat_file.exists():
        msg = f"No {prefix}.dat found in {job_dir} (was write_movies enabled?)"
        raise FileNotFoundError(msg)

    log.info("Loading minimization trajectory from %s", job_dir)
    atoms_list = parse_min_con(movie_file)
    dat_df = parse_min_dat(dat_file)
    log.info("Loaded %d frames, %d data rows", len(atoms_list), dat_df.height)

    # Final structure: prefer explicit min.con, fall back to last movie frame
    min_con = job_dir / "min.con"
    if min_con.exists():
        final = readcon.read_con_as_ase(str(min_con))[0]
    else:
        final = atoms_list[-1]

    return MinTrajectoryData(
        atoms_list=atoms_list,
        dat_df=dat_df,
        initial_atoms=atoms_list[0],
        final_atoms=final,
    )
