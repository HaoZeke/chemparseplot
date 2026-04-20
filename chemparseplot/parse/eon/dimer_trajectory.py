"""Dimer/saddle search trajectory parser for eOn output.

Reads structured per-iteration data from ``climb.dat`` (TSV) and
concatenated trajectory from ``climb.con`` (movie file), as produced
by eOn with ``write_movies=true``.

.. versionadded:: 1.5.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import readcon
from ase import Atoms

from ._trajectory_common import (
    frame_rows_to_table,
    load_movie_and_table,
    load_optional_payload,
    read_first_structure,
    read_optional_first,
)

log = logging.getLogger(__name__)

DIMER_METADATA_COLUMNS = (
    "frame_index",
    "step_size",
    "delta_e",
    "convergence",
    "eigenvalue",
    "torque",
    "angle",
    "rotations",
)


@dataclass(frozen=True, slots=True)
class DimerTrajectoryData:
    """Container for a dimer/saddle search trajectory.

    Attributes
    ----------
    atoms_list
        Per-iteration structures from the movie file.
    dat_df
        Polars DataFrame with per-iteration metrics from ``climb.dat``.
    initial_atoms
        Starting structure (from ``reactant.con`` or ``pos.con``).
    saddle_atoms
        Final saddle point structure (from ``saddle.con``), or None.
    mode_vector
        Eigenvector at the saddle (from ``mode.dat``), or None.
    """

    atoms_list: list[Atoms]
    dat_df: pl.DataFrame
    initial_atoms: Atoms
    saddle_atoms: Atoms | None = None
    mode_vector: np.ndarray | None = None


def parse_climb_dat(path: Path) -> pl.DataFrame:
    """Read the structured ``climb.dat`` TSV file.

    Parameters
    ----------
    path
        Path to the ``climb.dat`` file.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns matching the TSV header.
    """
    return pl.read_csv(path, separator="\t")


def parse_climb_con(path: Path) -> list[Atoms]:
    """Read concatenated structures from the ``climb.con`` movie file.

    Parameters
    ----------
    path
        Path to the ``climb`` or ``climb.con`` file.

    Returns
    -------
    list[Atoms]
        List of ASE Atoms objects, one per iteration.
    """
    # eOn .con files may not have a .con extension for movie files
    return readcon.read_con_as_ase(str(path))


def _load_mode_dat(path: Path) -> np.ndarray | None:
    """Load eigenvector from mode.dat (Nx3 whitespace-separated)."""
    if not path.exists():
        return None
    return np.loadtxt(path)


def table_from_dimer_metadata(frames: list[readcon.ConFrame]) -> pl.DataFrame:
    """Reconstruct climb metrics from per-frame CON metadata."""

    df = frame_rows_to_table(
        frames,
        DIMER_METADATA_COLUMNS,
        allow_leading_incomplete=True,
    )
    if df.is_empty():
        return df
    return df.rename({"frame_index": "iteration"})


def load_dimer_trajectory(job_dir: Path) -> DimerTrajectoryData:
    """Load a complete dimer/saddle search trajectory from an eOn job directory.

    Expects the job to have been run with ``write_movies=true``.

    Parameters
    ----------
    job_dir
        Path to the eOn job output directory containing ``climb``,
        ``climb.dat``, ``saddle.con``, etc.

    Returns
    -------
    DimerTrajectoryData
        Combined trajectory data.

    Raises
    ------
    FileNotFoundError
        If required files (``climb``, ``climb.dat``) are missing.
    """
    atoms_list, dat_df = load_movie_and_table(
        job_dir,
        movie_stem="climb",
        dat_name="climb.dat",
        parse_dat=parse_climb_dat,
        metadata_columns=DIMER_METADATA_COLUMNS,
        build_table_from_metadata=table_from_dimer_metadata,
        log_label="dimer",
    )

    initial = read_optional_first(job_dir, ("reactant.con", "pos.con"))
    if initial is None:
        log.warning("No reactant.con or pos.con found; using first movie frame")
        initial = atoms_list[0]

    saddle = load_optional_payload(job_dir / "saddle.con", read_first_structure)
    mode = load_optional_payload(job_dir / "mode.dat", _load_mode_dat)

    return DimerTrajectoryData(
        atoms_list=atoms_list,
        dat_df=dat_df,
        initial_atoms=initial,
        saddle_atoms=saddle,
        mode_vector=mode,
    )
