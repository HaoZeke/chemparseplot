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

log = logging.getLogger(__name__)


@dataclass
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


def _find_initial_structure(job_dir: Path) -> Atoms | None:
    """Locate the initial/reactant structure in the job directory."""
    for name in ("reactant.con", "pos.con"):
        p = job_dir / name
        if p.exists():
            return readcon.read_con_as_ase(str(p))[0]
    return None


def _load_mode_dat(path: Path) -> np.ndarray | None:
    """Load eigenvector from mode.dat (Nx3 whitespace-separated)."""
    if not path.exists():
        return None
    return np.loadtxt(path)


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
    # Find the movie file (may be "climb" or "climb.con")
    climb_con = job_dir / "climb"
    if not climb_con.exists():
        climb_con = job_dir / "climb.con"
    if not climb_con.exists():
        msg = f"No climb movie file found in {job_dir}"
        raise FileNotFoundError(msg)

    climb_dat = job_dir / "climb.dat"
    if not climb_dat.exists():
        msg = f"No climb.dat found in {job_dir} (was write_movies enabled?)"
        raise FileNotFoundError(msg)

    log.info("Loading dimer trajectory from %s", job_dir)
    atoms_list = parse_climb_con(climb_con)
    dat_df = parse_climb_dat(climb_dat)
    log.info("Loaded %d frames, %d data rows", len(atoms_list), dat_df.height)

    initial = _find_initial_structure(job_dir)
    if initial is None:
        log.warning("No reactant.con or pos.con found; using first movie frame")
        initial = atoms_list[0]

    saddle_path = job_dir / "saddle.con"
    saddle = readcon.read_con_as_ase(str(saddle_path))[0] if saddle_path.exists() else None

    mode = _load_mode_dat(job_dir / "mode.dat")

    return DimerTrajectoryData(
        atoms_list=atoms_list,
        dat_df=dat_df,
        initial_atoms=initial,
        saddle_atoms=saddle,
        mode_vector=mode,
    )
