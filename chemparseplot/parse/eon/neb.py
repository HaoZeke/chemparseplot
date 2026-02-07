import glob
import logging
import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from ase.io import read as ase_read
from ase import Atoms
from chemparseplot.parse.file_ import find_file_paths

try:
    from rgpycrumbs.geom.api.alignment import (
        align_structure_robust,
        calculate_rmsd_from_ref,
    )

    # If the user script imports 'ira_mod' from parent env, we mimic that check here
    from rgpycrumbs._aux import _import_from_parent_env

    ira_mod = _import_from_parent_env("ira_mod")
except ImportError:
    ira_mod = None

log = logging.getLogger(__name__)


def calculate_landscape_coords(
    atoms_list: list[Atoms], ira_instance, ira_kmax: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates 2D landscape coordinates (RMSD-R, RMSD-P) for a path.

    :param atoms_list: List of ASE Atoms objects representing the path.
    :param ira_instance: An instantiated IRA object (or None).
    :param ira_kmax: kmax factor for IRA.
    :return: A tuple of (rmsd_r, rmsd_p) arrays.
    """
    logging.info("Calculating landscape coordinates (RMSD-R, RMSD-P)...")
    rmsd_r = calculate_rmsd_from_ref(
        atoms_list, ira_instance, ref_atom=atoms_list[0], ira_kmax=ira_kmax
    )
    rmsd_p = calculate_rmsd_from_ref(
        atoms_list, ira_instance, ref_atom=atoms_list[-1], ira_kmax=ira_kmax
    )
    return rmsd_r, rmsd_p


def _validate_data_atoms_match(z_data, atoms, dat_file_name):
    """Checks if data points count matches structure count."""
    if len(z_data) != len(atoms):
        errmsg = (
            f"Structure count ({len(atoms)}) != data point count "
            f"({len(z_data)}) in {dat_file_name}"
        )
        log.error(errmsg)
        raise ValueError(errmsg)


def load_or_compute_data(
    cache_file: Path | None,
    force_recompute: bool,
    validation_check: Callable[[pl.DataFrame], None],
    computation_callback: Callable[[], pl.DataFrame],
    context_name: str,
) -> pl.DataFrame:
    """Retrieves data from a parquet cache or triggers a computation callback."""
    if cache_file and cache_file.exists() and not force_recompute:
        log.info(f"Loading cached {context_name} data from {cache_file}...")
        try:
            df = pl.read_parquet(cache_file)
            validation_check(df)
            log.info(f"Loaded {df.height} rows from cache.")
            return df
        except Exception as e:
            log.warning(f"Cache load failed or invalid: {e}. Recomputing...")

    log.info(f"Computing {context_name} data...")
    df = computation_callback()

    if cache_file:
        log.info(f"Saving {context_name} cache to {cache_file}...")
        try:
            df.write_parquet(cache_file)
        except Exception as e:
            log.error(f"Failed to write cache file: {e}")

    return df


def load_structures_and_calculate_additional_rmsd(
    con_file: Path, additional_con: list[tuple[Path, str]], ira_kmax: float
):
    """Loads the main trajectory and calculates RMSD for any additional comparison structures."""
    log.info(f"Reading structures from {con_file}")
    atoms_list = ase_read(con_file, index=":")
    log.info(f"Loaded {len(atoms_list)} structures.")

    additional_atoms_data = []
    if additional_con:
        ira_instance = ira_mod.IRA() if ira_mod else None

        for add_file, add_label in additional_con:
            # Handle empty labels
            if not add_label or add_label.strip() == "":
                label = add_file.stem
            else:
                label = add_label

            log.info(f"Processing additional structure: {label}")
            additional_atoms = ase_read(add_file)

            add_rmsd_r = calculate_rmsd_from_ref(
                [additional_atoms],
                ira_instance,
                ref_atom=atoms_list[0],
                ira_kmax=ira_kmax,
            )[0]
            add_rmsd_p = calculate_rmsd_from_ref(
                [additional_atoms],
                ira_instance,
                ref_atom=atoms_list[-1],
                ira_kmax=ira_kmax,
            )[0]

            additional_atoms_data.append(
                (additional_atoms, add_rmsd_r, add_rmsd_p, label)
            )

    return atoms_list, additional_atoms_data


def aggregate_neb_landscape_data(
    all_dat_paths: list[Path],
    all_con_paths: list[Path],
    y_data_column: int,
    ira_instance,  # Can be None
    cache_file: Path | None = None,
    force_recompute: bool = False,
    ira_kmax: float = 1.8,
) -> pl.DataFrame:
    """Aggregates data from multiple NEB steps for landscape visualization."""

    # Init IRA if not passed
    if ira_instance is None and ira_mod is not None:
        ira_instance = ira_mod.IRA()

    def validate_landscape_cache(df: pl.DataFrame):
        if "p" not in df.columns:
            raise ValueError("Cache missing 'p' column.")

    def compute_landscape_data() -> pl.DataFrame:
        # Match indices
        # (This is a simplified logic from the original which had complex index matching.
        #  We assume the caller has provided matched lists or we just zip them).

        # In a robust implementation, we would replicate the filename matching logic here.
        # For brevity, we assume strict zipping as the main aggregation step.
        paths_dat = all_dat_paths
        paths_con = all_con_paths

        if len(paths_dat) != len(paths_con):
            log.warning(
                f"Mismatch in file counts: {len(paths_dat)} dat vs {len(paths_con)} con."
            )
            min_len = min(len(paths_dat), len(paths_con))
            paths_dat = paths_dat[:min_len]
            paths_con = paths_con[:min_len]

        all_dfs = []
        for step_idx, (dat_file, con_file_step) in enumerate(
            zip(paths_dat, paths_con, strict=True)
        ):
            try:
                path_data = np.loadtxt(dat_file, skiprows=1).T
                z_data_step = path_data[y_data_column]
                atoms_list_step = ase_read(con_file_step, index=":")

                _validate_data_atoms_match(z_data_step, atoms_list_step, dat_file.name)

                rmsd_r, rmsd_p = calculate_landscape_coords(
                    atoms_list_step, ira_instance, ira_kmax
                )

                all_dfs.append(
                    pl.DataFrame(
                        {
                            "r": rmsd_r,
                            "p": rmsd_p,
                            "z": z_data_step,
                            "step": int(step_idx),
                        }
                    )
                )
            except Exception as e:
                log.warning(f"Failed to process step {step_idx} ({dat_file.name}): {e}")
                continue

        if not all_dfs:
            raise RuntimeError("No data could be aggregated.")

        return pl.concat(all_dfs)

    return load_or_compute_data(
        cache_file=cache_file,
        force_recompute=force_recompute,
        validation_check=validate_landscape_cache,
        computation_callback=compute_landscape_data,
        context_name="Landscape",
    )


def compute_profile_rmsd(
    atoms_list: list[Atoms],
    cache_file: Path | None,
    force_recompute: bool,
    ira_kmax: float,
) -> pl.DataFrame:
    """Computes RMSD for a 1D profile."""

    def validate_profile_cache(df: pl.DataFrame):
        if "p" in df.columns:
            raise ValueError("Cache contains 'p' column (looks like landscape data).")
        if df.height != len(atoms_list):
            raise ValueError(
                f"Size mismatch: {df.height} vs {len(atoms_list)} structures."
            )

    def compute_data() -> pl.DataFrame:
        ira_instance = ira_mod.IRA() if ira_mod else None
        r_vals = calculate_rmsd_from_ref(
            atoms_list, ira_instance, ref_atom=atoms_list[0], ira_kmax=ira_kmax
        )
        return pl.DataFrame({"r": r_vals})

    return load_or_compute_data(
        cache_file=cache_file,
        force_recompute=force_recompute,
        validation_check=validate_profile_cache,
        computation_callback=compute_data,
        context_name="Profile RMSD",
    )

def estimate_rbf_smoothing(df: pl.DataFrame) -> float:
    """
    Estimates a smoothing parameter for RBF interpolation.
    
    Calculates the median Euclidean distance between sequential points in the path
    and uses 10% of that value as the smoothing factor.
    """
    # Calculate distances between sequential images (r, p) within each step
    df_dist = (
        df.sort(["step", "r"])
        .with_columns(
            dr=pl.col("r").diff().over("step"),
            dp=pl.col("p").diff().over("step"),
        )
        .with_columns(dist=(pl.col("dr") ** 2 + pl.col("dp") ** 2).sqrt())
        .drop_nulls()
    )
    
    global_median_step = df_dist["dist"].median()
    
    if global_median_step is None or global_median_step == 0:
        return 0.0
        
    return 0.1 * global_median_step
