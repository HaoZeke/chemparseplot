import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
from ase import Atoms
from ase.io import read as ase_read

from chemparseplot.parse.neb_utils import (
    compute_synthetic_gradients,
    create_landscape_dataframe,
)

try:
    from rgpycrumbs._aux import _import_from_parent_env
    from rgpycrumbs.geom.api.alignment import (
        calculate_rmsd_from_ref,
    )

    ira_mod = _import_from_parent_env("ira_mod")
except ImportError:
    ira_mod = None

log = logging.getLogger(__name__)


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
    con_file: Path,
    additional_con: list[tuple[Path, str]],
    ira_kmax: float,
    sp_file: Path | None = None,
):
    """Loads the main trajectory and calculates RMSD for any additional comparison structures."""
    log.info(f"Reading structures from {con_file}")
    atoms_list = ase_read(con_file, index=":")
    log.info(f"Loaded {len(atoms_list)} structures.")

    # --- Explicit Saddle Point Loading ---
    sp_data = None
    ira_instance = ira_mod.IRA() if ira_mod else None

    if sp_file and sp_file.exists():
        log.info(f"Loading explicit saddle point from {sp_file}")
        sp_atoms = ase_read(sp_file)
        sp_rmsd_r = calculate_rmsd_from_ref(
            [sp_atoms],
            ira_instance,
            ref_atom=atoms_list[0],
            ira_kmax=ira_kmax,
        )[0]
        sp_rmsd_p = calculate_rmsd_from_ref(
            [sp_atoms],
            ira_instance,
            ref_atom=atoms_list[-1],
            ira_kmax=ira_kmax,
        )[0]
        sp_data = {"atoms": sp_atoms, "r": sp_rmsd_r, "p": sp_rmsd_p}

    # --- Additional Structures Loading ---
    additional_atoms_data = []
    if additional_con:
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

    return atoms_list, additional_atoms_data, sp_data


def _process_single_path_step(
    dat_file,
    con_file,
    y_data_column,
    ira_instance,
    ira_kmax,
    step_idx,
    ref_atoms=None,
    prod_atoms=None,
):
    """Helper to process a single .dat/.con pair into a DataFrame row."""
    path_data = np.loadtxt(dat_file, skiprows=1).T
    z_data_step = path_data[y_data_column]
    atoms_list_step = ase_read(con_file, index=":")
    f_para_step = path_data[3]

    _validate_data_atoms_match(z_data_step, atoms_list_step, dat_file.name)

    # If ref/prod not provided, assume self-contained NEB (0=Ref, -1=Prod)
    # If provided (augmentation mode), use them.
    ref = ref_atoms if ref_atoms is not None else atoms_list_step[0]
    prod = prod_atoms if prod_atoms is not None else atoms_list_step[-1]

    rmsd_r = calculate_rmsd_from_ref(atoms_list_step, ira_instance, ref_atom=ref, ira_kmax=ira_kmax)
    rmsd_p = calculate_rmsd_from_ref(atoms_list_step, ira_instance, ref_atom=prod, ira_kmax=ira_kmax)

    grad_r, grad_p = compute_synthetic_gradients(rmsd_r, rmsd_p, f_para_step)
    return create_landscape_dataframe(rmsd_r, rmsd_p, grad_r, grad_p, z_data_step, int(step_idx))


def aggregate_neb_landscape_data(
    all_dat_paths: list[Path],
    all_con_paths: list[Path],
    y_data_column: int,
    ira_instance,  # Can be None
    cache_file: Path | None = None,
    force_recompute: bool = False,
    ira_kmax: float = 1.8,
    # Caching augmentation
    augment_dat: str | None = None,
    augment_con: str | None = None,
    ref_atoms: Atoms | None = None,
    prod_atoms: Atoms | None = None,
) -> pl.DataFrame:
    """Aggregates data from multiple NEB steps for landscape visualization."""

    # Init IRA if not passed
    if ira_instance is None and ira_mod is not None:
        ira_instance = ira_mod.IRA()

    def validate_landscape_cache(df: pl.DataFrame):
        if "p" not in df.columns:
            raise ValueError("Cache missing 'p' column.")
        if "grad_r" not in df.columns:
            raise ValueError("Cache missing gradient columns (outdated).")

    def compute_landscape_data() -> pl.DataFrame:
        all_dfs = []
        # --- Load Augmentation Data (Inside Cache Block) ---
        if augment_dat and augment_con and ref_atoms and prod_atoms:
            log.info(f"Loading augmentation data for cache: {augment_dat}")
            df_aug = load_augmenting_neb_data(
                augment_dat,
                augment_con,
                ref_atoms=ref_atoms,
                prod_atoms=prod_atoms,
                y_data_column=y_data_column,
                ira_kmax=ira_kmax,
            )
            if not df_aug.is_empty():
                 all_dfs.append(df_aug)

        # Synchronization check
        paths_dat = all_dat_paths
        paths_con = all_con_paths
        if len(paths_dat) != len(paths_con):
            log.warning(f"Mismatch: {len(paths_dat)} dat vs {len(paths_con)} con.")
            min_len = min(len(paths_dat), len(paths_con))
            paths_dat = paths_dat[:min_len]
            paths_con = paths_con[:min_len]

        for step_idx, (dat_file, con_file_step) in enumerate(
            zip(paths_dat, paths_con, strict=True)
        ):
            try:
                df_step = _process_single_path_step(
                    dat_file,
                    con_file_step,
                    y_data_column,
                    ira_instance,
                    ira_kmax,
                    step_idx,
                )
                all_dfs.append(df_step)
            except Exception as e:
                log.warning(f"Failed to process step {step_idx} ({dat_file.name}): {e}")
                continue

        if not all_dfs:
            rerr = "No data could be aggregated."
            raise RuntimeError(rerr)

        return pl.concat(all_dfs)

    return load_or_compute_data(
        cache_file=cache_file,
        force_recompute=force_recompute,
        validation_check=validate_landscape_cache,
        computation_callback=compute_landscape_data,
        context_name="Landscape",
    )


def load_augmenting_neb_data(
    dat_pattern: str,
    con_pattern: str,
    ref_atoms: Atoms,
    prod_atoms: Atoms,
    y_data_column: int,
    ira_kmax: float,
) -> pl.DataFrame:
    """
    Loads external NEB paths (dat+con) to augment the landscape fit.
    Forces projection onto the MAIN path's R/P coordinates.
    """
    from chemparseplot.parse.file_ import find_file_paths

    dat_paths = find_file_paths(dat_pattern)
    con_paths = find_file_paths(con_pattern)

    if not dat_paths or not con_paths:
        log.warning("Augmentation patterns did not match files.")
        return pl.DataFrame()

    # Sync lengths
    min_len = min(len(dat_paths), len(con_paths))
    dat_paths = dat_paths[:min_len]
    con_paths = con_paths[:min_len]

    log.info(f"Augmenting with {min_len} external paths...")

    all_dfs = []
    ira_instance = ira_mod.IRA() if ira_mod else None

    for i, (d, c) in enumerate(zip(dat_paths, con_paths)):
        try:
            # Step -1 indicates 'background/augmented' data
            df = _process_single_path_step(
                d,
                c,
                y_data_column,
                ira_instance,
                ira_kmax,
                -1,
                ref_atoms=ref_atoms,
                prod_atoms=prod_atoms,
            )
            all_dfs.append(df)
        except Exception as e:
            log.warning(f"Failed to load augmentation pair {d.name}: {e}")

    return pl.concat(all_dfs) if all_dfs else pl.DataFrame()


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
    and uses that value as the smoothing factor.
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

    return global_median_step
