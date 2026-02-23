# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""HDF5 NEB reader for ChemGP output.

Reads ``neb_result.h5`` (final converged path) and ``neb_history.h5``
(per-step optimization history) produced by ChemGP's Julia NEB optimizer.

These files contain pre-computed ``f_para`` and ``rxn_coord``, so profile
data can be read directly without recomputing the Henkelman-Jonsson tangent.
"""

import logging

import h5py
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from chemparseplot.parse.neb_utils import (
    calculate_landscape_coords,
    compute_synthetic_gradients,
    create_landscape_dataframe,
)

log = logging.getLogger(__name__)


def _read_path_group(grp: h5py.Group) -> dict:
    """Extract arrays from an HDF5 path group.

    :param grp: An h5py Group containing ``images``, ``energies``,
        ``gradients``, ``f_para``, and ``rxn_coord`` datasets.
    :return: Dictionary with numpy arrays for each field.
    """
    return {
        "images": np.asarray(grp["images"]),
        "energies": np.asarray(grp["energies"]),
        "gradients": np.asarray(grp["gradients"]),
        "f_para": np.asarray(grp["f_para"]),
        "rxn_coord": np.asarray(grp["rxn_coord"]),
    }


def _reconstruct_atoms(
    images: np.ndarray,
    atomic_numbers: np.ndarray | None,
    cell: np.ndarray | None,
    gradients: np.ndarray,
    energies: np.ndarray,
) -> list[Atoms]:
    """Rebuild ASE Atoms from flattened positions and metadata.

    Each image's positions are ``(ndof,)`` where ``ndof = 3 * n_atoms``.
    A SinglePointCalculator is attached with energy and forces (``-gradients``).

    :param images: Array of shape ``(n_images, ndof)``.
    :param atomic_numbers: Optional array of shape ``(n_atoms,)``.
    :param cell: Optional array of shape ``(9,)`` (flattened 3x3 cell).
    :param gradients: Array of shape ``(n_images, ndof)``.
    :param energies: Array of shape ``(n_images,)``.
    :return: List of ASE Atoms objects with calculators attached.
    """
    n_images = images.shape[0]
    ndof = images.shape[1]
    n_atoms = ndof // 3

    if atomic_numbers is None:
        atomic_numbers = np.ones(n_atoms, dtype=int)  # default to H

    cell_matrix = None
    pbc = False
    if cell is not None:
        cell_matrix = cell.reshape(3, 3)
        pbc = True

    atoms_list = []
    for i in range(n_images):
        positions = images[i].reshape(n_atoms, 3)
        forces = -gradients[i].reshape(n_atoms, 3)
        energy = float(energies[i])

        atoms = Atoms(
            numbers=atomic_numbers,
            positions=positions,
            cell=cell_matrix,
            pbc=pbc,
        )
        calc = SinglePointCalculator(
            atoms, energy=energy, forces=forces
        )
        atoms.calc = calc
        atoms_list.append(atoms)

    return atoms_list


def load_neb_result(h5_file: str) -> dict:
    """Read a ChemGP ``neb_result.h5`` file.

    :param h5_file: Path to the HDF5 result file.
    :return: Dictionary with keys ``path`` (from :func:`_read_path_group`),
        ``convergence`` (dict of arrays), and ``metadata`` (dict of scalars
        and optional arrays).
    """
    with h5py.File(h5_file, "r") as f:
        path_data = _read_path_group(f["path"])

        convergence = {}
        if "convergence" in f:
            conv_grp = f["convergence"]
            for key in conv_grp:
                convergence[key] = np.asarray(conv_grp[key])

        metadata = {}
        if "metadata" in f:
            meta_grp = f["metadata"]
            for key in meta_grp:
                val = meta_grp[key]
                if val.shape == ():
                    metadata[key] = val[()]
                else:
                    metadata[key] = np.asarray(val)

    log.info(
        "Loaded NEB result: %d images from %s",
        path_data["images"].shape[0],
        h5_file,
    )
    return {
        "path": path_data,
        "convergence": convergence,
        "metadata": metadata,
    }


def load_neb_history(h5_file: str) -> list[dict]:
    """Read a ChemGP ``neb_history.h5`` file.

    :param h5_file: Path to the HDF5 history file.
    :return: List of dicts (one per optimization step), sorted by step
        number. Each dict has the same structure as :func:`_read_path_group`.
    """
    steps = []
    with h5py.File(h5_file, "r") as f:
        steps_grp = f["steps"]
        step_keys = sorted(steps_grp.keys(), key=int)
        for key in step_keys:
            steps.append(_read_path_group(steps_grp[key]))

    log.info(
        "Loaded NEB history: %d steps from %s", len(steps), h5_file
    )
    return steps


def result_to_profile_dat(h5_file: str) -> np.ndarray:
    """Convert a result HDF5 to a (5, N) profile array.

    Layout matches eOn ``.dat`` format:
    ``[index, rxn_coord, energy, f_para, zeros]``.

    Reads directly from pre-computed HDF5 fields -- no tangent recomputation.

    :param h5_file: Path to ``neb_result.h5``.
    :return: Array of shape ``(5, n_images)``.
    """
    result = load_neb_result(h5_file)
    path = result["path"]
    n = len(path["energies"])
    return np.array([
        np.arange(n, dtype=float),
        path["rxn_coord"],
        path["energies"],
        path["f_para"],
        np.zeros(n),
    ])


def result_to_atoms_list(h5_file: str) -> list[Atoms]:
    """Reconstruct ASE Atoms from a result HDF5 file.

    :param h5_file: Path to ``neb_result.h5``.
    :return: List of Atoms with SinglePointCalculators attached.
    """
    result = load_neb_result(h5_file)
    path = result["path"]
    meta = result["metadata"]
    return _reconstruct_atoms(
        path["images"],
        meta.get("atomic_numbers"),
        meta.get("cell"),
        path["gradients"],
        path["energies"],
    )


def history_to_profile_dats(h5_file: str) -> list[np.ndarray]:
    """Convert a history HDF5 to a list of (5, N) profile arrays.

    One array per optimization step, sorted by step number.

    :param h5_file: Path to ``neb_history.h5``.
    :return: List of arrays, each of shape ``(5, n_images)``.
    """
    steps = load_neb_history(h5_file)
    dats = []
    for step_data in steps:
        n = len(step_data["energies"])
        dat = np.array([
            np.arange(n, dtype=float),
            step_data["rxn_coord"],
            step_data["energies"],
            step_data["f_para"],
            np.zeros(n),
        ])
        dats.append(dat)
    return dats


def history_to_landscape_df(
    h5_file: str, ira_kmax: float = 1.8
):
    """Convert a history HDF5 to a landscape DataFrame.

    Reconstructs Atoms per step for RMSD coordinates, uses pre-computed
    ``f_para`` for synthetic gradients.

    :param h5_file: Path to ``neb_history.h5``.
    :param ira_kmax: kmax factor for IRA alignment.
    :return: Polars DataFrame with columns
        ``[r, p, grad_r, grad_p, z, step]``.
    """
    import polars as pl

    try:
        from rgpycrumbs._aux import _import_from_parent_env

        ira_mod = _import_from_parent_env("ira_mod")
        ira_instance = ira_mod.IRA()
    except (ImportError, AttributeError):
        ira_instance = None

    with h5py.File(h5_file, "r") as f:
        meta = f.get("metadata", {})
        atomic_numbers = (
            np.asarray(meta["atomic_numbers"])
            if "atomic_numbers" in meta
            else None
        )
        cell = (
            np.asarray(meta["cell"]) if "cell" in meta else None
        )

    steps = load_neb_history(h5_file)
    frames = []

    for step_idx, step_data in enumerate(steps):
        atoms_list = _reconstruct_atoms(
            step_data["images"],
            atomic_numbers,
            cell,
            step_data["gradients"],
            step_data["energies"],
        )

        rmsd_r, rmsd_p = calculate_landscape_coords(
            atoms_list, ira_instance, ira_kmax
        )
        grad_r, grad_p = compute_synthetic_gradients(
            rmsd_r, rmsd_p, step_data["f_para"]
        )
        df = create_landscape_dataframe(
            rmsd_r,
            rmsd_p,
            grad_r,
            grad_p,
            step_data["energies"],
            step_idx,
        )
        frames.append(df)

    return pl.concat(frames)
