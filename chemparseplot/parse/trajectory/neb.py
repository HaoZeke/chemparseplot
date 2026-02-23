"""NEB trajectory parser for ASE-readable formats (extxyz, .traj, etc.).

```{versionadded} 1.2.0
```

Loads trajectory files via ASE, computes NEB profile data (cumulative
distance, improved Henkelman-Jonsson tangent forces), and converts into
the formats expected by chemparseplot's plotting functions.
"""

import logging

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

from chemparseplot.parse.neb_utils import (
    calculate_landscape_coords,
    compute_synthetic_gradients,
    create_landscape_dataframe,
)

log = logging.getLogger(__name__)


def _get_energy(atoms: Atoms) -> float:
    """Extract energy from an Atoms object, checking calc then info."""
    if atoms.calc is not None:
        try:
            return atoms.get_potential_energy()
        except Exception:
            pass
    return atoms.info.get("energy", 0.0)


def load_trajectory(traj_file: str) -> list[Atoms]:
    """Load an extxyz trajectory file, returning all frames.

    ```{versionadded} 1.2.0
    ```
    """
    atoms_list = ase_read(traj_file, index=":")
    if isinstance(atoms_list, Atoms):
        atoms_list = [atoms_list]
    log.info(f"Loaded {len(atoms_list)} frames from {traj_file}")
    return atoms_list


def compute_cumulative_distance(atoms_list: list[Atoms]) -> np.ndarray:
    """Compute cumulative Euclidean path length along the NEB band.

    ```{versionadded} 1.2.0
    ```

    Returns a 1D array of length n_images with the cumulative distance
    from the first image.
    """
    n = len(atoms_list)
    dists = np.zeros(n)
    for i in range(1, n):
        delta = atoms_list[i].positions - atoms_list[i - 1].positions
        dists[i] = dists[i - 1] + np.linalg.norm(delta)
    return dists


def compute_tangent_force(atoms_list: list[Atoms], energies: np.ndarray) -> np.ndarray:
    """Compute force component parallel to the path using the improved
    Henkelman-Jonsson tangent definition.

    ```{versionadded} 1.2.0
    ```

    For interior images, the tangent is energy-weighted (uphill neighbor
    gets more weight). Forces are read from each frame's arrays.

    Returns a 1D array of f_parallel values (one per image).
    """
    n = len(atoms_list)
    f_para = np.zeros(n)

    for i in range(n):
        forces_i = (
            atoms_list[i].get_forces()
            if atoms_list[i].calc
            else np.zeros_like(atoms_list[i].positions)
        )

        if i == 0 or i == n - 1:
            # Endpoint: f_parallel = 0 by convention
            f_para[i] = 0.0
            continue

        # Tangent via improved H&J method
        pos_prev = atoms_list[i - 1].positions.ravel()
        pos_curr = atoms_list[i].positions.ravel()
        pos_next = atoms_list[i + 1].positions.ravel()

        tau_plus = pos_next - pos_curr
        tau_minus = pos_curr - pos_prev

        e_prev = energies[i - 1]
        e_curr = energies[i]
        e_next = energies[i + 1]

        de_plus = e_next - e_curr
        de_minus = e_curr - e_prev

        if e_next > e_curr > e_prev:
            tau = tau_plus
        elif e_next < e_curr < e_prev:
            tau = tau_minus
        else:
            # At extremum: energy-weighted bisection
            de_max = max(abs(de_plus), abs(de_minus))
            de_min = min(abs(de_plus), abs(de_minus))
            if e_next > e_prev:
                tau = tau_plus * de_max + tau_minus * de_min
            else:
                tau = tau_plus * de_min + tau_minus * de_max

        tau_norm = np.linalg.norm(tau)
        if tau_norm > 0:
            tau_hat = tau / tau_norm
        else:
            tau_hat = tau

        forces_flat = forces_i.ravel()
        f_para[i] = np.dot(forces_flat, tau_hat)

    return f_para


def extract_profile_data(
    atoms_list: list[Atoms],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract profile data from trajectory frames.

    ```{versionadded} 1.2.0
    ```

    Returns (index, distance, energy, f_parallel) arrays.
    """
    n = len(atoms_list)
    indices = np.arange(n, dtype=float)
    distances = compute_cumulative_distance(atoms_list)
    energies = np.array([_get_energy(a) for a in atoms_list])
    f_para = compute_tangent_force(atoms_list, energies)
    return indices, distances, energies, f_para


def trajectory_to_profile_dat(atoms_list: list[Atoms]) -> np.ndarray:
    """Convert trajectory to a (5, N) array matching eOn .dat layout.

    ```{versionadded} 1.2.0
    ```

    Columns: [index, reaction_coordinate, energy, f_parallel, zeros]
    This output can be fed directly into plot_energy_path as:
        plot_energy_path(ax, data[1], data[2], data[3], ...)
    """
    indices, distances, energies, f_para = extract_profile_data(atoms_list)
    zeros = np.zeros(len(atoms_list))
    return np.array([indices, distances, energies, f_para, zeros])


def trajectory_to_landscape_df(
    atoms_list: list[Atoms],
    ira_kmax: float = 1.8,
    step: int = 0,
):
    """Convert trajectory to a polars DataFrame for plot_landscape_surface.

    ```{versionadded} 1.2.0
    ```

    Columns: r, p, grad_r, grad_p, z, step

    Uses RMSD from reactant (first frame) and product (last frame) as
    the 2D coordinate system, with synthetic gradients projected from
    the parallel force component.
    """
    try:
        from rgpycrumbs._aux import _import_from_parent_env

        ira_mod = _import_from_parent_env("ira_mod")
        ira_instance = ira_mod.IRA()
    except (ImportError, AttributeError):
        ira_instance = None

    rmsd_r, rmsd_p = calculate_landscape_coords(atoms_list, ira_instance, ira_kmax)

    energies = np.array([_get_energy(a) for a in atoms_list])
    f_para = compute_tangent_force(atoms_list, energies)

    grad_r, grad_p = compute_synthetic_gradients(rmsd_r, rmsd_p, f_para)
    return create_landscape_dataframe(rmsd_r, rmsd_p, grad_r, grad_p, energies, step)
