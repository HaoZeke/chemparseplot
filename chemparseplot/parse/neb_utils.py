"""Shared NEB analysis utilities.

```{versionadded} 1.2.0
```

Functions common to all NEB trajectory sources (eOn, extxyz, etc.):
RMSD landscape coordinate calculation, synthetic 2D gradient projection,
and landscape DataFrame construction.
"""

import logging

import numpy as np
import polars as pl
from ase import Atoms

log = logging.getLogger(__name__)


def calculate_landscape_coords(
    atoms_list: list[Atoms], ira_instance, ira_kmax: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate 2D landscape coordinates (RMSD-R, RMSD-P) for a path.

    ```{versionadded} 1.2.0
    ```

    Uses the first frame as reactant reference and the last as product.

    :param atoms_list: List of ASE Atoms objects representing the path.
    :param ira_instance: An instantiated IRA object (or None).
    :param ira_kmax: kmax factor for IRA.
    :return: A tuple of (rmsd_r, rmsd_p) arrays.
    """
    from rgpycrumbs.geom.api.alignment import calculate_rmsd_from_ref

    log.info("Calculating landscape coordinates (RMSD-R, RMSD-P)...")
    rmsd_r = calculate_rmsd_from_ref(
        atoms_list, ira_instance, ref_atom=atoms_list[0], ira_kmax=ira_kmax
    )
    rmsd_p = calculate_rmsd_from_ref(
        atoms_list, ira_instance, ref_atom=atoms_list[-1], ira_kmax=ira_kmax
    )
    return rmsd_r, rmsd_p


def compute_synthetic_gradients(
    rmsd_r: np.ndarray, rmsd_p: np.ndarray, f_para: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project parallel force onto the 2D RMSD coordinate system.

    ```{versionadded} 1.2.0
    ```

    Computes synthetic gradients by projecting the NEB parallel force
    component along the tangent direction in (RMSD-R, RMSD-P) space.

    :param rmsd_r: RMSD from reactant for each image.
    :param rmsd_p: RMSD from product for each image.
    :param f_para: Force component parallel to the path for each image.
    :return: (grad_r, grad_p) arrays.
    """
    dr = np.gradient(rmsd_r)
    dp = np.gradient(rmsd_p)
    norm_ds = np.sqrt(dr**2 + dp**2)
    norm_ds[norm_ds == 0] = 1.0
    tr = dr / norm_ds
    tp = dp / norm_ds
    grad_r = -f_para * tr
    grad_p = -f_para * tp
    return grad_r, grad_p


def create_landscape_dataframe(
    rmsd_r: np.ndarray,
    rmsd_p: np.ndarray,
    grad_r: np.ndarray,
    grad_p: np.ndarray,
    z: np.ndarray,
    step: int,
) -> pl.DataFrame:
    """Build a landscape DataFrame with the standard schema.

    ```{versionadded} 1.2.0
    ```

    :param rmsd_r: RMSD from reactant.
    :param rmsd_p: RMSD from product.
    :param grad_r: Synthetic gradient in R direction.
    :param grad_p: Synthetic gradient in P direction.
    :param z: Energy (or eigenvalue) data.
    :param step: Step index for this path segment.
    :return: Polars DataFrame with columns [r, p, grad_r, grad_p, z, step].
    """
    return pl.DataFrame(
        {
            "r": rmsd_r,
            "p": rmsd_p,
            "grad_r": grad_r,
            "grad_p": grad_p,
            "z": z,
            "step": step,
        }
    )
