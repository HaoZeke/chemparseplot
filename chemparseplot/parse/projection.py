"""Reaction valley (s, d) projection utilities.

Extracts the 2D RMSD-plane rotation into reusable functions.
The projection maps ``(rmsd_a, rmsd_b)`` coordinates into
progress ``s`` (along the path) and deviation ``d`` (perpendicular).

For NEB paths, reference A is the reactant and B is the product.
For single-ended methods, A is the initial structure and B is the
final (saddle or minimum).

Implements the method from :cite:`goswami2026valley`.

.. versionadded:: 1.3.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ProjectionBasis:
    """Orthonormal basis for the (s, d) reaction valley projection.

    Attributes
    ----------
    a_start, b_start
        RMSD values of the first point (origin of the rotated frame).
    u_a, u_b
        Unit vector along the path direction in (a, b) space.
    v_a, v_b
        Unit vector perpendicular to the path (``v = rotate(u, +90deg)``).
    path_norm
        Euclidean length of the path vector in (a, b) space.
    """

    a_start: float
    b_start: float
    u_a: float
    u_b: float
    v_a: float
    v_b: float
    path_norm: float


def compute_projection_basis(
    rmsd_a: np.ndarray,
    rmsd_b: np.ndarray,
) -> ProjectionBasis:
    """Compute the projection basis from first/last points of the arrays.

    Parameters
    ----------
    rmsd_a, rmsd_b
        RMSD distance arrays (to reference A and B respectively).
        The first element defines the origin; the last defines the
        path direction.

    Returns
    -------
    ProjectionBasis
        Frozen dataclass with the orthonormal basis vectors.

    Raises
    ------
    ValueError
        If the path has zero length (first and last points coincide
        in RMSD space).
    """
    a_start, b_start = float(rmsd_a[0]), float(rmsd_b[0])
    a_end, b_end = float(rmsd_a[-1]), float(rmsd_b[-1])

    vec_a, vec_b = a_end - a_start, b_end - b_start
    path_norm = np.hypot(vec_a, vec_b)

    if path_norm < 1e-12:
        msg = (
            "Path has zero length in RMSD space "
            f"(start=({a_start:.6f}, {b_start:.6f}), "
            f"end=({a_end:.6f}, {b_end:.6f}))"
        )
        raise ValueError(msg)

    u_a = vec_a / path_norm
    u_b = vec_b / path_norm

    return ProjectionBasis(
        a_start=a_start,
        b_start=b_start,
        u_a=u_a,
        u_b=u_b,
        v_a=-u_b,
        v_b=u_a,
        path_norm=path_norm,
    )


def project_to_sd(
    rmsd_a: np.ndarray,
    rmsd_b: np.ndarray,
    basis: ProjectionBasis,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (rmsd_a, rmsd_b) into (s, d) reaction valley coordinates.

    Parameters
    ----------
    rmsd_a, rmsd_b
        RMSD arrays to project.
    basis
        Pre-computed projection basis.

    Returns
    -------
    s, d
        Progress and deviation arrays.
    """
    da = rmsd_a - basis.a_start
    db = rmsd_b - basis.b_start
    s = da * basis.u_a + db * basis.u_b
    d = da * basis.v_a + db * basis.v_b
    return s, d


def inverse_sd_to_ab(
    s: np.ndarray,
    d: np.ndarray,
    basis: ProjectionBasis,
) -> tuple[np.ndarray, np.ndarray]:
    """Map (s, d) grid coordinates back to (a, b) RMSD space.

    Used for evaluating the RBF surface on a projected grid.

    Parameters
    ----------
    s, d
        Progress and deviation arrays (can be meshgrid raveled).
    basis
        Pre-computed projection basis.

    Returns
    -------
    rmsd_a, rmsd_b
        Coordinates in the original RMSD plane.
    """
    rmsd_a = basis.a_start + s * basis.u_a + d * basis.v_a
    rmsd_b = basis.b_start + s * basis.u_b + d * basis.v_b
    return rmsd_a, rmsd_b
