# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Parse ORCA NEB calculations using OPI (ORCA Python Interface).

Requires ORCA 6.1+ and the orca-pi package.
Falls back to legacy parsing for older ORCA versions.

Returns data in format compatible with chemparseplot.plot.neb plotting functions.

```{versionadded} 0.2.0
```
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chemparseplot.parse.types import OrcaNebResult

# Lazy import - will be loaded on first use
_opi_output = None


def _get_opi_output():
    """Get OPI Output class, installing if needed."""
    global _opi_output
    if _opi_output is None:
        from rgpycrumbs._aux import ensure_import

        _opi_output = ensure_import("opi.output.core").Output
    return _opi_output


HAS_OPI = True  # Will fail gracefully at runtime if not installed


@dataclass(frozen=True, slots=True)
class _OpiGeometry:
    """Typed geometry payload extracted from one OPI image."""

    coordinates: np.ndarray
    atomic_numbers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _OpiNebImage:
    """Typed per-image NEB record extracted from OPI output."""

    energy_ev: float
    geometry: _OpiGeometry | None = None
    gradient: np.ndarray | None = None


def parse_orca_neb(basename: str, working_dir: Path | None = None) -> OrcaNebResult:
    """Parse ORCA NEB calculation using OPI.

    Returns data compatible with chemparseplot.plot.neb functions:
    - energies: array of energies in eV
    - rmsd_r, rmsd_p: RMSD from reactant/product (if geometries available)
    - grad_r, grad_p: gradients (if forces available)
    - converged: bool
    - n_images: number of images

    Parameters
    ----------
    basename
        ORCA job basename (without extension)
    working_dir
        Working directory containing ORCA output files

    Returns
    -------
    OrcaNebResult
        Structured NEB data compatible with existing plotting helpers

    Example
    -------
    >>> from chemparseplot.parse.orca.neb import parse_orca_neb
    >>> from chemparseplot.plot.neb import plot_energy_path, plot_landscape_path_overlay
    >>> data = parse_orca_neb("job", Path("calc"))
    >>> # Use with existing plotting functions
    """
    Output = _get_opi_output()

    if working_dir is None:
        working_dir = Path.cwd()

    # Parse ORCA output using OPI
    output = Output(basename, working_dir=working_dir)
    output.parse()

    converged = output.terminated_normally()
    n_images = output.num_results_gbw
    images = [_read_opi_neb_image(output, index=i) for i in range(n_images)]
    energies = np.array([image.energy_ev for image in images])

    # Calculate RMSD from reactant and product if geometries available
    rmsd_r = None
    rmsd_p = None
    grad_r = None
    grad_p = None
    forces = [image.gradient for image in images]

    if len(images) >= 2 and all(image.geometry is not None for image in images):
        try:
            atoms_list = [_geometry_to_atoms(image.geometry) for image in images]  # type: ignore[arg-type]

            # Calculate RMSD from reactant and product
            rmsd_r = np.array(
                [_calculate_rmsd(atoms_list[0], atoms) for atoms in atoms_list]
            )
            rmsd_p = np.array(
                [_calculate_rmsd(atoms_list[-1], atoms) for atoms in atoms_list]
            )

            # Calculate synthetic gradients if forces available
            if all(f is not None for f in forces):
                grad_r, grad_p = _compute_synthetic_gradients(
                    rmsd_r, rmsd_p, forces, atoms_list
                )
        except ImportError:
            # ASE not available, skip RMSD calculation
            pass

    # Get barrier heights
    if len(energies) > 1:
        e_reactant = energies[0]
        e_product = energies[-1]
        e_max = energies.max()
        barrier_forward = e_max - e_reactant
        barrier_reverse = e_max - e_product
    else:
        barrier_forward = None
        barrier_reverse = None

    return OrcaNebResult(
        energies=energies,
        rmsd_r=rmsd_r,
        rmsd_p=rmsd_p,
        grad_r=grad_r,
        grad_p=grad_p,
        forces=forces if all(f is not None for f in forces) else None,
        converged=converged,
        n_images=n_images,
        barrier_forward=barrier_forward,
        barrier_reverse=barrier_reverse,
        source="opi",
        orca_version=str(output.orca_version)
        if hasattr(output, "orca_version")
        else "unknown",
    )


def _read_opi_neb_image(output, *, index: int) -> _OpiNebImage:
    """Read one NEB image record from an OPI output object."""

    energy_eh = output.get_final_energy(index=index)
    geometry = None
    gradient = None

    try:
        geom = output.get_geometry(index=index)
        geometry = _OpiGeometry(
            coordinates=np.asarray(geom.coordinates.cartesians),
            atomic_numbers=tuple(atom.atomic_number for atom in geom.atoms),
        )
    except (AttributeError, KeyError):
        geometry = None

    try:
        gradient = np.asarray(output.get_gradient(index=index))
    except (AttributeError, KeyError):
        gradient = None

    return _OpiNebImage(
        energy_ev=energy_eh * 27.211386245988,
        geometry=geometry,
        gradient=gradient,
    )


def _geometry_to_atoms(geometry: _OpiGeometry):
    """Convert a typed OPI geometry payload to ASE Atoms."""

    from ase import Atoms

    return Atoms(numbers=geometry.atomic_numbers, positions=geometry.coordinates)


def _calculate_rmsd(ref: Any, mobile: Any) -> float:
    """Calculate RMSD between two ASE Atoms objects (simple, no alignment)."""
    pos_ref = ref.get_positions()
    pos_mob = mobile.get_positions()
    diff = pos_ref - pos_mob
    return float(np.sqrt((diff * diff).sum() / len(ref)))


def _compute_synthetic_gradients(rmsd_r, rmsd_p, forces, atoms_list):
    """Compute synthetic gradients for landscape plotting."""
    # Simple projection of forces onto RMSD coordinates
    # This is a simplified version - full implementation would use IRA
    grad_r = np.zeros_like(rmsd_r)
    grad_p = np.zeros_like(rmsd_p)

    if forces[0] is not None:
        # Project forces onto RMSD direction
        for i, force in enumerate(forces):
            if force is not None:
                force_norm = np.linalg.norm(force)
                grad_r[i] = -force_norm * (rmsd_r[i] / max(rmsd_r.max(), 1e-10))
                grad_p[i] = -force_norm * (rmsd_p[i] / max(rmsd_p.max(), 1e-10))

    return grad_r, grad_p


def parse_orca_neb_fallback(
    basename: str, working_dir: Path | None = None
) -> OrcaNebResult | None:
    """Parse ORCA NEB using legacy regex parsing (ORCA < 6.1).

    Falls back to parsing .interp files if OPI is not available.

    Returns
    -------
    OrcaNebResult or None
        NEB data if successful, None if parsing fails
    """
    from chemparseplot.parse.orca.neb.interp import extract_interp_points

    if working_dir is None:
        working_dir = Path.cwd()

    interp_file = working_dir / f"{basename}.interp"
    if not interp_file.exists():
        return None

    try:
        text = interp_file.read_text()
        data = extract_interp_points(text)

        if not data:
            return None

        # Extract last iteration
        last_iter = data[-1]
        energies = last_iter.nebpath.energy.magnitude

        return OrcaNebResult(
            energies=np.asarray(energies),
            rmsd_r=None,
            rmsd_p=None,
            grad_r=None,
            grad_p=None,
            forces=None,
            converged=True,
            n_images=len(energies),
            barrier_forward=None,
            barrier_reverse=None,
            source="legacy_interp",
            orca_version="<6.1",
        )
    except Exception:
        return None
