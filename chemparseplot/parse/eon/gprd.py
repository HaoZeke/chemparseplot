import logging
import os
import typing

import ase
import ase.io as aseio
import h5py
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.nwchem import NWChem
from ase.io.trajectory import Trajectory

log = logging.getLogger(__name__)


class HDF5CalculatorDimerMidpoint(Calculator):
    """ASE calculator reading energy/forces from an HDF5 dimer midpoint group.

    ```{versionadded} 0.0.3
    ```
    """

    implemented_properties: typing.ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, from_hdf5, natms, **kwargs):
        Calculator.__init__(self, **kwargs)
        # Reference to the HDF5 group or dataset
        self.from_hdf5 = from_hdf5
        self.natoms = natms

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = ["energy", "forces"]
        Calculator.calculate(self, atoms, properties, system_changes)

        # Access energy and gradients from the referenced HDF5 object
        energy = self.from_hdf5["energy"][0]
        forces = self.from_hdf5["gradients"][: self.natoms * 3].reshape(-1, 3) * -1

        # Store results
        self.results["energy"] = energy
        self.results["forces"] = forces


class HDF5Calculator(Calculator):
    """ASE calculator reading energy/forces from an HDF5 group.

    ```{versionadded} 0.0.3
    ```
    """

    implemented_properties: typing.ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, from_hdf5, **kwargs):
        Calculator.__init__(self, **kwargs)
        # Reference to the HDF5 group or dataset
        self.from_hdf5 = from_hdf5

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = ["energy", "forces"]
        Calculator.calculate(self, atoms, properties, system_changes)

        # Access energy and gradients from the referenced HDF5 object
        energy = self.from_hdf5["energy"][0]
        forces = self.from_hdf5["gradients"][:].reshape(-1, 3) * -1

        # Store results
        self.results["energy"] = energy
        self.results["forces"] = forces


def get_atoms_from_hdf5(
    template_atoms: ase.Atoms,
    hdf5_group: h5py.Group,
) -> ase.Atoms:
    """Create an ASE Atoms object from a template and HDF5 group.

    Contains optimization data.

    ```{versionadded} 0.0.3
    ```

    Args:
        template_atoms (ase.Atoms): The template ASE Atoms
            object (initial structure).
        hdf5_group (h5py.Group): The HDF5 group containing
            'energy', 'gradients', and 'positions' datasets.

    Returns:
        ase.Atoms: An ASE Atoms object with positions, energy,
            and forces from the HDF5 group.
    """

    atoms = template_atoms.copy()
    calculator = HDF5Calculator(from_hdf5=hdf5_group)
    atoms.calc = calculator
    atoms.set_positions(hdf5_group["positions"][()].reshape(-1, 3))
    return atoms


def create_geom_traj_from_hdf5(
    hdf5_file: str,
    output_traj_file: str,
    initial_structure_file: str,
    outer_loop_group_name: str = "outer_loop",
):
    """
    Creates an ASE trajectory file from an HDF5 file containing optimization
    data. This only outputs the geometry steps after the initial rotations.

    ```{versionadded} 0.0.3
    ```

    Generally this is what you want to see for checking the change in geometry
    along the run. There are initial rotations and 2 additional calls (one in
    the beginning and one at the end) which are not accounted for here.

    Args:
        hdf5_file (str): Path to the HDF5 file.
        output_traj_file (str): Path to the output trajectory file (e.g.,
        'gprd_run.traj').
        initial_structure_file (str): Path to the file containing
            the initial structure (e.g., 'pos.con').
        outer_loop_group_name (str, optional): Name of the group
            containing outer loop data. Defaults to
            "outer_loop".
    """

    try:
        f = h5py.File(hdf5_file, "r")
    except FileNotFoundError:
        log.error("HDF5 file '%s' not found.", hdf5_file)
        return
    except Exception as e:
        log.error("Error opening HDF5 file: %s", e)
        return

    outer_loop_keys = [
        str(x) for x in np.sort([int(x) for x in f[outer_loop_group_name].keys()])
    ]
    log.info("Available outer loop keys: %s", outer_loop_keys)

    try:
        init = aseio.read(initial_structure_file)
    except FileNotFoundError:
        log.error(
            "Initial structure file '%s' not found.",
            initial_structure_file,
        )
        f.close()
        return
    except Exception as e:
        log.error("Error reading initial structure file: %s", e)
        f.close()
        return

    traj = Trajectory(output_traj_file, "w")

    for key in outer_loop_keys:
        try:
            # Create atoms object directly here
            atoms = init.copy()
            calculator = HDF5Calculator(from_hdf5=f[outer_loop_group_name][key])
            atoms.calc = calculator
            atoms.set_positions(
                f[outer_loop_group_name][key]["positions"][:].reshape(-1, 3)
            )

            # Trigger calculation of energy and forces
            atoms.get_potential_energy()

            traj.write(atoms)
        except KeyError as e:
            log.warning(
                "Skipping key %s due to missing data: %s",
                key,
                e,
            )
        except Exception as e:
            log.error("Error processing key %s: %s", key, e)

    f.close()
    traj.close()
    log.info(
        "Trajectory file '%s' created successfully.",
        output_traj_file,
    )


def create_nwchem_trajectory(
    template_atoms: ase.Atoms,
    hdf5_file: str,
    output_traj_file: str,
    mult=1,
    outer_loop_group_name: str = "outer_loop",
):
    """Create an ASE trajectory with NWChem energies and forces.

    Positions are taken from an HDF5 file.

    ```{versionadded} 0.0.3
    ```

    Args:
        template_atoms (ase.Atoms): The template ASE Atoms
            object (initial structure).
        hdf5_file (str): Path to the HDF5 file containing
            positions.
        output_traj_file (str): Path to the output trajectory
            file (e.g., 'nwchem_run.traj').
        outer_loop_group_name (str, optional): Name of the
            group containing outer loop data.
            Defaults to "outer_loop".
    """

    try:
        f = h5py.File(hdf5_file, "r")
    except FileNotFoundError:
        log.error("HDF5 file '%s' not found.", hdf5_file)
        return
    except Exception as e:
        log.error("Error opening HDF5 file: %s", e)
        return

    outer_loop_keys = [
        str(x) for x in np.sort([int(x) for x in f[outer_loop_group_name].keys()])
    ]
    log.info("Available outer loop keys: %s", outer_loop_keys)

    traj = Trajectory(output_traj_file, "w")

    nwchem_path = os.environ["NWCHEM_COMMAND"]
    memory = "2 gb"
    nwchem_kwargs = {
        "command": f"{nwchem_path} PREFIX.nwi > PREFIX.nwo",
        "memory": memory,
        "scf": {
            "nopen": mult - 1,
            "thresh": 1e-8,
            "maxiter": 200,
        },
        "basis": "3-21G",
        "task": "gradient",
    }
    if mult == 2:
        nwchem_kwargs["scf"]["uhf"] = None  # switch to unrestricted calculation

    for key in outer_loop_keys:
        try:
            # Create a copy of the template atoms
            atoms = template_atoms.copy()

            # Set positions from the HDF5 file
            atoms.positions = f[outer_loop_group_name][key]["positions"][()].reshape(
                -1, 3
            )

            # Assign calculator and calculate energy and forces
            atoms.calc = NWChem(**nwchem_kwargs)
            log.info("Calculating for %s", key)
            atoms.get_potential_energy()

            # Write to trajectory
            traj.write(atoms)

        except KeyError as e:
            log.warning(
                "Skipping key %s due to missing data: %s",
                key,
                e,
            )

    f.close()
    traj.close()
    log.info(
        "NWChem trajectory file '%s' created successfully.",
        output_traj_file,
    )


def create_full_traj_from_hdf5(
    hdf5_file: str,
    output_traj_file: str,
    initial_structure_file: str,
    outer_loop_group_name: str = "outer_loop",
    inner_loop_group_name: str = "initial_rotations",
):
    """Create an ASE trajectory from an HDF5 optimization file.

    Includes **estimated points** for the initial rotations
    and the endpoints.

    ```{versionadded} 0.0.3
    ```

    These are correct (correspond to the actual counts) but
    is slightly convoluted, since the HDF5 contains both the
    midpoint and the "forward dimer". Instead, the length of
    the inner rotations keys is the number of (0 energy)
    points added to the trajectory. This again makes
    intuitive sense, since we have the Elvl cutoff in the
    GPRD as well.

    Args:
        hdf5_file (str): Path to the HDF5 file.
        output_traj_file (str): Path to the output trajectory
            file (e.g., 'gprd_run.traj').
        initial_structure_file (str): Path to the file
            containing the initial structure
            (e.g., 'pos.con').
        outer_loop_group_name (str, optional): Name of the
            group containing outer loop data.
            Defaults to "outer_loop".
    """

    try:
        f = h5py.File(hdf5_file, "r")
    except FileNotFoundError:
        log.error("HDF5 file '%s' not found.", hdf5_file)
        return
    except Exception as e:
        log.error("Error opening HDF5 file: %s", e)
        return

    outer_loop_keys = [
        str(x) for x in np.sort([int(x) for x in f[outer_loop_group_name].keys()])
    ]

    inner_loop_keys = [
        str(x) for x in np.sort([int(x) for x in f[inner_loop_group_name].keys()])
    ]

    log.info("Available inner loop keys: %s", inner_loop_keys)
    log.info("Available outer loop keys: %s", outer_loop_keys)

    try:
        init = aseio.read(initial_structure_file)
    except FileNotFoundError:
        log.error(
            "Initial structure file '%s' not found.",
            initial_structure_file,
        )
        f.close()
        return
    except Exception as e:
        log.error("Error reading initial structure file: %s", e)
        f.close()
        return

    traj = Trajectory(output_traj_file, "w")

    # Generate the initial rotation stuff here
    # Basically the number of keys, + 1
    for idx, key in enumerate(inner_loop_keys):
        atoms = init.copy()
        iloop_dat = f[inner_loop_group_name][key]
        calculator = HDF5CalculatorDimerMidpoint(from_hdf5=iloop_dat, natms=len(atoms))
        atoms.calc = calculator
        atoms.set_positions(iloop_dat["positions"][: len(atoms) * 3].reshape(-1, 3))

        # Trigger calculation of energy and forces
        atoms.get_potential_energy()
        traj.write(atoms)

        # Do it again for the first step
        if idx == 0:
            traj.write(atoms)

    for idx, key in enumerate(outer_loop_keys):
        try:
            # Create atoms object directly here
            atoms = init.copy()
            calculator = HDF5Calculator(from_hdf5=f[outer_loop_group_name][key])
            atoms.calc = calculator
            atoms.set_positions(
                f[outer_loop_group_name][key]["positions"][:].reshape(-1, 3)
            )

            # Trigger calculation of energy and forces
            atoms.get_potential_energy()

            traj.write(atoms)
            # Now for the final calculation done to finish the run
            if idx == len(outer_loop_keys) - 1:
                # Just write it one more time, same thing
                traj.write(atoms)

        except KeyError as e:
            log.warning(
                "Skipping key %s due to missing data: %s",
                key,
                e,
            )
        except Exception as e:
            log.error("Error processing key %s: %s", key, e)

    f.close()
    traj.close()
    log.info(
        "Trajectory file '%s' created successfully.",
        output_traj_file,
    )
