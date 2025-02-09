import ase
from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
import ase.io as aseio
import numpy as np
import h5py


class HDF5Calculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, from_hdf5, **kwargs):
        Calculator.__init__(self, **kwargs)
        # Reference to the HDF5 group or dataset
        self.from_hdf5 = from_hdf5

    def calculate(
        self, atoms=None, properties=["energy", "forces"], system_changes=all_changes
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        # Access energy and gradients directly from the referenced HDF5 object
        energy = self.from_hdf5["energy"][0]
        forces = self.from_hdf5["gradients"][:].reshape(-1, 3) * -1

        # Store results
        self.results["energy"] = energy
        self.results["forces"] = forces


def get_atoms_from_hdf5(template_atoms: ase.Atoms, hdf5_group: h5py.Group) -> ase.Atoms:
    """
    Creates an ASE Atoms object from a template and an HDF5 group containing optimization data.

    Args:
        template_atoms (ase.Atoms): The template ASE Atoms object (initial structure).
        hdf5_group (h5py.Group): The HDF5 group containing 'energy', 'gradients', and 'positions' datasets.

    Returns:
        ase.Atoms: An ASE Atoms object with positions, energy, and forces from the HDF5 group.
    """

    atoms = template_atoms.copy()
    calculator = HDF5Calculator(from_hdf5=hdf5_group)
    atoms.calc = calculator
    atoms.set_positions(hdf5_group["positions"][()].reshape(-1, 3))
    return atoms


def create_trajectory_from_hdf5(
    hdf5_file: str,
    output_traj_file: str,
    initial_structure_file: str,
    outer_loop_group_name: str = "outer_loop",
):
    """
    Creates an ASE trajectory file from an HDF5 file containing optimization data.

    Args:
        hdf5_file (str): Path to the HDF5 file.
        output_traj_file (str): Path to the output trajectory file (e.g., 'gprd_run.traj').
        initial_structure_file (str): Path to the file containing the initial structure (e.g., 'pos.con').
        outer_loop_group_name (str, optional): Name of the group containing outer loop data. Defaults to "outer_loop".
    """

    try:
        f = h5py.File(hdf5_file, "r")
    except FileNotFoundError:
        print(f"Error: HDF5 file '{hdf5_file}' not found.")
        return
    except Exception as e:
        print(f"An error occurred while opening HDF5 file: {e}")
        return

    outer_loop_keys = [
        str(x) for x in np.sort([int(x) for x in f[outer_loop_group_name].keys()])
    ]
    print(f"Available outer loop keys: {outer_loop_keys}")

    try:
        init = aseio.read(initial_structure_file)
    except FileNotFoundError:
        print(f"Error: Initial structure file '{initial_structure_file}' not found.")
        f.close()
        return
    except Exception as e:
        print(f"An error occurred while reading initial structure file: {e}")
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
            print(f"Skipping key {key} due to missing data: {e}")
        except Exception as e:
            print(f"An error occurred while processing key {key}: {e}")

    f.close()
    traj.close()
    print(f"Trajectory file '{output_traj_file}' created successfully.")


def create_nwchem_trajectory(
    template_atoms: ase.Atoms,
    hdf5_file: str,
    output_traj_file: str,
    mult=1,
    outer_loop_group_name: str = "outer_loop",
):
    """
    Creates an ASE trajectory file with NWChem energy and forces calculated for positions
    taken from an HDF5 file.

    Args:
        template_atoms (ase.Atoms): The template ASE Atoms object (initial structure).
        hdf5_file (str): Path to the HDF5 file containing positions.
        output_traj_file (str): Path to the output trajectory file (e.g., 'nwchem_run.traj').
        outer_loop_group_name (str, optional): Name of the group containing outer loop data. Defaults to "outer_loop".
    """

    try:
        f = h5py.File(hdf5_file, "r")
    except FileNotFoundError:
        print(f"Error: HDF5 file '{hdf5_file}' not found.")
        return
    except Exception as e:
        print(f"An error occurred while opening HDF5 file: {e}")
        return

    outer_loop_keys = [
        str(x) for x in np.sort([int(x) for x in f[outer_loop_group_name].keys()])
    ]
    print(f"Available outer loop keys: {outer_loop_keys}")

    traj = Trajectory(output_traj_file, "w")

    nwchem_path = os.environ["NWCHEM_COMMAND"]
    memory = "2 gb"
    nwchem_kwargs = dict(
        command=f"{nwchem_path} PREFIX.nwi > PREFIX.nwo",
        memory=memory,
        scf=dict(
            nopen=mult - 1,
            thresh=1e-8,
            maxiter=200,
        ),
        basis="3-21G",
        task="gradient",
    )
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
            print(f"Calculating for {key}")
            atoms.get_potential_energy()

            # Write to trajectory
            traj.write(atoms)

        except KeyError as e:
            print(f"Skipping key {key} due to missing data: {e}")

    f.close()
    traj.close()
    print(f"NWChem trajectory file '{output_traj_file}' created successfully.")
