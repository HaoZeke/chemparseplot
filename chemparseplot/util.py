import logging

import numpy as np
from ase.io import read

log = logging.getLogger(__name__)

# --- Configuration ---
POSCON_FILENAME = "pos.con"
_EXPECTED_COORD_COLS = 3

# --- Input Target Coordinates ---
# e.g. from an OVITO selection
target_coords_text = """
19.7267 23.4973 21.4053
21.6919 21.7746 21.7746
19.7274 21.4053 23.4968
21.6918 21.7752 25.2201
21.6915 25.2205 21.7761
18.0999 23.4978 23.4978
19.7265 25.5905 23.4985
21.6915 25.2206 25.2206
22.7758 23.4966 23.4983
20.6077 23.4982 23.4977
19.7268 23.4982 25.5902
23.6566 21.4042 23.4962
23.6566 23.4979 21.4058
23.6561 25.5897 23.4984
23.6566 23.4984 25.5906
25.2834 23.4976 23.4978
"""


# --- Function to parse target coordinates ---
def parse_target_coords(text_block):
    """Parses the multiline string of target coordinates.

    ```{versionadded} 0.0.3
    ```
    """
    coords = []
    lines = text_block.strip().split("\n")
    for i, line in enumerate(lines):
        try:
            parts = line.strip().split()
            if len(parts) == _EXPECTED_COORD_COLS:
                coords.append([float(p) for p in parts])
            elif parts:
                log.warning(
                    "Skipping target coordinate line %d due to incorrect format: %s",
                    i + 1,
                    line.strip(),
                )
        except ValueError:
            log.warning(
                "Skipping target coordinate line %d due to non-numeric value: %s",
                i + 1,
                line.strip(),
            )
    return np.array(coords)


# --- Toy Script using ASE ---

atoms = None
try:
    # 1. Read the structure using ASE (handles format detection)
    print(f"Reading {POSCON_FILENAME} using ASE...")
    atoms = read(POSCON_FILENAME)
    print(f"Successfully read {len(atoms)} atoms.")
    atom_coords_from_poscon = atoms.get_positions()  # Get positions as NumPy array

except FileNotFoundError:
    print(f"Error: File not found at {POSCON_FILENAME}")
except Exception as e:
    print(f"Error reading {POSCON_FILENAME} with ASE: {e}")

# 2. Parse the target coordinates
print("Parsing target coordinates...")
target_coords = parse_target_coords(target_coords_text)

# Proceed only if both steps were successful
if atoms is not None and target_coords.size > 0:
    print(f"\nFound {len(atom_coords_from_poscon)} atoms in {POSCON_FILENAME}.")
    print(f"Found {len(target_coords)} target coordinates to match.")

    results = []
    atom_symbols = atoms.get_chemical_symbols()  # Get symbols for richer output

    # 3. For each target coordinate, find the closest atom
    print("\nMatching target coordinates to closest atoms...")
    for i, target_pos in enumerate(target_coords):
        # Calculate distances (same numpy method)
        distances = np.linalg.norm(atom_coords_from_poscon - target_pos, axis=1)

        # Find the index of the minimum distance
        closest_atom_index = np.argmin(distances)  # 0-based index from ASE Atoms object

        # Atom ID in eOn is 0 based so no change here
        # XXX: LAMMPS uses 1 based indexing..
        closest_atom_id = closest_atom_index
        min_dist = distances[closest_atom_index]

        results.append(
            {
                "target_index": i + 1,
                "target_pos": target_pos,
                "closest_atom_id": closest_atom_id,
                "closest_atom_symbol": atom_symbols[closest_atom_index],
                "closest_atom_pos": atom_coords_from_poscon[closest_atom_index],
                "distance": min_dist,
            }
        )

    # 4. Print the results
    print("\n--- Results ---")
    for result in results:
        print(
            f"Target #{result['target_index']} ({result['target_pos'][0]:.4f}, {result['target_pos'][1]:.4f}, {result['target_pos'][2]:.4f})"
        )
        print(
            f"  -> Closest Atom ID: {result['closest_atom_id']} (Symbol: {result['closest_atom_symbol']})"
        )
        print(
            f"     Position: ({result['closest_atom_pos'][0]:.4f}, {result['closest_atom_pos'][1]:.4f}, {result['closest_atom_pos'][2]:.4f})"
        )
        print(f"     Distance: {result['distance']:.6f}\n")

else:
    print("\nAborted due to errors during parsing or file reading.")
