# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""PLUMED metadynamics HILLS file parsing and FES reconstruction."""

import string
import warnings

import numpy as np

_MIN_COLS_1D = 5
_MIN_COLS_2D = 7
_ALPHABET_SIZE = 26
_MIN_BIN_SIZE = 2


def _calculate_fes_2d(hills_data, x, y, per, npoints):
    """
    Helper function to calculate the 2D Free Energy Surface.

    The FES is calculated as the negative sum of Gaussian kernels.
    FES(x, y) = - sum_{i} [ H_i * exp( - (dx_i^2 / (2*sx_i^2)) - (dy_i^2 / (2*sy_i^2)) ) ]
    """
    # Create a 2D grid of coordinates
    # Note: The 'ij' indexing ensures the output grid has shape (len(x), len(y))
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")

    # Initialize FES to zero
    fes = np.zeros((npoints, npoints))

    # Get periodicity and ranges
    per_x, per_y = per
    range_x = x[-1] - x[0]
    range_y = y[-1] - y[0]

    # Unpack hills data for clarity
    centers_x = hills_data[:, 1]
    centers_y = hills_data[:, 2]
    sigmas_x = hills_data[:, 3]
    sigmas_y = hills_data[:, 4]
    heights = hills_data[:, 5]

    # Loop over each Gaussian hill and add it to the FES
    for i in range(len(heights)):
        # Calculate distance from grid points to the center of the current hill
        dx = grid_x - centers_x[i]
        dy = grid_y - centers_y[i]

        # Apply periodic boundary conditions if necessary
        if per_x:
            dx = dx - range_x * np.round(dx / range_x)
        if per_y:
            dy = dy - range_y * np.round(dy / range_y)

        # Calculate the Gaussian kernel for the current hill
        arg_x = (dx / sigmas_x[i]) ** 2
        arg_y = (dy / sigmas_y[i]) ** 2

        # Add the contribution of this hill to the total bias potential
        fes += heights[i] * np.exp(-0.5 * (arg_x + arg_y))

    # The Free Energy Surface is the negative of the summed bias potential
    return -fes


def _calculate_fes_1d(hills_data, x, per, npoints):
    """
    Helper function to calculate the 1D Free Energy Surface.

    FES(x) = - sum_{i} [ H_i * exp( - (dx_i^2 / (2*sx_i^2)) ) ]
    """
    # Grid is just the 1D vector x
    grid_x = x

    # Initialize FES to zero
    fes = np.zeros(npoints)

    # Get periodicity and range
    per_x = per[0]
    range_x = x[-1] - x[0]

    # Unpack hills data
    centers_x = hills_data[:, 1]
    sigmas_x = hills_data[:, 2]
    heights = hills_data[:, 3]

    # Loop over each Gaussian hill
    for i in range(len(heights)):
        # Calculate distance
        dx = grid_x - centers_x[i]

        # Apply periodic boundary conditions if necessary
        if per_x:
            dx = dx - range_x * np.round(dx / range_x)

        # Calculate Gaussian kernel
        arg_x = (dx / sigmas_x[i]) ** 2

        # Add to bias potential
        fes += heights[i] * np.exp(-0.5 * arg_x)

    # The FES is the negative of the bias potential
    return -fes


def calculate_fes_from_hills(hills, imin=1, imax=None, xlim=None, ylim=None, npoints=256):
    """
    Calculates a 1D or 2D Free Energy Surface (FES) from a metadynamics hills file.

    .. versionadded:: 0.0.3

    This is a Python/NumPy translation of the R function fes2.hillsfile.

    Args:
        hills (dict): A dictionary containing the metadynamics data. Expected keys:
            'hillsfile' (np.ndarray): The hills data. For 1D FES, shape is (N, 5).
                                      For 2D FES, shape is (N, 7).
                                      Columns are typically:
                                      time, cv1, (cv2), sigma_cv1,
                                      (sigma_cv2), height, ...
            'per' (list or tuple): A boolean list indicating periodicity for each CV.
                                   e.g., [False, False].
            'pcv1' (list or tuple): Periodic boundary limits for CV1. e.g., [0, 2*pi].
            'pcv2' (list or tuple): Periodic boundary limits for CV2.
        imin (int, optional): The starting hill index (1-based) to include in the FES.
                              Defaults to 1.
        imax (int, optional): The final hill index (1-based) to include. If None, all
                              hills from imin to the end are used. Defaults to None.
        xlim (list or tuple, optional): Manual limits for the x-axis (CV1).
                                        Defaults to None, which auto-detects limits.
        ylim (list or tuple, optional): Manual limits for the y-axis (CV2).
                                        Defaults to None, which auto-detects limits.
        npoints (int, optional): The number of grid points for each dimension.
                                 Defaults to 256.

    Returns:
        dict: A dictionary containing the FES and associated metadata, with keys:
              'fes': The calculated FES as a 1D or 2D NumPy array.
              'hills': The original hills data used.
              'rows': Number of grid points (npoints).
              'dimension': 1 or 2.
              'per': Periodicity flags.
              'x': The grid coordinates for the first dimension (CV1).
              'y': The grid coordinates for the second dimension (CV2, if applicable).
              'pcv1', 'pcv2': Periodic boundary values.
    """
    hills_data = hills["hillsfile"]
    num_hills, num_cols = hills_data.shape

    # --- Parameter Validation and Setup ---
    if imax is not None and num_hills < imax:
        warnings.warn(
            f"Warning: You requested imax={imax}, but only "
            f"{num_hills} hills are available. Using all hills.",
            stacklevel=2,
        )
        imax = num_hills

    if imax is None:
        imax = num_hills

    if imax > 0 and imin > imax:
        msg = "Error: imax cannot be lower than imin."
        raise ValueError(msg)

    # Convert 1-based R-style indexing to 0-based Python slicing
    # Note: The `imax` in a Python slice [start:end] is exclusive, so it's correct.
    start_index = imin - 1
    end_index = imax

    # --- Main Logic: Branch based on dimension (number of columns) ---

    # --- Case 1: 2D FES (CV1, CV2, sigma1, sigma2, height) ---
    if num_cols >= _MIN_COLS_2D:  # Usually 7 for Plumed output
        dimension = 2

        if imax == 0:
            # Create an empty grid if no hills are requested
            min_cv1, max_cv1 = (0, 1) if xlim is None else xlim
            min_cv2, max_cv2 = (0, 1) if ylim is None else ylim

            # Add 5% padding
            dx = max_cv1 - min_cv1
            dy = max_cv2 - min_cv2
            xlims = [min_cv1 - 0.05 * dx, max_cv1 + 0.05 * dx]
            ylims = [min_cv2 - 0.05 * dy, max_cv2 + 0.05 * dy]

            x = np.linspace(xlims[0], xlims[1], npoints)
            y = np.linspace(ylims[0], ylims[1], npoints)
            fesm = np.zeros((npoints, npoints))
        else:
            # Determine grid boundaries
            min_cv1, max_cv1 = np.min(hills_data[:, 1]), np.max(hills_data[:, 1])
            min_cv2, max_cv2 = np.min(hills_data[:, 2]), np.max(hills_data[:, 2])

            dx = max_cv1 - min_cv1
            dy = max_cv2 - min_cv2
            xlims = [min_cv1 - 0.05 * dx, max_cv1 + 0.05 * dx]
            ylims = [min_cv2 - 0.05 * dy, max_cv2 + 0.05 * dy]

            # Override with user-defined or periodic limits
            if xlim is not None:
                xlims = xlim
            elif hills["per"][0] and "pcv1" in hills:
                xlims = hills["pcv1"]

            if ylim is not None:
                ylims = ylim
            elif hills["per"][1] and "pcv2" in hills:
                ylims = hills["pcv2"]

            # Create grid coordinates
            x = np.linspace(xlims[0], xlims[1], npoints)
            y = np.linspace(ylims[0], ylims[1], npoints)

            # Select hills and calculate FES
            selected_hills = hills_data[start_index:end_index]
            fesm = _calculate_fes_2d(selected_hills, x, y, hills["per"], npoints)

        # Prepare results
        result = {
            "fes": fesm,
            "hills": hills_data,
            "rows": npoints,
            "dimension": dimension,
            "per": hills["per"],
            "x": x,
            "y": y,
            "pcv1": hills.get("pcv1"),
            "pcv2": hills.get("pcv2"),
        }

    # --- Case 2: 1D FES (CV1, sigma1, height) ---
    elif num_cols >= _MIN_COLS_1D:  # Usually 5 for Plumed output
        dimension = 1

        if imax == 0:
            # Create an empty line
            min_cv1, max_cv1 = (0, 1) if xlim is None else xlim
            dx = max_cv1 - min_cv1
            xlims = [min_cv1 - 0.05 * dx, max_cv1 + 0.05 * dx]

            x = np.linspace(xlims[0], xlims[1], npoints)
            fesm = np.zeros(npoints)
        else:
            # Determine grid boundaries
            min_cv1, max_cv1 = np.min(hills_data[:, 1]), np.max(hills_data[:, 1])
            dx = max_cv1 - min_cv1
            xlims = [min_cv1 - 0.05 * dx, max_cv1 + 0.05 * dx]

            # Override with user-defined or periodic limits
            if xlim is not None:
                xlims = xlim
            elif hills["per"][0] and "pcv1" in hills:
                xlims = hills["pcv1"]

            # Create grid coordinates
            x = np.linspace(xlims[0], xlims[1], npoints)

            # Select hills and calculate FES
            selected_hills = hills_data[start_index:end_index]
            fesm = _calculate_fes_1d(selected_hills, x, hills["per"], npoints)

        # Prepare results
        result = {
            "fes": fesm,
            "hills": hills_data,
            "rows": npoints,
            "dimension": dimension,
            "per": hills["per"],
            "x": x,
            "pcv1": hills.get("pcv1"),
            "pcv2": hills.get("pcv2"),
        }

    else:
        msg = (
            f"Unsupported number of columns in hillsfile: {num_cols}. "
            f"Expected >= {_MIN_COLS_1D} for 1D or >= {_MIN_COLS_2D} for 2D."
        )
        raise ValueError(msg)

    return result


def find_fes_minima(fes_result, nbins=8):
    """
    Finds local minima on a Free Energy Surface by dividing it into bins.

    .. versionadded:: 0.0.3

    This is a Python/Pandas translation of the R function fesminima.fes.

    Args:
        fes_result (dict): The output dictionary from calculate_fes_from_hills.
        nbins (int): The number of bins to divide each dimension into for the search.

    Returns:
        dict: A dictionary containing a pandas DataFrame of the minima and the original
              FES data. Returns None if no minima are found.
    """
    fes = fes_result["fes"]
    rows = fes_result["rows"]
    dimension = fes_result["dimension"]
    per = fes_result["per"]

    if rows % nbins != 0:
        msg = "Error: npoints (rows) in FES must be an integer multiple of nbins."
        raise ValueError(msg)

    rb = rows // nbins
    if rb < _MIN_BIN_SIZE:
        msg = "Error: nbins is too high for the grid size, try reducing it."
        raise ValueError(msg)

    minima_info = []

    if dimension == 2:  # noqa: PLR2004
        for i in range(nbins):
            for j in range(nbins):
                # Define the search window for this bin, including a 1-point border
                # This is the key to checking if a minimum is truly local
                # and not on the edge of the search window.
                i_start, i_end = i * rb - 1, (i + 1) * rb + 1
                j_start, j_end = j * rb - 1, (j + 1) * rb + 1

                # Create index arrays for slicing, handling periodicity and boundaries
                indices_i = np.arange(i_start, i_end)
                indices_j = np.arange(j_start, j_end)

                # Handle non-periodic boundaries by clipping
                if not per[0]:
                    indices_i = np.clip(indices_i, 0, rows - 1)
                if not per[1]:
                    indices_j = np.clip(indices_j, 0, rows - 1)

                # Create the sub-FES view, using np.take for periodic wrapping
                sub_fes = np.take(
                    np.take(fes, indices_i, axis=0, mode="wrap" if per[0] else "clip"),
                    indices_j,
                    axis=1,
                    mode="wrap" if per[1] else "clip",
                )

                # Find the location of the minimum within this sub-grid
                min_loc_flat = np.argmin(sub_fes)
                min_loc_i, min_loc_j = np.unravel_index(min_loc_flat, sub_fes.shape)

                # CRUCIAL CHECK: Is the minimum on the border of the search window?
                # If so, it's not a true local minimum for this bin, so we discard it.
                is_on_border = (
                    min_loc_i == 0
                    or min_loc_i == len(indices_i) - 1
                    or min_loc_j == 0
                    or min_loc_j == len(indices_j) - 1
                )

                if not is_on_border:
                    # Convert local sub-grid index to global FES index
                    global_i = indices_i[min_loc_i] % rows
                    global_j = indices_j[min_loc_j] % rows
                    minima_info.append(
                        {
                            "CV1bin": global_i,
                            "CV2bin": global_j,
                            "CV1": fes_result["x"][global_i],
                            "CV2": fes_result["y"][global_j],
                            "free_energy": fes[global_i, global_j],
                        }
                    )

    elif dimension == 1:
        for i in range(nbins):
            i_start, i_end = i * rb - 1, (i + 1) * rb + 1
            indices_i = np.arange(i_start, i_end)

            if not per[0]:
                indices_i = np.clip(indices_i, 0, rows - 1)

            sub_fes = np.take(fes, indices_i, mode="wrap" if per[0] else "clip")
            min_loc_i = np.argmin(sub_fes)

            is_on_border = min_loc_i == 0 or min_loc_i == len(indices_i) - 1

            if not is_on_border:
                global_i = indices_i[min_loc_i] % rows
                minima_info.append(
                    {
                        "CV1bin": global_i,
                        "CV1": fes_result["x"][global_i],
                        "free_energy": fes[global_i],
                    }
                )

    if not minima_info:
        return None  # No minima found

    # Convert to a pandas DataFrame and remove duplicates
    import pandas as pd

    minima_df = pd.DataFrame(minima_info)
    if dimension == 2:  # noqa: PLR2004
        minima_df = minima_df.drop_duplicates(subset=["CV1bin", "CV2bin"]).reset_index(
            drop=True
        )
    else:
        minima_df = minima_df.drop_duplicates(subset=["CV1bin"]).reset_index(drop=True)

    # Sort by free energy and add letter labels
    minima_df = minima_df.sort_values(by="free_energy").reset_index(drop=True)

    # Generate labels (A, B, ..., Z, AA, AB, ...)
    labels = list(string.ascii_uppercase)
    if len(minima_df) > _ALPHABET_SIZE:
        extra_labels = [
            f"{c1}{c2}" for c1 in string.ascii_uppercase for c2 in string.ascii_uppercase
        ]
        labels.extend(extra_labels)
    minima_df.insert(0, "letter", labels[: len(minima_df)])

    # Create a copy of the input dictionary and add the minima DataFrame
    minima_result = fes_result.copy()
    minima_result["minima"] = minima_df
    return minima_result
