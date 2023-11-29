# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

# https://regex101.com/r/jHAG2T/1
# DIGIT pattern for a floating-point number, possibly negative
DIGIT = r"-?\d+\.\d+"


def create_multicol_pattern(num_cols, pname="multicolnum"):
    if num_cols < 1:
        error_message = "Number of columns must be at least 1"
        raise ValueError(error_message)

    # Building the pattern for N columns
    pattern = (
        r"\s*"  # Optional leading whitespace
        rf"(?P<{pname}>"  # Named group
        r"(?:"
    )

    # Add DIGIT pattern for each column, with whitespace
    for _ in range(num_cols):
        pattern += r"\s*"  # Optional whitespace before each number
        pattern += DIGIT
    pattern += r")+"  # Repeat for multiple lines
    pattern += r")"  # End of named group
    return pattern


TWO_COL_NUM = create_multicol_pattern(2, "twocolnum")
THREE_COL_NUM = create_multicol_pattern(3, "threecolnum")
