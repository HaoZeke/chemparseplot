# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

# https://regex101.com/r/jHAG2T/1
# DIGIT pattern for a floating-point number, possibly negative
DIGIT = r"-?\d+\.\d+"
TWO_COL_NUM = (
    r"\s*"  # Optional whitespace
    r"(?P<twocolnum>"  # Named group 'twocolnum' starts
    r"(?:"
    r"\s*"  # Optional whitespace
    + DIGIT
    + r"\s+"  # whitespace characters between numbers
    + DIGIT
    + r")+)"  # The non-capturing group ends;
    # '+' allows for one or more occurrences of the pattern
)
