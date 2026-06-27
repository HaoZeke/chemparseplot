# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
import importlib.util

import pytest

from tests._optional_imports import has_module_spec, optional_import_available

# Define the requirements for each suite/marker
ENVIRONMENT_REQUIREMENTS = {
    "pure": ["numpy"],
    "neb": ["numpy", "polars", "ase", "h5py", "scipy", "matplotlib", "rgpycrumbs"],
}


def check_missing_modules(marker_name):
    """Returns a list of missing modules for a given marker."""
    modules = ENVIRONMENT_REQUIREMENTS.get(marker_name, [])
    missing = []
    for mod in modules:
        if mod in {"rgpycrumbs", "chemparseplot"}:
            if not has_module_spec(mod):
                missing.append(mod)
            elif not optional_import_available(mod):
                missing.append(mod)
        elif importlib.util.find_spec(mod) is None:
            missing.append(mod)
    return missing


def skip_if_not_env(marker_name):
    """Skips the entire module if dependencies for the marker remain uninstalled."""
    missing = check_missing_modules(marker_name)
    if missing:
        pytest.skip(
            f"Missing dependencies for '{marker_name}': {', '.join(missing)}",
            allow_module_level=True,
        )
