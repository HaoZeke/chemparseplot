# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Internal OPI (ORCA Python Interface) access for chemparseplot.

OPI is an **optional** dependency. Public callers must use
:mod:`chemparseplot.parse.orca.neb` (e.g. :func:`parse_orca_neb`) and must
**not** import ``opi`` for suite-supported workflows.

```{versionadded} 1.9.0
```
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

_opi_output_cls: Any | None = None
_opi_probe: bool | None = None

_INSTALL_HINT = (
    "ORCA Python Interface (OPI) is not installed. "
    "Install with: pip install 'chemparseplot[opi]' "
    "(package provides the ``opi`` import). "
    "Callers should use chemparseplot.parse.orca.neb.parse_orca_neb "
    "rather than importing opi directly."
)


def opi_available() -> bool:
    """Return True if the ``opi`` package can be imported."""
    global _opi_probe
    if _opi_probe is not None:
        return _opi_probe
    try:
        _opi_probe = importlib.util.find_spec("opi") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        _opi_probe = False
    return _opi_probe


def get_opi_output_class() -> Any:
    """Return ``opi.output.core.Output``, importing lazily.

    Raises
    ------
    ImportError
        If OPI is not installed (with install hint). Does not auto-pip-install.
    """
    global _opi_output_cls
    if _opi_output_cls is not None:
        return _opi_output_cls
    try:
        core = importlib.import_module("opi.output.core")
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    _opi_output_cls = core.Output
    return _opi_output_cls


def reset_opi_cache() -> None:
    """Clear cached OPI import state (tests only)."""
    global _opi_output_cls, _opi_probe
    _opi_output_cls = None
    _opi_probe = None
