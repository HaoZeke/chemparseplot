# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Lazy loading for optional ``parsimonious`` (grammar track).

```{versionadded} 1.9.9
```
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

_grammar_mod: Any | None = None
_nodes_mod: Any | None = None
_probe: bool | None = None

_INSTALL_HINT = (
    "parsimonious is not installed. "
    "Install with: pip install 'chemparseplot[grammar]' "
    "for grammar/AST text parsers (XYZ, text-heavy ORCA sections)."
)


def grammar_available() -> bool:
    """Return True if ``parsimonious`` can be imported."""
    global _probe
    if _probe is not None:
        return _probe
    try:
        _probe = importlib.util.find_spec("parsimonious") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        _probe = False
    return _probe


def get_grammar_class() -> Any:
    """Return ``parsimonious.grammar.Grammar``, importing lazily."""
    global _grammar_mod
    if _grammar_mod is not None:
        return _grammar_mod.Grammar
    try:
        _grammar_mod = importlib.import_module("parsimonious.grammar")
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return _grammar_mod.Grammar


def get_node_visitor_class() -> Any:
    """Return ``parsimonious.nodes.NodeVisitor``, importing lazily."""
    global _nodes_mod
    if _nodes_mod is not None:
        return _nodes_mod.NodeVisitor
    try:
        _nodes_mod = importlib.import_module("parsimonious.nodes")
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return _nodes_mod.NodeVisitor


def reset_grammar_cache() -> None:
    """Clear cached parsimonious import state (tests only)."""
    global _grammar_mod, _nodes_mod, _probe
    _grammar_mod = None
    _nodes_mod = None
    _probe = None
