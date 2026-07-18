# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Grammar dependency probe tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from chemparseplot.parse.grammar import _deps as deps
from chemparseplot.parse.grammar import grammar_available, reset_grammar_cache


def test_grammar_available_true_when_installed():
    pytest.importorskip("parsimonious")
    reset_grammar_cache()
    assert grammar_available() is True
    reset_grammar_cache()


def test_grammar_available_false_without_package():
    reset_grammar_cache()
    with patch("importlib.util.find_spec", return_value=None):
        reset_grammar_cache()
        assert deps.grammar_available() is False
    reset_grammar_cache()


def test_get_grammar_class_import_error_message():
    reset_grammar_cache()
    with patch.object(deps, "_grammar_mod", None):
        with patch("importlib.import_module", side_effect=ImportError("no peg")):
            reset_grammar_cache()
            with pytest.raises(ImportError, match="chemparseplot\\[grammar\\]"):
                deps.get_grammar_class()
    reset_grammar_cache()
