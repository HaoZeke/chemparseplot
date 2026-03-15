# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Tests for plt-neb CLI commands."""

import pytest

adjustText = pytest.importorskip("adjustText")
polars = pytest.importorskip("polars")
pytestmark = pytest.mark.pure


class TestPltNebCLI:
    """Test plt-neb CLI commands."""

    def test_cli_main_command_exists(self) -> None:
        """Test that main CLI command exists."""
        from chemparseplot.scripts.plt_neb import main

        assert main is not None
        assert hasattr(main, "params")  # Click command has params

    def test_cli_help_output(self) -> None:
        """Test that CLI help is available."""
        from click.testing import CliRunner

        from chemparseplot.scripts.plt_neb import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "NEB" in result.output or "neb" in result.output
        assert "--plot-type" in result.output

    def test_cli_options_present(self) -> None:
        """Test that expected CLI options are present."""
        from click.testing import CliRunner

        from chemparseplot.scripts.plt_neb import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        # Check for key options
        assert "--plot-type" in result.output
        assert "--landscape-mode" in result.output
        assert "--surface-type" in result.output
        assert "--project-path" in result.output
        assert "--plot-structures" in result.output
        assert "--show-legend" in result.output
        assert "--output-file" in result.output or "-o" in result.output
        assert "--ira-kmax" in result.output
