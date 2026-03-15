# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Tests for ChemGP CLI registration."""

import pytest

h5py = pytest.importorskip("h5py")
pd = pytest.importorskip("pandas")
plotnine = pytest.importorskip("plotnine")
pytestmark = pytest.mark.pure


class TestCLIRegistration:
    """Test that CLI commands are properly registered."""

    def test_plot_gp_commands_registered(self) -> None:
        """Test that chemgp CLI commands are registered."""
        pytest.importorskip("click")
        from click.testing import CliRunner

        from chemparseplot.scripts.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "convergence" in result.output
        assert "surface" in result.output
        assert "quality" in result.output
        assert "rff" in result.output
        assert "nll" in result.output
        assert "sensitivity" in result.output
        assert "trust" in result.output
        assert "variance" in result.output
        assert "fps" in result.output
        assert "profile" in result.output

    def test_cli_command_help(self) -> None:
        """Test that individual CLI commands have help."""
        pytest.importorskip("click")
        from click.testing import CliRunner

        from chemparseplot.scripts.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["surface", "--help"])

        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--output" in result.output
        assert "--clamp-lo" in result.output
        assert "--clamp-hi" in result.output
