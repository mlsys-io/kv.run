"""
Basic tests for MLOC main module.
"""

import pytest
from typer.testing import CliRunner

from mloc.main import app


def test_version_command():
    """Test version command"""
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "MLOC v" in result.output


def test_help_command():
    """Test help command"""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "MLOC - Modular LLM Operations Container" in result.output