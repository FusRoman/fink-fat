"""Tests of the `ff-ops config check` command (no external command is really run)."""

from pathlib import Path

from typer.testing import CliRunner

from finkfat_ops.cli import AppContext, app
from finkfat_ops.config import DeployPaths
from finkfat_ops.runner import CommandResult, FakeRunner


def _invoke(deploy_dir: Path, git_stdout: str = ""):
    runner = FakeRunner().on(["git"], CommandResult(stdout=git_stdout))
    context = AppContext(paths=DeployPaths(deploy_dir), runner=runner)
    return CliRunner().invoke(app, ["config", "check"], obj=context)


def test_valid_configuration_passes(deploy_dir: Path) -> None:
    result = _invoke(deploy_dir)
    assert result.exit_code == 0, result.output
    assert "configuration is valid" in result.output
    assert "pg-password" not in result.output


def test_invalid_configuration_fails(deploy_dir: Path) -> None:
    (deploy_dir / "secrets.env").write_text("POSTGRES_PASSWORD=short\n", encoding="utf-8")
    result = _invoke(deploy_dir)
    assert result.exit_code == 1
    assert "PGADMIN_PASSWORD" in result.output


def test_tracked_secret_file_fails(deploy_dir: Path) -> None:
    result = _invoke(deploy_dir, git_stdout="secrets.env\n")
    assert result.exit_code == 1
    assert "tracked by git" in result.output
