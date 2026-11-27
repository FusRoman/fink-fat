"""Tests of finkfat_ops.checks."""

from pathlib import Path

from finkfat_ops.checks import (
    Level,
    check_sensitive_files,
    evaluate_file_mode,
    evaluate_git_tracked,
    git_tracked_command,
)
from finkfat_ops.config import DeployPaths
from finkfat_ops.runner import CommandResult, FakeRunner


def test_git_command_uses_paths_relative_to_deploy(tmp_path: Path) -> None:
    paths = DeployPaths(tmp_path)
    command = git_tracked_command(tmp_path, paths.sensitive_files())
    assert command.argv == (
        "git",
        "-C",
        str(tmp_path),
        "ls-files",
        "--",
        "infra.local.yaml",
        "secrets.env",
        "config/stream.yaml",
    )


def test_tracked_stream_config_is_an_error(tmp_path: Path) -> None:
    files = DeployPaths(tmp_path).sensitive_files()
    findings = evaluate_git_tracked(tmp_path, files, "config/stream.yaml\n")
    assert [f.level for f in findings] == [Level.ERROR]
    assert "config/stream.yaml is tracked" in findings[0].message


def test_tracked_sensitive_file_is_an_error(tmp_path: Path) -> None:
    files = DeployPaths(tmp_path).sensitive_files()
    findings = evaluate_git_tracked(tmp_path, files, "secrets.env\n")
    assert [f.level for f in findings] == [Level.ERROR]
    assert "secrets.env is tracked" in findings[0].message
    assert "rm --cached secrets.env" in (findings[0].fix or "")


def test_untracked_files_are_fine(tmp_path: Path) -> None:
    findings = evaluate_git_tracked(tmp_path, DeployPaths(tmp_path).sensitive_files(), "")
    assert [f.level for f in findings] == [Level.OK]


def test_file_mode_must_be_private(tmp_path: Path) -> None:
    path = tmp_path / "secrets.env"
    assert evaluate_file_mode(path, 0o100600).level == Level.OK
    finding = evaluate_file_mode(path, 0o100644)
    assert finding.level == Level.WARN and finding.fix == f"chmod 600 {path}"


def test_check_sensitive_files_combines_git_and_modes(deploy_dir: Path) -> None:
    (deploy_dir / "secrets.env").chmod(0o640)
    runner = FakeRunner().on(["git"], CommandResult(stdout=""))
    findings = check_sensitive_files(DeployPaths(deploy_dir), runner)
    levels = [f.level for f in findings]
    assert levels.count(Level.OK) == 3 and levels.count(Level.WARN) == 1


def test_git_unavailable_is_a_warning(deploy_dir: Path) -> None:
    runner = FakeRunner().on(["git"], CommandResult(returncode=128, stderr="not a git repository"))
    findings = check_sensitive_files(DeployPaths(deploy_dir), runner)
    assert findings[0].level == Level.WARN and "not a git repository" in findings[0].message
