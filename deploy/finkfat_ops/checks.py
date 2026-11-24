"""Local safety checks shared by ``ff-ops config check`` and ``ff-ops doctor``.

Each check returns :class:`Finding` values carrying the fix to apply. The evaluation
logic is pure; the ``check_*`` functions only gather the inputs (git, file modes).
"""

from __future__ import annotations

import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from finkfat_ops.config import DeployPaths
from finkfat_ops.runner import Command, Runner


class Level(StrEnum):
    """Severity of a finding."""

    OK = "ok"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class Finding:
    """Result of one check.

    Attributes
    ----------
    level : Level
        Severity.
    message : str
        What was found.
    fix : str | None
        Command or action that fixes it, if any.
    """

    level: Level
    message: str
    fix: str | None = None


def git_tracked_command(root: Path, files: list[Path]) -> Command:
    """Build the command listing which of ``files`` are tracked by git.

    Parameters
    ----------
    root : Path
        Directory the command runs in (``deploy/``).
    files : list[Path]
        Files to test, inside ``root``.
    """
    names = [str(path.relative_to(root)) for path in files]
    return Command.of("git", "-C", str(root), "ls-files", "--", *names, timeout_s=30)


def evaluate_git_tracked(root: Path, files: list[Path], ls_files_output: str) -> list[Finding]:
    """Turn the output of :func:`git_tracked_command` into findings.

    Parameters
    ----------
    root : Path
        Same directory as given to :func:`git_tracked_command`.
    files : list[Path]
        Files that were tested.
    ls_files_output : str
        Standard output of ``git ls-files`` (one tracked path per line).

    Returns
    -------
    list[Finding]
        One ERROR per tracked file, or a single OK finding.
    """
    tracked = {line.strip() for line in ls_files_output.splitlines() if line.strip()}
    findings = [
        Finding(
            Level.ERROR,
            f"{name} is tracked by git: it would be published in the public repository",
            fix=f"git -C {root} rm --cached {name}   (then rotate every secret it contains)",
        )
        for name in (str(path.relative_to(root)) for path in files)
        if name in tracked
    ]
    return findings or [Finding(Level.OK, "sensitive files are not tracked by git")]


def evaluate_file_mode(path: Path, mode: int) -> Finding:
    """Check that a secret file is readable by its owner only.

    Parameters
    ----------
    path : Path
        The file (used in messages).
    mode : int
        Its ``st_mode``.

    Returns
    -------
    Finding
        OK when group and others have no access, WARN otherwise.
    """
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        return Finding(
            Level.WARN,
            f"{path.name} is readable by other users (mode {stat.S_IMODE(mode):o})",
            fix=f"chmod 600 {path}",
        )
    return Finding(Level.OK, f"{path.name} permissions are 600")


def check_sensitive_files(paths: DeployPaths, runner: Runner) -> list[Finding]:
    """Check that the gitignored files are neither tracked nor world-readable.

    Parameters
    ----------
    paths : DeployPaths
        Deployment file locations.
    runner : Runner
        Runs the ``git`` command.

    Returns
    -------
    list[Finding]
        Findings for git tracking and for the mode of each existing sensitive file.
    """
    files = paths.sensitive_files()
    result = runner.run(git_tracked_command(paths.root, files), check=False)
    if result.ok:
        findings = evaluate_git_tracked(paths.root, files, result.stdout)
    else:
        findings = [Finding(Level.WARN, f"could not query git: {result.stderr.strip()}")]
    for path in files:
        if path.exists():
            findings.append(evaluate_file_mode(path, path.stat().st_mode))
    return findings
