"""Builders for ``openstack`` command lines and parsers for their JSON output.

Every cloud command goes through :class:`OpenStackCli`, which always passes
``--os-cloud`` (the profile of ``infra.local.yaml``) so that no ambient ``OS_*``
environment variable can redirect ff-ops to another project.

A profile authenticating with a password (``auth_type: password``) keeps the password out of
``clouds.yaml``: ff-ops asks for it once and hands it to every command as ``OS_PASSWORD``
(:func:`profile_password`). Heat needs such a profile on clouds where a token obtained from an
application credential cannot create the Keystone trust of a stack.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finkfat_ops.runner import Command, CommandResult, Runner

DEFAULT_CLOUD_TIMEOUT_S = 60.0
"""Timeout of a short OpenStack API call (an unreachable endpoint fails instead of hanging)."""

WAIT_TIMEOUT_S = 1800.0
"""Timeout of a ``--wait`` call (stack creation, update, deletion)."""


def default_openstack_binary() -> str:
    """Return the ``openstack`` executable of ff-ops' own environment, if installed there.

    Returns
    -------
    str
        Path of the CLI next to the running Python interpreter, or ``"openstack"``
        (resolved through ``PATH``) when it is not installed there.
    """
    candidate = Path(sys.executable).with_name("openstack")
    return str(candidate) if candidate.exists() else "openstack"


@dataclass(frozen=True)
class OpenStackCli:
    """Builds ``openstack`` commands bound to one cloud profile.

    Attributes
    ----------
    cloud : str
        Profile name in ``clouds.yaml``.
    binary : str
        The ``openstack`` executable.
    password : str | None
        Password of a password-based profile, passed as ``OS_PASSWORD``; ``None`` for a
        profile that needs none (application credential).
    """

    cloud: str
    binary: str = "openstack"
    password: str | None = field(default=None, repr=False)

    def command(
        self, *args: str, json_output: bool = True, timeout_s: float = DEFAULT_CLOUD_TIMEOUT_S
    ) -> Command:
        """Build an ``openstack`` command.

        Parameters
        ----------
        *args : str
            Sub-command and its arguments, e.g. ``"stack", "show", "ffat-prod-net"``.
        json_output : bool
            Append ``-f json`` (for commands that print a result).
        timeout_s : float
            Seconds before the command is killed.

        Returns
        -------
        Command
            The full command line.
        """
        argv = [self.binary, "--os-cloud", self.cloud, *args]
        if json_output:
            argv += ["-f", "json"]
        env = (("OS_PASSWORD", self.password),) if self.password is not None else ()
        return Command(argv=tuple(argv), timeout_s=timeout_s, env=env)


def profile_password(
    ask_password: bool, environ: Mapping[str, str], ask: Callable[[str], str]
) -> str | None:
    """Return the password of the cloud profile, asking for it only when needed.

    Parameters
    ----------
    ask_password : bool
        The profile authenticates with a password (``openstack.ask_password``).
    environ : Mapping[str, str]
        The process environment; a non-empty ``OS_PASSWORD`` there is used as is.
    ask : Callable[[str], str]
        Asks the operator for the password without echoing it.

    Returns
    -------
    str | None
        The password, or ``None`` when the profile needs none.
    """
    if not ask_password:
        return None
    return environ.get("OS_PASSWORD") or ask("OpenStack password")


def is_not_found(result: CommandResult) -> bool:
    """Tell whether a failed ``openstack … show`` means "this resource does not exist".

    Parameters
    ----------
    result : CommandResult
        Result of the command.

    Returns
    -------
    bool
        ``True`` for a failure whose message says the object was not found.
    """
    text = f"{result.stderr}\n{result.stdout}".lower()
    return not result.ok and "not found" in text


def parse_json(result: CommandResult) -> Any:
    """Parse the JSON output of a command; an empty output (dry-run) gives ``None``.

    Parameters
    ----------
    result : CommandResult
        Result of an ``openstack … -f json`` command.

    Returns
    -------
    Any
        The decoded JSON, or ``None`` for an empty output.
    """
    return json.loads(result.stdout) if result.stdout.strip() else None


class ProjectMismatchError(RuntimeError):
    """The cloud profile points to another project than the one in ``infra.local.yaml``."""


def verify_project(cli: OpenStackCli, runner: Runner, expected: str) -> str | None:
    """Check that the cloud profile authenticates into the expected project.

    ``expected`` may be the project name or its ID. The name is resolved with
    ``openstack project show``, which members of a project are normally allowed to run.

    Parameters
    ----------
    cli : OpenStackCli
        Command builder bound to the profile.
    runner : Runner
        Runs the commands.
    expected : str
        Project name or ID from ``infra.local.yaml``.

    Returns
    -------
    str | None
        The project ID, or ``None`` in dry-run mode (no real answer).

    Raises
    ------
    ProjectMismatchError
        The profile authenticates into another project, or the name cannot be resolved.
    """
    token = parse_json(runner.run(cli.command("token", "issue")))
    if token is None:
        return None
    project_id = token["project_id"]
    if expected == project_id:
        return project_id
    shown = runner.run(cli.command("project", "show", project_id), check=False)
    if not shown.ok:
        raise ProjectMismatchError(
            f"cannot resolve the name of project {project_id} ({shown.stderr.strip()}); "
            "put the project ID in infra.local.yaml (openstack.project) instead of its name"
        )
    name = parse_json(shown)["name"]
    if name != expected:
        raise ProjectMismatchError(
            f"cloud profile '{cli.cloud}' authenticates into project '{name}' ({project_id}), "
            f"but infra.local.yaml expects '{expected}'"
        )
    return project_id
