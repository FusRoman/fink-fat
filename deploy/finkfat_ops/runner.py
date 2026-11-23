"""Execution of external commands (``openstack``, ``ssh``, ``docker``, ``git``…).

This is the only module of ff-ops that runs anything. Every other module builds
:class:`Command` values (pure data) and hands them to a :class:`Runner`:

* :class:`SubprocessRunner` — really runs the command (production use).
* :class:`DryRunRunner` — prints the command instead of running it (``--dry-run``).
* :class:`FakeRunner` — answers from recorded responses (tests; never touches the cloud).
"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_TIMEOUT_S = 120.0
"""Default timeout of a command, so that an unreachable endpoint never blocks forever."""


@dataclass(frozen=True)
class Command:
    """An external command to run.

    Attributes
    ----------
    argv : tuple[str, ...]
        Program and arguments, never interpreted by a shell.
    timeout_s : float | None
        Seconds before the command is killed; ``None`` waits forever
        (only for interactive commands such as ``ssh`` sessions).
    stdin : str | None
        Text sent to the command's standard input, if any.
    """

    argv: tuple[str, ...]
    timeout_s: float | None = DEFAULT_TIMEOUT_S
    stdin: str | None = None

    @classmethod
    def of(cls, *argv: str, timeout_s: float | None = DEFAULT_TIMEOUT_S) -> Command:
        """Build a command from positional arguments, e.g. ``Command.of("git", "status")``."""
        return cls(argv=tuple(argv), timeout_s=timeout_s)

    def display(self) -> str:
        """Return the command as a copy-pastable shell line."""
        return shlex.join(self.argv)


@dataclass(frozen=True)
class CommandResult:
    """Outcome of a command.

    Attributes
    ----------
    returncode : int
        Process exit status (0 means success).
    stdout : str
        Captured standard output.
    stderr : str
        Captured standard error.
    """

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        """Whether the command exited with status 0."""
        return self.returncode == 0

    def json(self) -> Any:
        """Parse the standard output as JSON (e.g. ``openstack … -f json``).

        Raises
        ------
        json.JSONDecodeError
            The output is not valid JSON.
        """
        return json.loads(self.stdout)


class CommandError(RuntimeError):
    """A command failed, timed out or could not be started."""

    def __init__(self, command: Command, message: str, result: CommandResult | None = None):
        super().__init__(f"{message}\n  command: {command.display()}")
        self.command = command
        self.result = result


class Runner(Protocol):
    """Something that runs :class:`Command` values."""

    def run(self, command: Command, *, check: bool = True) -> CommandResult:
        """Run ``command`` and return its result.

        Parameters
        ----------
        command : Command
            The command to run.
        check : bool
            Raise :class:`CommandError` when the exit status is not 0.
        """
        ...


def _check(command: Command, result: CommandResult, check: bool) -> CommandResult:
    """Raise :class:`CommandError` for a failed result when ``check`` is set."""
    if check and not result.ok:
        detail = result.stderr.strip() or result.stdout.strip() or "(no output)"
        raise CommandError(command, f"exit status {result.returncode}: {detail}", result)
    return result


class SubprocessRunner:
    """Run commands for real with :func:`subprocess.run` (no shell)."""

    def run(self, command: Command, *, check: bool = True) -> CommandResult:
        """Run ``command``; see :meth:`Runner.run`.

        Raises
        ------
        CommandError
            Non-zero exit (with ``check``), timeout, or missing program.
        """
        try:
            completed = subprocess.run(
                list(command.argv),
                input=command.stdin,
                capture_output=True,
                text=True,
                timeout=command.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CommandError(
                command, f"timed out after {command.timeout_s:.0f} s (endpoint unreachable?)"
            ) from exc
        except FileNotFoundError as exc:
            raise CommandError(command, f"program not found: {command.argv[0]}") from exc
        result = CommandResult(completed.returncode, completed.stdout, completed.stderr)
        return _check(command, result, check)


@dataclass
class DryRunRunner:
    """Print commands instead of running them.

    Every command "succeeds" with an empty output, so callers that need real data
    (e.g. reading stack outputs) must handle empty results in dry-run mode.

    Attributes
    ----------
    echo : Callable[[str], None]
        Where each command line is written (``print`` by default).
    commands : list[Command]
        Every command received, in order.
    """

    echo: Callable[[str], None] = print
    commands: list[Command] = field(default_factory=list)

    def run(self, command: Command, *, check: bool = True) -> CommandResult:
        """Record and print ``command``; see :meth:`Runner.run`."""
        self.commands.append(command)
        self.echo(f"[dry-run] {command.display()}")
        return CommandResult()


@dataclass(frozen=True)
class FakeResponse:
    """A recorded answer of :class:`FakeRunner`.

    Attributes
    ----------
    prefix : tuple[str, ...]
        The response applies to any command whose argv starts with this prefix.
    result : CommandResult
        The result returned for such a command.
    """

    prefix: tuple[str, ...]
    result: CommandResult


class UnexpectedCommandError(AssertionError):
    """:class:`FakeRunner` received a command it has no response for."""


@dataclass
class FakeRunner:
    """Test double answering commands from recorded responses.

    The first response whose ``prefix`` matches the command's argv wins. A command
    with no matching response raises :class:`UnexpectedCommandError`, so a test can never
    silently run something unexpected.

    Attributes
    ----------
    responses : list[FakeResponse]
        Recorded answers, tried in order.
    calls : list[Command]
        Every command received, in order.
    """

    responses: list[FakeResponse] = field(default_factory=list)
    calls: list[Command] = field(default_factory=list)

    def on(self, prefix: Sequence[str], result: CommandResult) -> FakeRunner:
        """Register ``result`` for commands starting with ``prefix``; returns ``self``."""
        self.responses.append(FakeResponse(tuple(prefix), result))
        return self

    def run(self, command: Command, *, check: bool = True) -> CommandResult:
        """Answer ``command`` from the recorded responses; see :meth:`Runner.run`.

        Raises
        ------
        UnexpectedCommandError
            No response matches the command.
        CommandError
            The matching result failed and ``check`` is set.
        """
        self.calls.append(command)
        for response in self.responses:
            if command.argv[: len(response.prefix)] == response.prefix:
                return _check(command, response.result, check)
        raise UnexpectedCommandError(f"no fake response for: {command.display()}")
