"""Tests of finkfat_ops.runner (only local, harmless commands are really executed)."""

import sys

import pytest

from finkfat_ops.runner import (
    Command,
    CommandError,
    CommandResult,
    DryRunRunner,
    FakeRunner,
    SubprocessRunner,
    UnexpectedCommandError,
)


def test_display_is_a_copy_pastable_shell_line() -> None:
    assert Command.of("echo", "a b", "c").display() == "echo 'a b' c"


def test_fake_runner_answers_by_prefix_and_records_calls() -> None:
    runner = FakeRunner().on(["openstack", "stack", "list"], CommandResult(stdout="[]"))
    result = runner.run(Command.of("openstack", "stack", "list", "-f", "json"))
    assert result.json() == []
    assert [c.argv for c in runner.calls] == [("openstack", "stack", "list", "-f", "json")]


def test_fake_runner_first_matching_response_wins() -> None:
    runner = (
        FakeRunner()
        .on(["git", "ls-files"], CommandResult(stdout="specific"))
        .on(["git"], CommandResult(stdout="generic"))
    )
    assert runner.run(Command.of("git", "ls-files")).stdout == "specific"
    assert runner.run(Command.of("git", "status")).stdout == "generic"


def test_fake_runner_rejects_unexpected_commands() -> None:
    with pytest.raises(UnexpectedCommandError):
        FakeRunner().run(Command.of("openstack", "server", "delete", "x"))


def test_failed_result_raises_only_when_checked() -> None:
    runner = FakeRunner().on(["false"], CommandResult(returncode=1, stderr="boom"))
    with pytest.raises(CommandError, match="boom"):
        runner.run(Command.of("false"))
    assert runner.run(Command.of("false"), check=False).returncode == 1


def test_dry_run_prints_and_never_executes() -> None:
    printed: list[str] = []
    runner = DryRunRunner(echo=printed.append)
    result = runner.run(Command.of("openstack", "stack", "delete", "ffat-prod-net"))
    assert result.ok and result.stdout == ""
    assert printed == ["[dry-run] openstack stack delete ffat-prod-net"]


def test_subprocess_runner_captures_output() -> None:
    result = SubprocessRunner().run(Command.of(sys.executable, "-c", "print('hello')"))
    assert result.stdout.strip() == "hello"


def test_subprocess_runner_reports_failures_and_timeouts() -> None:
    runner = SubprocessRunner()
    with pytest.raises(CommandError, match="exit status 3"):
        runner.run(Command.of(sys.executable, "-c", "import sys; sys.exit(3)"))
    with pytest.raises(CommandError, match="timed out"):
        runner.run(Command.of(sys.executable, "-c", "import time; time.sleep(5)", timeout_s=0.2))
    with pytest.raises(CommandError, match="program not found"):
        runner.run(Command.of("definitely-not-a-real-program-ffops"))
