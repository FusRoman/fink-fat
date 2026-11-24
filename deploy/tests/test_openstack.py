"""Tests of finkfat_ops.openstack (project verification, command building)."""

from __future__ import annotations

import json

import pytest

from finkfat_ops.openstack import (
    OpenStackCli,
    ProjectMismatchError,
    is_not_found,
    parse_json,
    profile_password,
    verify_project,
)
from finkfat_ops.runner import CommandResult, DryRunRunner, FakeRunner

CLI = OpenStackCli(cloud="test-cloud", binary="openstack")
TOKEN = CommandResult(stdout=json.dumps({"project_id": "p123", "user_id": "u", "id": "t"}))


def _project(name: str) -> CommandResult:
    return CommandResult(stdout=json.dumps({"id": "p123", "name": name}))


def test_command_always_names_the_cloud_and_asks_for_json() -> None:
    command = CLI.command("stack", "list")
    assert command.argv == ("openstack", "--os-cloud", "test-cloud", "stack", "list", "-f", "json")
    assert CLI.command("stack", "delete", "x", json_output=False).argv[-1] == "x"


def test_password_goes_to_the_environment_never_to_argv_or_repr() -> None:
    cli = OpenStackCli(cloud="test-cloud", binary="openstack", password="s3cret")
    command = cli.command("stack", "list")
    assert command.env == (("OS_PASSWORD", "s3cret"),)
    assert "s3cret" not in command.display()
    assert "s3cret" not in repr(command) and "s3cret" not in repr(cli)
    assert CLI.command("stack", "list").env == ()


def test_password_is_asked_only_for_a_password_profile_without_os_password() -> None:
    asked: list[str] = []

    def ask(question: str) -> str:
        asked.append(question)
        return "typed"

    assert profile_password(False, {"OS_PASSWORD": "env"}, ask) is None
    assert profile_password(True, {"OS_PASSWORD": "env"}, ask) == "env"
    assert profile_password(True, {"OS_PASSWORD": ""}, ask) == "typed"
    assert len(asked) == 1


def test_project_matching_by_id_needs_no_name_lookup() -> None:
    runner = FakeRunner().on(["openstack", "--os-cloud", "test-cloud", "token"], TOKEN)
    assert verify_project(CLI, runner, "p123") == "p123"
    assert len(runner.calls) == 1


def test_project_matching_by_name() -> None:
    runner = FakeRunner().on(["openstack"], TOKEN, times=1).on(["openstack"], _project("fink"))
    assert verify_project(CLI, runner, "fink") == "p123"


def test_wrong_project_is_refused() -> None:
    runner = FakeRunner().on(["openstack"], TOKEN, times=1).on(["openstack"], _project("other"))
    with pytest.raises(ProjectMismatchError, match="'other'"):
        verify_project(CLI, runner, "fink")


def test_unresolvable_name_asks_for_the_id() -> None:
    forbidden = CommandResult(returncode=1, stderr="Forbidden (HTTP 403)")
    runner = FakeRunner().on(["openstack"], TOKEN, times=1).on(["openstack"], forbidden)
    with pytest.raises(ProjectMismatchError, match="put the project ID"):
        verify_project(CLI, runner, "fink")


def test_dry_run_skips_the_check() -> None:
    assert verify_project(CLI, DryRunRunner(echo=lambda _: None), "fink") is None


def test_not_found_detection_and_empty_json() -> None:
    assert is_not_found(CommandResult(returncode=1, stderr="Stack not found: ffat-prod-net"))
    assert not is_not_found(CommandResult(returncode=1, stderr="Forbidden"))
    assert parse_json(CommandResult(stdout="")) is None
