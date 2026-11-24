"""Tests of finkfat_ops.infra with a fake cloud (no real command is run)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finkfat_ops import infra
from finkfat_ops.access import AccessState
from finkfat_ops.config import DeployPaths, load_deployment
from finkfat_ops.naming import Stack
from finkfat_ops.openstack import OpenStackCli
from finkfat_ops.runner import CommandResult, FakeRunner
from finkfat_ops.stack import ChangeSet

OS = ["openstack", "--os-cloud", "test-cloud"]
NOT_FOUND = CommandResult(returncode=1, stderr="Stack not found: ffat-prod-net")


def _show(status: str = "CREATE_COMPLETE", reason: str = "") -> CommandResult:
    data = {"stack_name": "ffat-prod-net", "stack_status": status, "stack_status_reason": reason}
    return CommandResult(stdout=json.dumps(data))


def _preview(**kinds: list[str]) -> CommandResult:
    data = {kind: [{"resource_name": name} for name in names] for kind, names in kinds.items()}
    return CommandResult(stdout=json.dumps(data))


def _session(deploy_dir: Path, runner: FakeRunner, answer: bool = True) -> infra.InfraSession:
    return infra.InfraSession(
        deployment=load_deployment(DeployPaths(deploy_dir)),
        paths=DeployPaths(deploy_dir),
        cli=OpenStackCli(cloud="test-cloud", binary="openstack"),
        runner=runner,
        echo=lambda _: None,
        confirm=lambda _: answer,
    )


def _spec(session: infra.InfraSession) -> infra.StackSpec:
    return infra.build_stack_spec(Stack.NET, session.deployment, AccessState())


def _verbs(runner: FakeRunner) -> list[str]:
    return [
        " ".join(c.argv[3:5]) + (" --dry-run" if "--dry-run" in c.argv else "")
        for c in runner.calls
    ]


def test_only_the_net_stack_is_implemented(deploy_dir: Path) -> None:
    session = _session(deploy_dir, FakeRunner())
    with pytest.raises(infra.InfraError, match="not implemented"):
        infra.build_stack_spec(Stack.SERVERS, session.deployment, AccessState())


def test_render_writes_the_template(deploy_dir: Path) -> None:
    session = _session(deploy_dir, FakeRunner())
    path = infra.render(session, _spec(session))
    assert path == deploy_dir / ".build" / "net.yaml"
    assert "ffat-prod-sg-admin-ssh" in path.read_text(encoding="utf-8")


def test_plan_of_a_missing_stack_is_a_creation(deploy_dir: Path) -> None:
    runner = FakeRunner().on([*OS, "stack", "show"], NOT_FOUND)
    session = _session(deploy_dir, runner)
    planned = infra.plan(session, _spec(session))
    assert planned.existing is None
    assert "net" in planned.changes.added and not planned.changes.updated


def test_apply_creates_a_missing_stack(deploy_dir: Path) -> None:
    runner = (
        FakeRunner()
        .on([*OS, "stack", "show"], NOT_FOUND, times=1)
        .on([*OS, "stack", "create"], CommandResult())
        .on([*OS, "stack", "show"], _show())
    )
    session = _session(deploy_dir, runner)
    final = infra.apply(session, _spec(session))
    assert final is not None and final.healthy
    assert _verbs(runner) == ["stack show", "stack create", "stack show"]


def test_apply_updates_only_after_a_preview(deploy_dir: Path) -> None:
    runner = (
        FakeRunner()
        .on([*OS, "stack", "update", "--dry-run"], _preview(updated=["sg_admin_ssh"]))
        .on([*OS, "stack", "update", "--wait"], CommandResult())
        .on([*OS, "stack", "show"], _show("UPDATE_COMPLETE"))
    )
    session = _session(deploy_dir, runner)
    infra.apply(session, _spec(session))
    assert _verbs(runner) == ["stack show", "stack update --dry-run", "stack update", "stack show"]


def test_up_to_date_stack_is_left_alone(deploy_dir: Path) -> None:
    runner = (
        FakeRunner()
        .on([*OS, "stack", "show"], _show())
        .on([*OS, "stack", "update", "--dry-run"], _preview(unchanged=["net"]))
    )
    session = _session(deploy_dir, runner)
    infra.apply(session, _spec(session))
    assert "stack update" not in _verbs(runner)


def test_guard_refuses_to_replace_the_network(deploy_dir: Path) -> None:
    runner = (
        FakeRunner()
        .on([*OS, "stack", "show"], _show())
        .on([*OS, "stack", "update", "--dry-run"], _preview(replaced=["subnet"]))
    )
    session = _session(deploy_dir, runner)
    with pytest.raises(infra.InfraError, match="--allow-replace subnet"):
        infra.apply(session, _spec(session))
    assert "stack update" not in _verbs(runner)


def test_only_refuses_unrelated_pending_changes(deploy_dir: Path) -> None:
    runner = (
        FakeRunner()
        .on([*OS, "stack", "show"], _show())
        .on([*OS, "stack", "update", "--dry-run"], _preview(updated=["sg_admin_ssh", "router"]))
    )
    session = _session(deploy_dir, runner)
    with pytest.raises(infra.InfraError, match="other pending changes \\(router"):
        infra.apply(session, _spec(session), only={"sg_admin_ssh"})


def test_operator_refusal_cancels(deploy_dir: Path) -> None:
    runner = FakeRunner().on([*OS, "stack", "show"], NOT_FOUND)
    session = _session(deploy_dir, runner, answer=False)
    with pytest.raises(infra.InfraError, match="cancelled"):
        infra.apply(session, _spec(session))
    assert "stack create" not in _verbs(runner)


def test_failed_stack_is_reported_with_its_reason(deploy_dir: Path) -> None:
    runner = (
        FakeRunner()
        .on([*OS, "stack", "show"], NOT_FOUND, times=1)
        .on([*OS, "stack", "create"], CommandResult())
        .on([*OS, "stack", "show"], _show("CREATE_FAILED", "Quota exceeded for routers"))
    )
    session = _session(deploy_dir, runner)
    with pytest.raises(infra.InfraError, match="Quota exceeded"):
        infra.apply(session, _spec(session))


def test_describe_changes() -> None:
    assert infra.describe_changes(ChangeSet()) == ["no change"]
    lines = infra.describe_changes(ChangeSet(added=("a",), replaced=("r",)))
    assert lines == ["+ added: a", "! replaced: r"]


def test_failed_create_reports_heats_reason(deploy_dir: Path) -> None:
    internal = CommandResult(returncode=1, stderr="ERROR: Internal Error")
    runner = (
        FakeRunner()
        .on([*OS, "stack", "show"], NOT_FOUND, times=1)
        .on([*OS, "stack", "create"], internal)
        .on([*OS, "stack", "show"], _show("CREATE_FAILED", "Resource CREATE failed: Forbidden"))
    )
    session = _session(deploy_dir, runner)
    with pytest.raises(infra.InfraError, match="CREATE_FAILED: Resource CREATE failed: Forbidden"):
        infra.apply(session, _spec(session))


def test_refused_create_says_nothing_was_created(deploy_dir: Path) -> None:
    internal = CommandResult(returncode=1, stderr="ERROR: Internal Error")
    runner = (
        FakeRunner().on([*OS, "stack", "show"], NOT_FOUND).on([*OS, "stack", "create"], internal)
    )
    session = _session(deploy_dir, runner)
    with pytest.raises(infra.InfraError, match="was not created"):
        infra.apply(session, _spec(session))
