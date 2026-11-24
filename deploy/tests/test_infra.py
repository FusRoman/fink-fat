"""Tests of finkfat_ops.infra with a fake cloud (no real command is run)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from finkfat_ops import infra
from finkfat_ops.access import AccessState
from finkfat_ops.config import DeployPaths, load_deployment
from finkfat_ops.naming import Role, Stack
from finkfat_ops.openstack import OpenStackCli
from finkfat_ops.runner import CommandResult, FakeRunner
from finkfat_ops.stack import ChangeSet, StackInfo

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


def _deployed(session: infra.InfraSession, *changed: str) -> CommandResult:
    """The net template as deployed, where the ``changed`` resources differ from now."""
    template = copy.deepcopy(_spec(session).template)
    for name in changed:
        template["resources"][name]["properties"]["description"] = "older definition"
    return CommandResult(stdout=json.dumps(template))


TEMPLATE_SHOW = [*OS, "stack", "template", "show"]


def _verbs(runner: FakeRunner) -> list[str]:
    return [
        " ".join(c.argv[3:5]) + (" --dry-run" if "--dry-run" in c.argv else "")
        for c in runner.calls
    ]


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
    runner = FakeRunner()
    session = _session(deploy_dir, runner)
    (
        runner.on(TEMPLATE_SHOW, _deployed(session, "sg_admin_ssh"))
        .on([*OS, "stack", "update", "--dry-run"], _preview(updated=["sg_admin_ssh"]))
        .on([*OS, "stack", "update", "--wait"], CommandResult())
        .on([*OS, "stack", "show"], _show("UPDATE_COMPLETE"))
    )
    infra.apply(session, _spec(session))
    assert _verbs(runner) == [
        "stack show",
        "stack template",
        "stack update --dry-run",
        "stack update",
        "stack show",
    ]


def test_up_to_date_stack_is_left_alone_without_asking_heat(deploy_dir: Path) -> None:
    runner = FakeRunner()
    session = _session(deploy_dir, runner)
    runner.on([*OS, "stack", "show"], _show()).on(TEMPLATE_SHOW, _deployed(session))
    infra.apply(session, _spec(session))
    assert _verbs(runner) == ["stack show", "stack template"]


def test_preview_false_alarms_on_references_are_ignored(deploy_dir: Path) -> None:
    """Heat flags every resource holding a get_resource; only sg_admin_ssh really changes."""
    runner = FakeRunner()
    session = _session(deploy_dir, runner)
    preview = _preview(
        updated=["sg_admin_ssh", "sg_db", "sg_internal_ssh"],
        replaced=["router_interface", "subnet"],
    )
    (
        runner.on([*OS, "stack", "show"], _show())
        .on(TEMPLATE_SHOW, _deployed(session, "sg_admin_ssh"))
        .on([*OS, "stack", "update", "--dry-run"], preview)
    )
    planned = infra.plan(session, _spec(session))
    assert planned.changes.updated == ("sg_admin_ssh",)
    assert planned.changes.replaced == ()
    assert planned.ignored == ("router_interface", "sg_db", "sg_internal_ssh", "subnet")
    assert "ignored" in infra.describe_ignored(planned)[0]


SUBNET_SCHEMA = CommandResult(
    stdout=json.dumps(
        {
            "resource_type": "OS::Neutron::Subnet",
            "properties": {
                "cidr": {"update_allowed": False},
                "dns_nameservers": {"update_allowed": True},
                "network": {"update_allowed": False},
            },
        }
    )
)
TYPE_SHOW = [*OS, "orchestration", "resource", "type", "show"]


def _deployed_subnet(session: infra.InfraSession, **properties: Any) -> CommandResult:
    """The net template as deployed, with other values for some subnet properties."""
    template = copy.deepcopy(_spec(session).template)
    subnet = template["resources"]["subnet"]["properties"]
    for key, value in properties.items():
        if value is None:
            subnet.pop(key, None)
        else:
            subnet[key] = value
    return CommandResult(stdout=json.dumps(template))


def test_guard_refuses_to_replace_the_network(deploy_dir: Path) -> None:
    runner = FakeRunner()
    session = _session(deploy_dir, runner)
    (
        runner.on([*OS, "stack", "show"], _show())
        .on(TEMPLATE_SHOW, _deployed_subnet(session, cidr="10.43.0.0/24"))
        .on(TYPE_SHOW, SUBNET_SCHEMA)
        .on([*OS, "stack", "update", "--dry-run"], _preview(replaced=["subnet"]))
    )
    with pytest.raises(infra.InfraError, match="--allow-replace subnet"):
        infra.apply(session, _spec(session))
    assert "stack update" not in _verbs(runner)


def test_only_refuses_unrelated_pending_changes(deploy_dir: Path) -> None:
    runner = FakeRunner()
    session = _session(deploy_dir, runner)
    (
        runner.on([*OS, "stack", "show"], _show())
        .on(TEMPLATE_SHOW, _deployed(session, "sg_admin_ssh", "router"))
        .on([*OS, "stack", "update", "--dry-run"], _preview(updated=["sg_admin_ssh", "router"]))
    )
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


# -------------------------------------------------------------------------- storage

STORAGE_NOT_FOUND = CommandResult(returncode=1, stderr="Stack not found: ffat-prod-storage")


def _volumes(*entries: tuple[str, str]) -> CommandResult:
    return CommandResult(stdout=json.dumps([{"ID": i, "Name": n} for i, n in entries]))


def _storage_show(parameters: dict[str, str], outputs: dict[str, str]) -> CommandResult:
    data = {
        "stack_name": "ffat-prod-storage",
        "stack_status": "CREATE_COMPLETE",
        "parameters": parameters,
        "outputs": [{"output_key": k, "output_value": v} for k, v in outputs.items()],
    }
    return CommandResult(stdout=json.dumps(data))


def _volume_lists(runner: FakeRunner, pgdata: CommandResult) -> FakeRunner:
    return runner.on([*OS, "volume", "list", "--name", "ffat-prod-vol-engine"], _volumes()).on(
        [*OS, "volume", "list", "--name", "ffat-prod-vol-pgdata"], pgdata
    )


def test_storage_spec_without_cloud_creates_everything(deploy_dir: Path) -> None:
    session = _session(deploy_dir, FakeRunner())
    spec = infra.build_stack_spec(Stack.STORAGE, session.deployment, AccessState())
    assert spec.name == "ffat-prod-storage"
    assert spec.protected == {"vol_engine", "vol_pgdata"}
    assert spec.parameters == {"vol_engine_existing_id": "", "vol_pgdata_existing_id": ""}
    assert spec.adoptions == ()


def test_orphan_volume_is_adopted_on_creation(deploy_dir: Path) -> None:
    runner = _volume_lists(
        FakeRunner().on([*OS, "stack", "show"], STORAGE_NOT_FOUND),
        _volumes(("v1", "ffat-prod-vol-pgdata"), ("x", "ffat-prod-vol-pgdata-old")),
    )
    session = _session(deploy_dir, runner)
    spec = infra.resolve_spec(session, Stack.STORAGE, AccessState())
    assert spec.parameters["vol_pgdata_existing_id"] == "v1"
    assert infra.describe_adoptions(spec) == ["= adopted: ffat-prod-vol-pgdata (v1)"]
    planned = infra.plan(session, spec)
    assert planned.changes.added == ("vol_engine",)


def test_apply_storage_passes_the_adoption_parameters(deploy_dir: Path) -> None:
    runner = _volume_lists(
        FakeRunner()
        .on([*OS, "stack", "show"], STORAGE_NOT_FOUND, times=2)
        .on([*OS, "stack", "create"], CommandResult())
        .on([*OS, "stack", "show"], _storage_show({}, {})),
        _volumes(("v1", "ffat-prod-vol-pgdata")),
    )
    session = _session(deploy_dir, runner)
    infra.apply(session, infra.resolve_spec(session, Stack.STORAGE, AccessState()))
    create = next(c for c in runner.calls if "create" in c.argv)
    assert "vol_pgdata_existing_id=v1" in create.argv
    assert "vol_engine_existing_id=" in create.argv


def test_volume_created_by_the_stack_is_not_adopted(deploy_dir: Path) -> None:
    show = _storage_show({"vol_pgdata_existing_id": ""}, {"vol_pgdata_id": "v1"})
    runner = _volume_lists(
        FakeRunner().on([*OS, "stack", "show"], show), _volumes(("v1", "ffat-prod-vol-pgdata"))
    )
    spec = infra.resolve_spec(_session(deploy_dir, runner), Stack.STORAGE, AccessState())
    assert spec.adoptions == ()


def test_ambiguous_volumes_refuse_the_storage_stack(deploy_dir: Path) -> None:
    runner = _volume_lists(
        FakeRunner().on([*OS, "stack", "show"], STORAGE_NOT_FOUND),
        _volumes(("v1", "ffat-prod-vol-pgdata"), ("v2", "ffat-prod-vol-pgdata")),
    )
    with pytest.raises(infra.InfraError, match="refused: 2 volumes"):
        infra.resolve_spec(_session(deploy_dir, runner), Stack.STORAGE, AccessState())


def test_volume_of_role(deploy_dir: Path) -> None:
    deployment = _session(deploy_dir, FakeRunner()).deployment
    assert infra.volume_of_role(deployment, Role.DB).name == "ffat-prod-vol-pgdata"
    with pytest.raises(infra.InfraError, match="no data volume"):
        infra.volume_of_role(deployment, Role.EXPLORER)


def test_creation_without_resources_says_nothing_to_create() -> None:
    assert infra.describe_changes(ChangeSet(), creating=True) == ["nothing to create"]
    assert infra.describe_changes(ChangeSet()) == ["no change"]


# -------------------------------------------------------------------------- servers

NET_OUTPUTS = {
    "network_id": "n",
    "subnet_id": "s",
    "sg_explorer_id": "g1",
    "sg_admin_ssh_id": "g2",
    "sg_internal_ssh_id": "g3",
    "sg_db_id": "g4",
    "sg_compute_id": "g5",
    "explorer_floating_ip_id": "f1",
    "explorer_public_ip": "157.136.0.9",
}


def _stack(name: str, outputs: dict[str, str], status: str = "CREATE_COMPLETE") -> CommandResult:
    data = {
        "stack_name": name,
        "stack_status": status,
        "outputs": [{"output_key": k, "output_value": v} for k, v in outputs.items()],
    }
    return CommandResult(stdout=json.dumps(data))


def _upstream_runner(storage_status: str = "CREATE_COMPLETE") -> FakeRunner:
    storage = {"vol_engine_id": "ve", "vol_pgdata_id": "vp"}
    return (
        FakeRunner()
        .on([*OS, "stack", "show", "ffat-prod-net"], _stack("ffat-prod-net", NET_OUTPUTS))
        .on(
            [*OS, "stack", "show", "ffat-prod-storage"],
            _stack("ffat-prod-storage", storage, storage_status),
        )
    )


def test_servers_parameters_come_from_the_other_stacks(deploy_dir: Path) -> None:
    session = _session(deploy_dir, _upstream_runner())
    spec = infra.resolve_spec(session, Stack.SERVERS, AccessState())
    assert spec.parameters["network_id"] == "n"
    assert spec.parameters["vol_pgdata_id"] == "vp"
    assert "server_db" in spec.protected
    assert spec.parameters["explorer_floating_ip_id"] == "f1"


def test_servers_refused_while_storage_is_broken(deploy_dir: Path) -> None:
    session = _session(deploy_dir, _upstream_runner("UPDATE_FAILED"))
    with pytest.raises(infra.InfraError, match="ffat-prod-storage is UPDATE_FAILED"):
        infra.resolve_spec(session, Stack.SERVERS, AccessState())


def test_servers_refused_without_storage(deploy_dir: Path) -> None:
    runner = (
        FakeRunner()
        .on([*OS, "stack", "show", "ffat-prod-net"], _stack("ffat-prod-net", NET_OUTPUTS))
        .on([*OS, "stack", "show"], STORAGE_NOT_FOUND)
    )
    with pytest.raises(infra.InfraError, match="ffat-prod-storage does not exist"):
        infra.resolve_spec(_session(deploy_dir, runner), Stack.SERVERS, AccessState())


def test_ssh_access_is_written_and_recreated_hosts_forgotten(deploy_dir: Path) -> None:
    runner = FakeRunner().on(["ssh-keygen"], CommandResult())
    session = _session(deploy_dir, runner)
    state = deploy_dir / ".state"
    state.mkdir()
    (state / "known_hosts").write_text("", encoding="utf-8")
    before = stack_info({"server_db_id": "s1", "server_explorer_id": "e1"})
    after = stack_info(
        {"server_db_id": "s2", "server_explorer_id": "e1", "explorer_public_ip": "157.136.0.9"}
    )
    config = infra.write_ssh_access(session, before, after)
    assert config == state / "ssh_config"
    assert "HostName 157.136.0.9" in config.read_text(encoding="utf-8")
    assert [c.argv[-1] for c in runner.calls] == ["ffat-db"]


def test_no_ssh_access_without_public_address(deploy_dir: Path) -> None:
    session = _session(deploy_dir, FakeRunner())
    assert infra.write_ssh_access(session, None, stack_info({})) is None


def stack_info(outputs: dict[str, str]) -> StackInfo:
    return StackInfo("ffat-prod-servers", "CREATE_COMPLETE", "", outputs)


def test_local_dns_servers_reach_the_subnet(deploy_dir: Path) -> None:
    local = deploy_dir / "infra.local.yaml"
    data = yaml.safe_load(local.read_text(encoding="utf-8"))
    data["dns_nameservers"] = ["192.0.2.53"]
    local.write_text(yaml.safe_dump(data), encoding="utf-8")
    session = _session(deploy_dir, FakeRunner())
    subnet = _spec(session).template["resources"]["subnet"]["properties"]
    assert subnet["dns_nameservers"] == ["192.0.2.53"]


def test_false_replacement_of_an_updatable_property_is_an_update(deploy_dir: Path) -> None:
    """Adding DNS servers: Heat's preview says 'replaced', its schema says 'in place'."""
    local = deploy_dir / "infra.local.yaml"
    data = yaml.safe_load(local.read_text(encoding="utf-8"))
    data["dns_nameservers"] = ["192.0.2.53"]
    local.write_text(yaml.safe_dump(data), encoding="utf-8")
    runner = FakeRunner()
    session = _session(deploy_dir, runner)
    preview = _preview(
        updated=["sg_db", "sg_internal_ssh"], replaced=["router_interface", "subnet"]
    )
    (
        runner.on([*OS, "stack", "show"], _show())
        .on(TEMPLATE_SHOW, _deployed_subnet(session, dns_nameservers=None))
        .on(TYPE_SHOW, SUBNET_SCHEMA)
        .on([*OS, "stack", "update", "--dry-run"], preview)
    )
    planned = infra.plan(session, _spec(session))
    assert planned.changes.updated == ("subnet",) and planned.changes.replaced == ()
    assert planned.in_place == ("subnet",)
    assert planned.ignored == ("router_interface", "sg_db", "sg_internal_ssh")
    assert len(infra.describe_ignored(planned)) == 2
    assert stack_guard_is_clear(planned)


def stack_guard_is_clear(planned: infra.Plan) -> bool:
    return not infra.stack_ops.guard_violations(planned.changes, planned.spec.protected)
