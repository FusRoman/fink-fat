"""Tests of the CLI commands (no external command is really run)."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from finkfat_ops.cli import AppContext, app
from finkfat_ops.config import DeployPaths, load_deployment
from finkfat_ops.heat.net import build_net_template
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


# ------------------------------------------------------------------- infra and access


OS = ["openstack", "--os-cloud", "test-cloud"]
TOKEN = CommandResult(stdout=json.dumps({"project_id": "p1"}))
PROJECT = CommandResult(stdout=json.dumps({"id": "p1", "name": "test-project"}))
SHOW = CommandResult(
    stdout=json.dumps({"stack_name": "ffat-prod-net", "stack_status": "UPDATE_COMPLETE"})
)


def _cloud_runner() -> FakeRunner:
    return (
        FakeRunner()
        .on([*OS, "token"], TOKEN)
        .on([*OS, "project", "show"], PROJECT)
        .on(["curl"], CommandResult(stdout="8.8.8.8\n"))
    )


def _run(deploy_dir: Path, runner: FakeRunner, *args: str):
    context = AppContext(
        paths=DeployPaths(deploy_dir),
        runner=runner,
        openstack_binary="openstack",
        confirm=lambda _: True,
        prompt=lambda _: "",
    )
    return CliRunner().invoke(app, list(args), obj=context)


def test_password_profile_asks_once_and_passes_it_to_every_command(
    deploy_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OS_PASSWORD", raising=False)
    local = yaml.safe_load((deploy_dir / "infra.local.yaml").read_text(encoding="utf-8"))
    local["openstack"]["ask_password"] = True
    (deploy_dir / "infra.local.yaml").write_text(yaml.safe_dump(local), encoding="utf-8")
    asked: list[str] = []
    context = AppContext(
        paths=DeployPaths(deploy_dir),
        runner=_cloud_runner().on([*OS, "stack", "show"], SHOW),
        openstack_binary="openstack",
        ask_secret=lambda question: asked.append(question) or "s3cret",
    )
    result = CliRunner().invoke(app, ["infra", "show"], obj=context)
    assert result.exit_code == 0, result.output
    assert len(asked) == 1
    assert context.runner.calls
    assert all(c.env == (("OS_PASSWORD", "s3cret"),) for c in context.runner.calls)
    assert "s3cret" not in result.output


def test_infra_render_needs_no_cloud(deploy_dir: Path) -> None:
    result = _run(deploy_dir, FakeRunner(), "infra", "render", "net")
    assert result.exit_code == 0, result.output
    assert (deploy_dir / ".build" / "net.yaml").exists()


def test_wrong_project_stops_everything(deploy_dir: Path) -> None:
    other = CommandResult(stdout=json.dumps({"id": "p1", "name": "someone-else"}))
    runner = FakeRunner().on([*OS, "token"], TOKEN).on([*OS, "project", "show"], other)
    result = _run(deploy_dir, runner, "infra", "plan", "net")
    assert result.exit_code == 1
    assert "someone-else" in result.output
    assert not any("stack" in c.argv for c in runner.calls)


def _deployed_net(deploy_dir: Path, *changed: str) -> CommandResult:
    """The net template as deployed before allow-me (no dynamic range yet)."""
    deployment = load_deployment(DeployPaths(deploy_dir))
    template = build_net_template(deployment.infra, deployment.local.admin_cidrs)
    for name in changed:
        template["resources"][name]["properties"]["description"] = "older definition"
    return CommandResult(stdout=json.dumps(template))


def test_allow_me_updates_only_the_ssh_rules_and_records_the_ip(deploy_dir: Path) -> None:
    preview = CommandResult(stdout=json.dumps({"updated": [{"resource_name": "sg_admin_ssh"}]}))
    runner = (
        _cloud_runner()
        .on([*OS, "stack", "template", "show"], _deployed_net(deploy_dir))
        .on([*OS, "stack", "show"], SHOW)
        .on([*OS, "stack", "update", "--dry-run"], preview)
        .on([*OS, "stack", "update", "--wait"], CommandResult())
    )
    result = _run(deploy_dir, runner, "access", "allow-me", "--yes")
    assert result.exit_code == 0, result.output
    rendered = (deploy_dir / ".build" / "net.yaml").read_text(encoding="utf-8")
    assert "8.8.8.8/32" in rendered
    state = yaml.safe_load((deploy_dir / ".state" / "access.yaml").read_text(encoding="utf-8"))
    assert state == {"dynamic_cidrs": ["8.8.8.8/32"]}


def test_allow_me_keeps_state_unchanged_when_apply_is_refused(deploy_dir: Path) -> None:
    preview = CommandResult(
        stdout=json.dumps(
            {"updated": [{"resource_name": "sg_admin_ssh"}, {"resource_name": "router"}]}
        )
    )
    runner = (
        _cloud_runner()
        .on([*OS, "stack", "template", "show"], _deployed_net(deploy_dir, "router"))
        .on([*OS, "stack", "show"], SHOW)
        .on([*OS, "stack", "update", "--dry-run"], preview)
    )
    result = _run(deploy_dir, runner, "access", "allow-me", "--yes")
    assert result.exit_code == 1
    assert "other pending changes" in result.output
    assert not (deploy_dir / ".state" / "access.yaml").exists()


def test_destroy_requires_the_exact_name(deploy_dir: Path) -> None:
    runner = _cloud_runner()
    result = _run(deploy_dir, runner, "infra", "destroy", "net")
    assert result.exit_code == 1 and "cancelled" in result.output
    assert not any("delete" in c.argv for c in runner.calls)


# ----------------------------------------------------------------------------- volume

STORAGE_SHOW = CommandResult(
    stdout=json.dumps(
        {
            "stack_name": "ffat-prod-storage",
            "stack_status": "CREATE_COMPLETE",
            "parameters": {"vol_pgdata_existing_id": "", "vol_engine_existing_id": ""},
            "outputs": [{"output_key": "vol_pgdata_id", "output_value": "v1"}],
        }
    )
)
PGDATA = CommandResult(
    stdout=json.dumps(
        [{"ID": "v1", "Name": "ffat-prod-vol-pgdata", "Status": "in-use", "Size": 500}]
    )
)


def test_volume_list_shows_state_and_management(deploy_dir: Path) -> None:
    runner = (
        _cloud_runner()
        .on([*OS, "stack", "show"], STORAGE_SHOW)
        .on([*OS, "volume", "list", "--name", "ffat-prod-vol-pgdata"], PGDATA)
        .on([*OS, "volume", "list"], CommandResult(stdout="[]"))
    )
    result = _run(deploy_dir, runner, "volume", "list")
    assert result.exit_code == 0, result.output
    assert "created" in result.output and "orphan" not in result.output
    assert "absent" in result.output  # the engine volume does not exist yet


def test_volume_snapshot_targets_the_role_volume(deploy_dir: Path) -> None:
    snapshot = CommandResult(stdout=json.dumps({"id": "s1", "status": "creating"}))
    runner = (
        _cloud_runner()
        .on([*OS, "volume", "list"], PGDATA)
        .on([*OS, "volume", "snapshot", "create"], snapshot)
    )
    result = _run(deploy_dir, runner, "volume", "snapshot", "db")
    assert result.exit_code == 0, result.output
    create = runner.calls[-1].argv
    assert create[create.index("--volume") + 1] == "v1"
    assert "s1" in result.output
    assert "openstack --os-cloud test-cloud volume snapshot show" in result.output


def test_volume_snapshot_of_a_role_without_volume_fails(deploy_dir: Path) -> None:
    result = _run(deploy_dir, _cloud_runner(), "volume", "snapshot", "explorer")
    assert result.exit_code == 1 and "no data volume" in result.output


def test_destroy_storage_reminds_that_volumes_are_kept(deploy_dir: Path) -> None:
    runner = _cloud_runner().on([*OS, "stack", "delete"], CommandResult())
    context = AppContext(
        paths=DeployPaths(deploy_dir),
        runner=runner,
        openstack_binary="openstack",
        prompt=lambda _: "ffat-prod-storage",
    )
    result = CliRunner().invoke(app, ["infra", "destroy", "storage"], obj=context)
    assert result.exit_code == 0, result.output
    assert "data volumes are kept" in result.output


# ------------------------------------------------------------------------ ssh, init, vm


def _servers_show(ip: str = "157.136.0.9") -> CommandResult:
    outputs = {"explorer_public_ip": ip, "server_db_id": "s1"}
    data = {
        "stack_name": "ffat-prod-servers",
        "stack_status": "CREATE_COMPLETE",
        "outputs": [{"output_key": k, "output_value": v} for k, v in outputs.items()],
    }
    return CommandResult(stdout=json.dumps(data))


def test_ssh_writes_its_config_once_then_runs_ssh(deploy_dir: Path) -> None:
    runner = (
        _cloud_runner()
        .on([*OS, "stack", "show"], _servers_show())
        .on(["ssh"], CommandResult(stdout="ok\n"))
    )
    result = _run(deploy_dir, runner, "ssh", "db", "--", "docker", "ps")
    assert result.exit_code == 0, result.output
    ssh = runner.calls[-1].argv
    assert ssh[-3:] == ("ffat-db", "docker", "ps") and "BatchMode=yes" in ssh
    config = deploy_dir / ".state" / "ssh_config"
    assert "HostName 157.136.0.9" in config.read_text(encoding="utf-8")
    second = FakeRunner().on(["ssh"], CommandResult())
    assert _run(deploy_dir, second, "ssh", "db").exit_code == 0
    assert [c.argv[0] for c in second.calls] == ["ssh"]  # no cloud call: config reused
    assert second.calls[0].interactive


def test_ssh_without_servers_stack_says_what_to_do(deploy_dir: Path) -> None:
    missing = CommandResult(returncode=1, stderr="Stack not found")
    runner = _cloud_runner().on([*OS, "stack", "show"], missing)
    result = _run(deploy_dir, runner, "ssh", "db")
    assert result.exit_code == 1 and "infra apply servers" in result.output


def test_init_checks_the_bastion_first_and_stops_at_the_first_failure(deploy_dir: Path) -> None:
    state = deploy_dir / ".state"
    state.mkdir()
    (state / "ssh_config").write_text("Host ffat-explorer\n", encoding="utf-8")
    ssh = ["ssh", "-F", str(state / "ssh_config"), "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]
    runner = (
        FakeRunner()
        .on([*ssh, "ffat-db", "systemctl"], CommandResult(returncode=3, stderr="inactive"))
        .on([*ssh, "ffat-db", "timedatectl"], CommandResult(stdout="yes\n"))
        .on([*ssh, "ffat-explorer", "timedatectl"], CommandResult(stdout="yes\n"))
        .on(["ssh"], CommandResult())
    )
    result = _run(deploy_dir, runner, "init", "db", "explorer")
    assert result.exit_code == 1, result.output
    hosts = [c.argv[7] for c in runner.calls]
    assert hosts[0] == "ffat-explorer" and hosts.count("ffat-explorer") == 3
    db_commands = [c.argv[8] for c in runner.calls if c.argv[7] == "ffat-db"]
    assert db_commands == ["sudo", "systemctl"]  # cloud-init (with sudo), then docker fails
    assert "docker running: failed (exit status 3: inactive)" in result.output


def test_vm_list_and_power(deploy_dir: Path) -> None:
    servers = CommandResult(
        stdout=json.dumps([{"ID": "1", "Name": "ffat-prod-db", "Status": "ACTIVE"}])
    )
    runner = _cloud_runner().on([*OS, "server", "list"], servers)
    result = _run(deploy_dir, runner, "vm", "list")
    assert result.exit_code == 0 and "ffat-prod-db" in result.output
    runner = _cloud_runner().on([*OS, "server", "stop"], CommandResult())
    result = _run(deploy_dir, runner, "vm", "stop", "db")
    assert result.exit_code == 0, result.output
    assert runner.calls[-1].argv[-2:] == ("stop", "ffat-prod-db")


def test_vm_stop_can_be_refused(deploy_dir: Path) -> None:
    runner = _cloud_runner()
    context = AppContext(
        paths=DeployPaths(deploy_dir),
        runner=runner,
        openstack_binary="openstack",
        confirm=lambda _: False,
    )
    result = CliRunner().invoke(app, ["vm", "stop", "db"], obj=context)
    assert result.exit_code == 1 and "cancelled" in result.output
    assert not any("stop" in c.argv for c in runner.calls)


# ------------------------------------------------------------------- plan / apply defaults


def test_apply_without_stack_only_shows_the_help(deploy_dir: Path) -> None:
    runner = FakeRunner()
    result = _run(deploy_dir, runner, "infra", "apply")
    assert result.exit_code == 0, result.output
    assert "The stack must be named" in result.output
    assert runner.calls == []


def test_plan_without_stack_previews_all_three_in_order(deploy_dir: Path) -> None:
    missing = CommandResult(returncode=1, stderr="Stack not found")
    runner = (
        _cloud_runner()
        .on([*OS, "stack", "show"], missing)
        .on([*OS, "volume", "list"], CommandResult(stdout="[]"))
    )
    result = _run(deploy_dir, runner, "infra", "plan")
    assert result.exit_code == 1, result.output  # servers cannot be previewed yet
    output = result.output
    assert output.index("ffat-prod-net: will be created") < output.index(
        "ffat-prod-storage: will be created"
    )
    assert "ffat-prod-servers: ✘ refused: ffat-prod-net does not exist" in output


def test_render_without_stack_only_shows_the_help(deploy_dir: Path) -> None:
    result = _run(deploy_dir, FakeRunner(), "infra", "render")
    assert result.exit_code == 0, result.output
    assert "The stack must be named" in result.output
    assert not (deploy_dir / ".build").exists()


def test_ssh_command_shows_the_remote_errors(deploy_dir: Path) -> None:
    state = deploy_dir / ".state"
    state.mkdir()
    (state / "ssh_config").write_text("Host ffat-db\n", encoding="utf-8")
    runner = FakeRunner().on(["ssh"], CommandResult(1, stdout="partial\n", stderr="denied\n"))
    result = _run(deploy_dir, runner, "ssh", "db", "--", "sudo", "cat", "/x")
    assert result.exit_code == 1
    assert "partial" in result.output and "denied" in result.output
