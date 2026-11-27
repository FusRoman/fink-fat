"""Tests of the CLI commands (no external command is really run)."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from finkfat_ops.cli import AppContext, app
from finkfat_ops.config import DeployPaths, load_deployment
from finkfat_ops.deploy import health_checks
from finkfat_ops.heat.net import build_net_template
from finkfat_ops.naming import Role
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
        .on(["curl", "-4"], CommandResult(stdout="8.8.8.8\n"))
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


# ------------------------------------------------------------------------------ images

VERSIONS = CommandResult(
    stdout=json.dumps(
        [{"name": "sha256:" + "a" * 64, "created_at": "2026-11-24T10:00:00Z",
          "metadata": {"container": {"tags": ["sha-2222222", "main"]}}}]
    )
)  # fmt: skip


def test_images_list_shows_the_published_tags(deploy_dir: Path) -> None:
    runner = FakeRunner().on(["gh", "api"], VERSIONS)
    result = _run(deploy_dir, runner, "images", "list", "engine")
    assert result.exit_code == 0, result.output
    assert "sha-2222222, main" in result.output
    assert "fink-fat-engine/versions" in runner.calls[0].argv[2]


def test_images_list_explains_a_missing_scope(deploy_dir: Path) -> None:
    denied = CommandResult(1, stderr="HTTP 403: You need at least read:packages scope")
    result = _run(deploy_dir, FakeRunner().on(["gh"], denied), "images", "list")
    assert result.exit_code == 1 and "gh auth refresh -s read:packages" in result.output


def test_images_push_refuses_a_dirty_tree(deploy_dir: Path) -> None:
    runner = FakeRunner().on(["git", "-C"], CommandResult(stdout=" M src/main.rs\n"))
    result = _run(deploy_dir, runner, "images", "build", "--push", "--allow-dirty")
    assert result.exit_code == 1 and "uncommitted changes" in result.output
    assert not any(c.argv[0] == "docker" for c in runner.calls)


def test_images_build_tags_with_the_commit(deploy_dir: Path) -> None:
    runner = (
        FakeRunner()
        .on(["git", "-C", str(deploy_dir.parent), "status"], CommandResult(stdout=""))
        .on(["git", "-C", str(deploy_dir.parent), "rev-parse"], CommandResult(stdout="abc1234ff\n"))
        .on(["docker"], CommandResult())
    )
    result = _run(deploy_dir, runner, "images", "build", "engine", "--push")
    assert result.exit_code == 0, result.output
    docker = [c for c in runner.calls if c.argv[0] == "docker"]
    assert len(docker) == 1
    assert "ghcr.io/fusroman/fink-fat-engine:sha-abc1234" in docker[0].argv
    assert "--push" in docker[0].argv


def test_images_list_reports_unpublished_images_and_goes_on(deploy_dir: Path) -> None:
    missing = CommandResult(1, stderr="gh: Package not found. (HTTP 404)")
    runner = FakeRunner().on(["gh", "api"], missing, times=1).on(["gh", "api"], VERSIONS)
    result = _run(deploy_dir, runner, "images", "list")
    assert result.exit_code == 0, result.output
    assert "fink-fat-engine: not published yet" in result.output
    assert "sha-2222222, main" in result.output  # the explorer is still listed


# ------------------------------------------------- deploy, logs, tunnel, status, doctor


def _state(deploy_dir: Path) -> Path:
    state = deploy_dir / ".state"
    state.mkdir(exist_ok=True)
    config = state / "ssh_config"
    config.write_text("Host ffat-explorer\n", encoding="utf-8")
    return config


MEMINFO = CommandResult(stdout="MemTotal:       32763204 kB\n")


def test_deploy_runs_the_roles_in_order(deploy_dir: Path) -> None:
    _state(deploy_dir)
    runner = (
        FakeRunner()
        .on(
            [
                "ssh",
                "-F",
                str(deploy_dir / ".state/ssh_config"),
                "-o",
                "BatchMode=yes",
                "-o",
                "LogLevel=ERROR",
                "ffat-db",
                "cat /proc/meminfo",
            ],
            MEMINFO,
        )
        .on(["ssh"], CommandResult())
        .on(["curl"], CommandResult())
    )
    result = _run(deploy_dir, runner, "deploy", "explorer", "db")
    assert result.exit_code == 0, result.output
    hosts = [c.argv[7] for c in runner.calls if c.argv[0] == "ssh"]
    assert hosts.index("ffat-explorer") > max(i for i, h in enumerate(hosts) if h == "ffat-db")
    assert "ffat-compute" not in hosts
    assert "with images tagged v1.0.0" in result.output


def test_deploy_stops_at_the_first_unhealthy_role(deploy_dir: Path) -> None:
    _state(deploy_dir)
    runner = FakeRunner().on(["ssh"], CommandResult(returncode=1, stdout="MemTotal: 1 kB"))
    result = _run(deploy_dir, runner, "deploy", "--tag", "v9.9.9")
    assert result.exit_code == 1
    assert all(c.argv[7] == "ffat-db" for c in runner.calls)


def test_deploy_unhealthy_db_stops_before_compute(deploy_dir: Path) -> None:
    config = _state(deploy_dir)
    ssh = ["ssh", "-F", str(config), "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "ffat-db"]
    (ready,) = health_checks(Role.DB)
    runner = (
        FakeRunner()
        .on([*ssh, "cat /proc/meminfo"], MEMINFO)
        .on([*ssh, *ready.command], CommandResult(returncode=1))
        .on(["ssh"], CommandResult())
    )
    result = _run(deploy_dir, runner, "deploy")
    assert result.exit_code == 1
    assert "db is unhealthy: deployment stopped" in result.output
    assert {c.argv[7] for c in runner.calls} == {"ffat-db"}
    assert not any("deployed.json" in c.argv[-1] for c in runner.calls)


def test_logs_and_tunnel_are_interactive(deploy_dir: Path) -> None:
    _state(deploy_dir)
    runner = FakeRunner().on(["ssh"], CommandResult())
    assert _run(deploy_dir, runner, "logs", "compute", "stream", "-f").exit_code == 0
    assert runner.calls[-1].interactive
    assert runner.calls[-1].argv[-1].endswith("docker compose logs --tail 200 --follow stream")
    result = _run(deploy_dir, runner, "tunnel", "pgadmin")
    assert result.exit_code == 0 and "localhost:5050" in result.output
    assert "-L" in runner.calls[-1].argv and "5050:127.0.0.1:5050" in runner.calls[-1].argv


def test_status_reports_and_fails_on_errors(deploy_dir: Path) -> None:
    _state(deploy_dir)
    servers = CommandResult(
        stdout=json.dumps(
            [
                {"ID": "1", "Name": "ffat-prod-db", "Status": "SHUTOFF"},
                {"ID": "2", "Name": "ffat-prod-compute", "Status": "ACTIVE"},
                {"ID": "3", "Name": "ffat-prod-explorer", "Status": "ACTIVE"},
            ]
        )
    )
    runner = (
        _cloud_runner()
        .on([*OS, "server", "list"], servers)
        .on(["ssh"], CommandResult(stdout=""))
        .on(
            ["curl", "--silent", "--verbose"],
            CommandResult(stderr="*  expire date: Jan 30 23:59:59 2099 GMT"),
        )
    )
    result = _run(deploy_dir, runner, "status")
    assert result.exit_code == 1, result.output
    assert "db: VM SHUTOFF" in result.output
    assert "compute: never deployed" in result.output
    as_json = _run(deploy_dir, runner, "status", "--json")
    assert '"vm_status": "SHUTOFF"' in as_json.output


def test_doctor_lists_every_problem(deploy_dir: Path) -> None:
    catalog = CommandResult(
        stdout=json.dumps(
            [
                {
                    "Name": "heat",
                    "Type": "orchestration",
                    "Endpoints": [{"interface": "public", "url": "https://heat.example.org:8004"}],
                },
                {
                    "Name": "nova",
                    "Type": "compute",
                    "Endpoints": "region\n  public: https://nova.example.org:8774\n",
                },
            ]
        )
    )
    runner = (
        _cloud_runner()
        .on(["git"], CommandResult())
        .on(["openstack", "stack", "list", "--help"], CommandResult())
        .on(["ssh", "-V"], CommandResult(stderr="OpenSSH_9.6"))
        .on(["gh"], CommandResult(returncode=1, stderr="You are not logged into any GitHub hosts"))
        .on([*OS, "catalog", "list"], catalog)
        .on(
            [
                "curl",
                "--silent",
                "--output",
                "/dev/null",
                "--connect-timeout",
                "5",
                "--max-time",
                "10",
                "https://heat.example.org:8004",
            ],
            CommandResult(returncode=28),
        )
        .on(["curl", "--silent", "--output"], CommandResult())
    )
    result = _run(deploy_dir, runner, "doctor")
    assert result.exit_code == 1, result.output
    assert "Heat unreachable" in result.output and "start sshuttle" in result.output
    assert "Nova reachable" in result.output
    assert "gh logged in" in result.output and "gh auth login" in result.output
    assert "public IP 8.8.8.8 is not in the allowed SSH ranges" in result.output


def test_deploy_checks_every_file_before_touching_a_machine(deploy_dir: Path) -> None:
    _state(deploy_dir)
    (deploy_dir / "config" / "stream.yaml").unlink()
    runner = FakeRunner().on(["ssh"], CommandResult())
    result = _run(deploy_dir, runner, "deploy")
    assert result.exit_code == 1
    assert "copy stream.example.yaml and fill it in" in result.output
    assert runner.calls == []  # not even the db was redeployed
    db_only = FakeRunner().on(["ssh"], MEMINFO)  # MemTotal for the db, ignored elsewhere
    result = _run(deploy_dir, db_only, "deploy", "db")
    assert result.exit_code == 0, result.output  # the db does not need stream.yaml


def test_config_check_reports_a_missing_stream_config(deploy_dir: Path) -> None:
    (deploy_dir / "config" / "stream.yaml").unlink()
    result = _invoke(deploy_dir)
    assert result.exit_code == 1
    assert "config/stream.yaml is missing" in result.output
    assert "cp " in result.output and "stream.example.yaml" in result.output
