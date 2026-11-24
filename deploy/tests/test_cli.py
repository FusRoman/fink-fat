"""Tests of the CLI commands (no external command is really run)."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from finkfat_ops.cli import AppContext, app
from finkfat_ops.config import DeployPaths
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


def test_allow_me_updates_only_the_ssh_rules_and_records_the_ip(deploy_dir: Path) -> None:
    preview = CommandResult(stdout=json.dumps({"updated": [{"resource_name": "sg_admin_ssh"}]}))
    runner = (
        _cloud_runner()
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
