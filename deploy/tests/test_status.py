"""Tests of finkfat_ops.status (parsers, report, collection)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from finkfat_ops.checks import Level
from finkfat_ops.config import DeployPaths, load_deployment
from finkfat_ops.deploy import DeployRecord
from finkfat_ops.naming import Role
from finkfat_ops.runner import CommandError, CommandResult, FakeRunner
from finkfat_ops.status import (
    SECTION_MARK,
    Expected,
    RoleStatus,
    ServiceState,
    SyncState,
    build_report,
    collect_role,
    parse_certificate_expiry,
    parse_disk_percent,
    parse_sections,
    parse_services,
    parse_sync,
    status_command,
    status_script,
    vm_states,
)
from finkfat_ops.vm import ServerInfo

NOW = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)
CONFIG = Path("/d/.state/ssh_config")
PS_LINES = (
    '{"Service":"stream","State":"running","Health":"","Status":"Up 3 days"}\n'
    '{"Service":"app","State":"restarting","Health":"","Status":"Restarting (1) 5 s ago"}\n'
)
DF = (
    "Filesystem     1024-blocks     Used Available Capacity Mounted on\n"
    "/dev/vdb        205314024 84000000 110000000      43% /srv/finkfat\n"
)
RECORD = '{"role": "compute", "tag": "v1.0.0", "config": "abc", "deployed_at": "2026-11-26"}'


def _output(**sections: str) -> str:
    return "".join(f"{SECTION_MARK}{name}\n{text}\n" for name, text in sections.items())


def test_sections_and_parsers() -> None:
    sections = parse_sections(_output(record=RECORD, ps=PS_LINES, df=DF))
    assert set(sections) == {"record", "ps", "df"}
    services = parse_services(sections["ps"])
    assert [s.name for s in services] == ["app", "stream"]
    assert not services[0].ok and services[1].ok
    assert parse_disk_percent(DF) == 43
    assert parse_disk_percent("") is None


def test_services_as_json_array() -> None:
    array = json.dumps([{"Service": "postgres", "State": "running", "Health": "unhealthy"}])
    (postgres,) = parse_services(array)
    assert postgres.health == "unhealthy" and not postgres.ok
    assert parse_services("not json") == []


def test_sync_and_certificate() -> None:
    sync = parse_sync("61001|2026-11-27 03:00:00.5+00\n")
    assert sync == SyncState(61001, datetime(2026, 11, 27, 3, 0, 0, 500000, tzinfo=UTC))
    assert parse_sync("") is None
    verbose = "*  start date: Nov  1 00:00:00 2026 GMT\n*  expire date: Jan 30 23:59:59 2027 GMT\n"
    assert parse_certificate_expiry(verbose) == datetime(2027, 1, 30, 23, 59, 59, tzinfo=UTC)
    assert parse_certificate_expiry("no tls") is None


def test_script_reads_the_sync_state_on_the_db_only(deploy_dir: Path) -> None:
    infra = load_deployment(DeployPaths(deploy_dir)).infra
    assert "fink_fat_sync_state" in status_script(Role.DB, infra)
    assert "df -P /srv/pgdata" in status_script(Role.DB, infra)
    assert "fink_fat_sync_state" not in status_script(Role.COMPUTE, infra)
    assert "df -P /;" in status_script(Role.EXPLORER, infra)


def test_vm_states(deploy_dir: Path) -> None:
    infra = load_deployment(DeployPaths(deploy_dir)).infra
    states = vm_states(infra, [ServerInfo("1", "ffat-prod-db", "SHUTOFF")])
    assert states[Role.DB] == "SHUTOFF"
    assert states[Role.COMPUTE] == "absent"


def _healthy(role: Role, tag: str = "v1.0.0", config: str = "abc") -> RoleStatus:
    return RoleStatus(
        role=role,
        vm_status="ACTIVE",
        services=[ServiceState("svc", "running", status="Up")],
        record=DeployRecord(tag, config, "2026-11-26"),
        disk_percent=40,
    )


def _expected(**fingerprints: str) -> Expected:
    prints = {Role(k): v for k, v in fingerprints.items()}
    return Expected("v1.0.0", prints, "fink-fat.example.org")


def test_healthy_instance_has_no_finding() -> None:
    statuses = [_healthy(role) for role in (Role.DB, Role.COMPUTE, Role.EXPLORER)]
    report = build_report(
        statuses,
        _expected(db="abc", compute="abc", explorer="abc"),
        sync=SyncState(61001, NOW - timedelta(hours=9)),
        certificate_expiry=NOW + timedelta(days=60),
        now=NOW,
    )
    assert report.findings == [] and report.healthy and report.drifted == []


def test_problems_are_reported() -> None:
    statuses = [
        RoleStatus(Role.DB, "SHUTOFF"),
        _healthy(Role.COMPUTE, tag="v0.9.0", config="old"),
        RoleStatus(Role.EXPLORER, "ACTIVE", error="Connection timed out"),
    ]
    report = build_report(
        statuses,
        _expected(compute="new"),
        sync=None,
        certificate_expiry=NOW + timedelta(days=5),
        now=NOW,
    )
    messages = {f.message: f.level for f in report.findings}
    assert messages["db: VM SHUTOFF"] == Level.ERROR
    assert messages["explorer: Connection timed out"] == Level.ERROR
    assert messages["compute: runs v0.9.0, infra.yaml says v1.0.0"] == Level.WARN
    assert messages["compute: configuration differs from deploy/ (drift)"] == Level.WARN
    assert messages["certificate expires in 5 days"] == Level.WARN
    assert report.drifted == [Role.COMPUTE]
    assert not report.healthy


def test_container_down_disk_full_and_old_sync() -> None:
    db = RoleStatus(
        Role.DB,
        "ACTIVE",
        services=[ServiceState("postgres", "exited", status="Exited (1)")],
        record=DeployRecord("v1.0.0", "abc", "x"),
        disk_percent=91,
    )
    report = build_report(
        [db],
        Expected("v1.0.0", {}, None),
        sync=SyncState(61000, NOW - timedelta(days=3)),
        certificate_expiry=None,
        now=NOW,
    )
    messages = [f.message for f in report.findings]
    assert "db: postgres is exited (Exited (1))" in messages
    assert "db: disk 91% full" in messages
    assert "last nightly sync is 3 d old" in messages


def test_never_deployed_and_missing_certificate() -> None:
    report = build_report(
        [RoleStatus(Role.EXPLORER, "ACTIVE")],
        _expected(),
        sync=None,
        certificate_expiry=None,
        now=NOW,
    )
    messages = {f.message: f.level for f in report.findings}
    assert messages["explorer: never deployed"] == Level.WARN
    assert messages["https://fink-fat.example.org/: no valid certificate"] == Level.ERROR


def test_collect_role(deploy_dir: Path) -> None:
    infra = load_deployment(DeployPaths(deploy_dir)).infra
    command = status_command(CONFIG, Role.COMPUTE, infra)
    output = _output(record=RECORD, ps=PS_LINES, df=DF)
    runner = FakeRunner().on(command.argv, CommandResult(stdout=output))
    status, raw = collect_role(runner, CONFIG, Role.COMPUTE, infra, "ACTIVE")
    assert status.record and status.record.tag == "v1.0.0"
    assert status.disk_percent == 43 and raw == output

    failing = FakeRunner().on(command.argv, CommandResult(returncode=255, stderr="timed out\n"))
    status, _ = collect_role(failing, CONFIG, Role.COMPUTE, infra, "ACTIVE")
    assert status.error == "timed out"

    class Raising:
        def run(self, command, *, check=True):
            raise CommandError(command, "program not found: ssh")

    status, _ = collect_role(Raising(), CONFIG, Role.COMPUTE, infra, "ACTIVE")
    assert status.error and "program not found" in status.error

    status, _ = collect_role(FakeRunner(), CONFIG, Role.COMPUTE, infra, "SHUTOFF")
    assert status.vm_status == "SHUTOFF"  # no SSH attempted
