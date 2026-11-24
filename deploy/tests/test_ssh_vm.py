"""Tests of finkfat_ops.ssh (SSH config, commands, init checks) and finkfat_ops.vm."""

from __future__ import annotations

import json
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any

from finkfat_ops.checks import Level
from finkfat_ops.config import parse_infra
from finkfat_ops.naming import Role
from finkfat_ops.openstack import OpenStackCli
from finkfat_ops.runner import CommandResult
from finkfat_ops.ssh import (
    build_ssh_config,
    evaluate_check,
    forget_host_command,
    init_checks,
    recreated_roles,
    ssh_command,
)
from finkfat_ops.vm import (
    Power,
    console_command,
    list_command,
    parse_server_list,
    power_command,
)

CLI = OpenStackCli(cloud="test-cloud", binary="openstack")
KNOWN_HOSTS = Path("/d/.state/known_hosts")


def _hosts(config: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    current = None
    for line in config.splitlines():
        if line.startswith("Host "):
            current = line.split()[1]
            blocks[current] = []
        elif current and line.strip():
            blocks[current].append(line.strip())
    return blocks


def test_ssh_config_reaches_private_machines_through_the_bastion(
    infra_data: dict[str, Any],
) -> None:
    config = build_ssh_config(
        parse_infra(infra_data), IPv4Address("157.136.0.9"), KNOWN_HOSTS, "~/.ssh/id_ed25519"
    )
    hosts = _hosts(config)
    assert set(hosts) == {"ffat-compute", "ffat-db", "ffat-explorer"}
    assert "HostName 157.136.0.9" in hosts["ffat-explorer"]
    assert not any(line.startswith("ProxyJump") for line in hosts["ffat-explorer"])
    assert "HostName 10.42.0.20" in hosts["ffat-db"]
    assert "ProxyJump ffat-explorer" in hosts["ffat-db"]
    assert "HostKeyAlias ffat-db" in hosts["ffat-db"]
    assert f"UserKnownHostsFile {KNOWN_HOSTS}" in hosts["ffat-db"]
    assert "IdentityFile ~/.ssh/id_ed25519" in hosts["ffat-db"]
    assert "User almalinux" in hosts["ffat-compute"]


def test_ssh_config_skips_disabled_roles_and_default_key(infra_data: dict[str, Any]) -> None:
    infra_data["roles"]["compute"]["enabled"] = False
    config = build_ssh_config(parse_infra(infra_data), IPv4Address("1.2.3.4"), KNOWN_HOSTS, None)
    assert "ffat-compute" not in config and "IdentityFile" not in config


def test_ssh_commands() -> None:
    shell = ssh_command(Path("/c"), Role.DB)
    assert shell.argv == ("ssh", "-F", "/c", "ffat-db") and shell.interactive
    remote = ssh_command(Path("/c"), Role.DB, ["docker", "ps"], timeout_s=5)
    assert remote.argv == (
        *("ssh", "-F", "/c", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"),
        *("ffat-db", "docker", "ps"),
    )
    assert not remote.interactive and remote.timeout_s == 5
    assert forget_host_command(KNOWN_HOSTS, Role.DB).argv[-2:] == ("-R", "ffat-db")


def test_recreated_roles_compare_vm_ids(infra_data: dict[str, Any]) -> None:
    infra = parse_infra(infra_data)
    after = {"server_db_id": "s2", "server_explorer_id": "e1", "server_compute_id": "c1"}
    before = {"server_db_id": "s1", "server_explorer_id": "e1"}
    assert recreated_roles(infra, before, after) == [Role.COMPUTE, Role.DB]
    assert recreated_roles(infra, None, after) == [Role.COMPUTE, Role.DB, Role.EXPLORER]
    assert recreated_roles(infra, after, after) == []


def test_init_checks_per_role(infra_data: dict[str, Any]) -> None:
    infra = parse_infra(infra_data)
    db = [c.label for c in init_checks(infra, Role.DB)]
    assert db[0].startswith("cloud-init") and "data volume mounted on /srv/pgdata" in db
    explorer = [c.label for c in init_checks(infra, Role.EXPLORER)]
    assert not any("volume" in label for label in explorer)
    first = init_checks(infra, Role.DB)[0]
    assert first.timeout_s >= 1800
    assert first.command[:2] == ("sudo", "cloud-init")  # /run/cloud-init is root-only


def test_check_evaluation(infra_data: dict[str, Any]) -> None:
    checks = {c.label: c for c in init_checks(parse_infra(infra_data), Role.DB)}
    clock = checks["clock synchronised"]
    assert evaluate_check(clock, CommandResult(stdout="yes\n")).level == Level.OK
    assert evaluate_check(clock, CommandResult(stdout="no\n")).level == Level.ERROR
    failed = evaluate_check(checks["docker running"], CommandResult(1, stderr="inactive"))
    assert failed.level == Level.ERROR and "inactive" in failed.message and failed.fix


# --------------------------------------------------------------------------------- vm


def test_vm_commands() -> None:
    assert list_command(CLI, "prod").argv[3:7] == ("server", "list", "--name", "^ffat-prod-")
    assert power_command(CLI, "prod", Role.DB, Power.REBOOT).argv[3:] == (
        "server",
        "reboot",
        "ffat-prod-db",
    )
    console = console_command(CLI, "prod", Role.DB, 50).argv
    assert console[3:] == ("console", "log", "show", "--lines", "50", "ffat-prod-db")


def test_server_list_parsing() -> None:
    data = json.loads(
        json.dumps(
            [
                {"ID": "2", "Name": "ffat-prod-explorer", "Status": "ACTIVE",
                 "Networks": {"ffat-prod-net": ["10.42.0.30", "157.136.0.9"]}},
                {"ID": "1", "Name": "ffat-prod-db", "Status": "SHUTOFF",
                 "Networks": "ffat-prod-net=10.42.0.20"},
            ]
        )
    )  # fmt: skip
    servers = parse_server_list(data)
    assert [s.name for s in servers] == ["ffat-prod-db", "ffat-prod-explorer"]
    assert servers[1].networks == "ffat-prod-net=10.42.0.30, 157.136.0.9"
    assert servers[0].networks == "ffat-prod-net=10.42.0.20"
    assert parse_server_list(None) == []


def test_recoverable_cloud_init_errors_are_a_warning_with_the_output(
    infra_data: dict[str, Any],
) -> None:
    cloud_init = init_checks(parse_infra(infra_data), Role.DB)[0]
    output = "status: done\nrecoverable_errors:\nWARNING:\n- deprecated key\n"
    finding = evaluate_check(cloud_init, CommandResult(2, stdout=output, stderr="ssh noise"))
    assert finding.level == Level.WARN
    assert "deprecated key" in finding.message
    assert finding.message.index("status: done") < finding.message.index("ssh noise")
    failed = evaluate_check(cloud_init, CommandResult(1, stdout="status: error"))
    assert failed.level == Level.ERROR and "status: error" in failed.message


def test_long_output_keeps_its_end(infra_data: dict[str, Any]) -> None:
    cloud_init = init_checks(parse_infra(infra_data), Role.DB)[0]
    traceback = "Traceback (most recent call last):\n" + "  frame\n" * 200 + "OSError: boom\n"
    finding = evaluate_check(cloud_init, CommandResult(1, stdout=traceback))
    assert finding.message.endswith("OSError: boom)")
