"""Tests of finkfat_ops.config."""

from __future__ import annotations

from ipaddress import ip_network
from pathlib import Path
from typing import Any

import pytest
import yaml

from finkfat_ops.config import (
    ConfigError,
    DeployPaths,
    load_deployment,
    parse_env_text,
    parse_infra,
    parse_local,
    parse_secrets,
)
from finkfat_ops.naming import Role


def _problems(fn, *args) -> str:
    with pytest.raises(ConfigError) as info:
        fn(*args)
    return "\n".join(info.value.problems)


# ------------------------------------------------------------------------ infra.yaml


def test_committed_infra_yaml_is_valid(infra_data: dict[str, Any]) -> None:
    infra = parse_infra(infra_data)
    assert infra.env == "prod"
    assert [role for role, _ in infra.roles.items()] == [Role.COMPUTE, Role.DB, Role.EXPLORER]
    assert infra.roles.explorer.floating_ip


@pytest.mark.parametrize("key", ["openstack", "admin_cidrs", "explorer_domain"])
def test_local_only_keys_are_refused_in_the_versioned_file(
    infra_data: dict[str, Any], key: str
) -> None:
    infra_data[key] = "anything"
    assert "must live in infra.local.yaml" in _problems(parse_infra, infra_data)


def test_typos_are_reported_with_their_path(infra_data: dict[str, Any]) -> None:
    infra_data["roles"]["db"]["flavour"] = "vd.16"
    assert "roles.db.flavour" in _problems(parse_infra, infra_data)


@pytest.mark.parametrize(
    ("ip", "expected"),
    [
        ("10.99.0.5", "is not a host of"),
        ("10.42.0.0", "is not a host of"),
        ("10.42.0.255", "is not a host of"),
        ("10.42.0.1", "router gateway"),
        ("10.42.0.20", "duplicates"),
    ],
)
def test_fixed_ip_must_be_a_usable_unique_host(
    infra_data: dict[str, Any], ip: str, expected: str
) -> None:
    infra_data["roles"]["compute"]["fixed_ip"] = ip
    assert expected in _problems(parse_infra, infra_data)


def test_data_roles_need_a_volume(infra_data: dict[str, Any]) -> None:
    del infra_data["roles"]["db"]["volume"]
    assert "roles.db needs a data volume" in _problems(parse_infra, infra_data)


def test_only_the_explorer_is_public(infra_data: dict[str, Any]) -> None:
    infra_data["roles"]["db"]["floating_ip"] = True
    assert "roles.db.floating_ip must be False" in _problems(parse_infra, infra_data)
    infra_data["roles"]["db"]["floating_ip"] = False
    infra_data["roles"]["explorer"]["floating_ip"] = False
    assert "roles.explorer.floating_ip must be True" in _problems(parse_infra, infra_data)


def test_volume_names_are_unique(infra_data: dict[str, Any]) -> None:
    infra_data["roles"]["db"]["volume"]["name"] = "engine"
    assert "also used by" in _problems(parse_infra, infra_data)


def test_root_volume_cannot_be_smaller_than_the_flavor_disk(infra_data: dict[str, Any]) -> None:
    infra_data["roles"]["db"]["root_gb"] = 10
    assert "roles.db.root_gb" in _problems(parse_infra, infra_data)


# ------------------------------------------------------------------- infra.local.yaml


def test_valid_local_config(local_data: dict[str, Any]) -> None:
    local = parse_local(local_data)
    assert local.openstack.project == "test-project"
    assert len(local.admin_cidrs) == 2


def test_ssh_open_to_the_world_is_refused(local_data: dict[str, Any]) -> None:
    local_data["admin_cidrs"] = ["0.0.0.0/0"]
    assert "must not be open to the whole Internet" in _problems(parse_local, local_data)


def test_at_least_one_ssh_range_is_required(local_data: dict[str, Any]) -> None:
    local_data["admin_cidrs"] = []
    assert "admin_cidrs" in _problems(parse_local, local_data)


def test_domain_is_optional_but_must_be_valid(local_data: dict[str, Any]) -> None:
    del local_data["explorer_domain"]
    assert parse_local(local_data).explorer_domain is None
    local_data["explorer_domain"] = "not a domain"
    assert "not a valid domain name" in _problems(parse_local, local_data)


# ------------------------------------------------------------------------ secrets.env


def test_env_text_parsing_rules() -> None:
    text = "# comment\n\nA=1\nB='two words'\nC=\"x=y\"\nEMPTY=\n"
    assert parse_env_text(text) == {"A": "1", "B": "two words", "C": "x=y"}


def test_env_text_rejects_bad_lines_and_duplicates() -> None:
    problems = _problems(parse_env_text, "A=1\nnot an assignment\nA=2\n")
    assert "line 2: expected KEY=VALUE" in problems
    assert "line 3: A is set twice" in problems


def test_secrets_never_leak_into_errors_or_repr() -> None:
    short = "tooshort"
    problems = _problems(parse_secrets, {"POSTGRES_PASSWORD": short, "PGADMIN_PASSWORD": short})
    assert "POSTGRES_PASSWORD" in problems and short not in problems
    secrets = parse_secrets({"POSTGRES_PASSWORD": "a" * 20, "PGADMIN_PASSWORD": "b" * 20})
    assert "a" * 20 not in repr(secrets)
    assert secrets.FINK_FAT_KAFKA_PASSWORD is None


def test_unknown_secret_keys_are_refused() -> None:
    values = {"POSTGRES_PASSWORD": "a" * 20, "PGADMIN_PASSWORD": "b" * 20, "POSTGRES_PASWORD": "x"}
    assert "POSTGRES_PASWORD" in _problems(parse_secrets, values)


# ------------------------------------------------------------------------- loading


def test_load_deployment_reads_the_three_files(deploy_dir: Path) -> None:
    deployment = load_deployment(DeployPaths(deploy_dir))
    assert deployment.infra.env == "prod"
    assert deployment.local.explorer_domain == "fink-fat.example.org"
    assert deployment.secrets.POSTGRES_PASSWORD.get_secret_value() == "pg-password-0123456789"


def test_every_file_problem_is_reported_at_once(deploy_dir: Path) -> None:
    (deploy_dir / "infra.local.yaml").unlink()
    (deploy_dir / "secrets.env").write_text("POSTGRES_PASSWORD=short\n", encoding="utf-8")
    problems = _problems(load_deployment, DeployPaths(deploy_dir))
    assert "infra.local.yaml: file not found" in problems
    assert "secrets.env: PGADMIN_PASSWORD" in problems
    assert "infra.yaml" not in problems.replace("infra.local.yaml", "")


def test_invalid_yaml_is_reported(deploy_dir: Path) -> None:
    (deploy_dir / "infra.yaml").write_text("roles: [unclosed\n", encoding="utf-8")
    assert "infra.yaml: invalid YAML" in _problems(load_deployment, DeployPaths(deploy_dir))


DOCUMENTATION_RANGES = [
    ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
]


def test_committed_local_example_holds_only_placeholders() -> None:
    """The versioned example must never contain a real address or domain."""
    path = Path(__file__).resolve().parents[1] / "infra.local.example.yaml"
    local = parse_local(yaml.safe_load(path.read_text(encoding="utf-8")))
    for cidr in local.admin_cidrs:
        assert any(cidr.subnet_of(doc) for doc in DOCUMENTATION_RANGES), cidr
    assert local.explorer_domain is None or local.explorer_domain.endswith(".example.org")


def test_ssh_user_must_be_a_valid_login(infra_data: dict[str, Any]) -> None:
    assert parse_infra(infra_data).ssh_user == "almalinux"
    infra_data["ssh_user"] = "root; rm -rf /"
    assert "ssh_user" in _problems(parse_infra, infra_data)


def test_fixed_ips_stay_below_the_dhcp_pool(infra_data: dict[str, Any]) -> None:
    infra_data["roles"]["db"]["fixed_ip"] = "10.42.0.150"
    assert "inside the DHCP pool (starting at 10.42.0.100)" in _problems(parse_infra, infra_data)


def test_private_subnet_must_be_at_least_a_slash_24(infra_data: dict[str, Any]) -> None:
    infra_data["network"]["cidr"] = "10.42.0.0/25"
    assert "too small" in _problems(parse_infra, infra_data)
