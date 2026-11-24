"""Tests of the net stack template."""

from __future__ import annotations

from ipaddress import IPv4Network
from typing import Any

import pytest
import yaml

from finkfat_ops.config import parse_infra
from finkfat_ops.heat.net import ADMIN_SSH_RESOURCE, PROTECTED_RESOURCES, build_net_template
from finkfat_ops.heat.render import HEADER, render_template

SSH = [IPv4Network("198.51.100.7/32"), IPv4Network("192.0.2.0/24")]


@pytest.fixture
def template(infra_data: dict[str, Any]) -> dict[str, Any]:
    return build_net_template(parse_infra(infra_data), SSH)


def _rules(template: dict[str, Any], resource: str) -> list[dict[str, Any]]:
    return template["resources"][resource]["properties"]["rules"]


def test_resources_and_names(template: dict[str, Any]) -> None:
    resources = template["resources"]
    assert set(resources) == {
        "net",
        "subnet",
        "router",
        "router_interface",
        "sg_explorer",
        "sg_admin_ssh",
        "sg_internal_ssh",
        "sg_compute",
        "sg_db",
    }
    assert resources["net"]["properties"]["name"] == "ffat-prod-net"
    assert resources["sg_db"]["properties"]["name"] == "ffat-prod-sg-db"
    assert set(resources) >= PROTECTED_RESOURCES
    assert template["heat_template_version"] == "2021-04-16"


def test_subnet_keeps_fixed_ips_out_of_dhcp(template: dict[str, Any]) -> None:
    subnet = template["resources"]["subnet"]["properties"]
    assert subnet["cidr"] == "10.42.0.0/24"
    assert subnet["allocation_pools"] == [{"start": "10.42.0.100", "end": "10.42.0.254"}]
    assert "dns_nameservers" not in subnet


def test_router_uses_the_external_network(template: dict[str, Any]) -> None:
    router = template["resources"]["router"]["properties"]
    assert router["external_gateway_info"] == {"network": "public-2"}


def test_web_is_public_but_ssh_is_restricted(template: dict[str, Any]) -> None:
    web = _rules(template, "sg_explorer")
    assert [(r["port_range_min"], r["remote_ip_prefix"]) for r in web] == [
        (80, "0.0.0.0/0"),
        (443, "0.0.0.0/0"),
    ]
    ssh = _rules(template, ADMIN_SSH_RESOURCE)
    assert [(r["port_range_min"], r["remote_ip_prefix"]) for r in ssh] == [
        (22, "192.0.2.0/24"),
        (22, "198.51.100.7/32"),
    ]
    assert all(r["direction"] == "ingress" for r in web + ssh)


def test_private_rules_use_remote_groups(template: dict[str, Any]) -> None:
    internal = _rules(template, "sg_internal_ssh")
    assert internal == [
        {
            "direction": "ingress",
            "ethertype": "IPv4",
            "protocol": "tcp",
            "port_range_min": 22,
            "port_range_max": 22,
            "remote_mode": "remote_group_id",
            "remote_group_id": {"get_resource": "sg_explorer"},
        }
    ]
    db_sources = [r["remote_group_id"] for r in _rules(template, "sg_db")]
    assert db_sources == [{"get_resource": "sg_compute"}, {"get_resource": "sg_explorer"}]
    assert all(r["port_range_min"] == 5432 for r in _rules(template, "sg_db"))
    assert _rules(template, "sg_compute") == []


def test_outputs_expose_every_identifier(template: dict[str, Any]) -> None:
    assert set(template["outputs"]) == {
        "network_id",
        "subnet_id",
        "router_id",
        "sg_explorer_id",
        "sg_admin_ssh_id",
        "sg_internal_ssh_id",
        "sg_db_id",
        "sg_compute_id",
    }


def test_duplicate_ssh_ranges_are_merged(infra_data: dict[str, Any]) -> None:
    template = build_net_template(parse_infra(infra_data), [*SSH, SSH[0]])
    assert len(_rules(template, ADMIN_SSH_RESOURCE)) == 2


@pytest.mark.parametrize("cidrs", [[], [IPv4Network("0.0.0.0/0")]])
def test_ssh_must_be_restricted(infra_data: dict[str, Any], cidrs: list[IPv4Network]) -> None:
    with pytest.raises(ValueError):
        build_net_template(parse_infra(infra_data), cidrs)


def test_dns_servers_are_passed_when_configured(infra_data: dict[str, Any]) -> None:
    infra_data["network"]["dns_nameservers"] = ["192.0.2.53"]
    template = build_net_template(parse_infra(infra_data), SSH)
    assert template["resources"]["subnet"]["properties"]["dns_nameservers"] == ["192.0.2.53"]


def test_rendered_yaml_round_trips(template: dict[str, Any]) -> None:
    text = render_template(template)
    assert text.startswith(HEADER)
    assert yaml.safe_load(text) == template
