"""Tests of finkfat_ops.doctor (pure evaluations)."""

from __future__ import annotations

from ipaddress import IPv4Address, IPv4Network
from pathlib import Path

from finkfat_ops.checks import Level
from finkfat_ops.doctor import (
    Endpoint,
    evaluate_identity_file,
    evaluate_public_ip,
    evaluate_reach,
    evaluate_tool,
    parse_catalog,
)
from finkfat_ops.runner import CommandResult


def test_catalog_keeps_one_public_endpoint_per_service() -> None:
    data = [
        {
            "Type": "identity",
            "Endpoints": [
                {"interface": "internal", "url": "http://10.0.0.1:5000"},
                {"interface": "public", "url": "https://keystone.example.org:5000"},
            ],
        },
        {"Type": "volumev3", "Endpoints": "RegionOne\n  public: https://cinder.example.org/v3\n"},
        {"Type": "image", "Endpoints": [{"interface": "public", "url": "https://glance"}]},
    ]
    assert parse_catalog(data) == [
        Endpoint("Cinder", "https://cinder.example.org/v3"),
        Endpoint("Keystone", "https://keystone.example.org:5000"),
    ]
    assert parse_catalog(None) == []


def test_unreachable_heat_suggests_sshuttle() -> None:
    heat = Endpoint("Heat", "https://heat.example.org")
    finding = evaluate_reach(heat, CommandResult(returncode=28))
    assert finding.level == Level.ERROR and "sshuttle" in (finding.fix or "")
    assert evaluate_reach(heat, CommandResult()).level == Level.OK


def test_tool_failure_shows_its_last_line() -> None:
    finding = evaluate_tool("gh", CommandResult(returncode=1, stderr="a\nnot logged in\n"), "fix")
    assert finding.message == "gh: not logged in" and finding.fix == "fix"


def test_identity_file() -> None:
    key = Path("/home/u/.ssh/key")
    assert evaluate_identity_file(None, None).level == Level.OK
    assert evaluate_identity_file(key, None).level == Level.ERROR
    assert evaluate_identity_file(key, 0o100644).fix == f"chmod 600 {key}"
    assert evaluate_identity_file(key, 0o100600).level == Level.OK


def test_public_ip() -> None:
    cidrs = [IPv4Network("192.0.2.0/24")]
    assert evaluate_public_ip(IPv4Address("192.0.2.9"), cidrs).level == Level.OK
    outside = evaluate_public_ip(IPv4Address("198.51.100.1"), cidrs)
    assert outside.level == Level.WARN and "allow-me" in (outside.fix or "")
