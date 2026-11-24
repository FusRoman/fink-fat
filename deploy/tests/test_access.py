"""Tests of finkfat_ops.access."""

from __future__ import annotations

from ipaddress import IPv4Address, IPv4Network
from pathlib import Path
from typing import Any

import pytest

from finkfat_ops import access
from finkfat_ops.access import AccessState
from finkfat_ops.config import parse_local

HOME = IPv4Address("198.51.100.7")
NEW_HOME = IPv4Address("203.0.113.9")


def test_allow_me_replaces_the_previous_dynamic_ip() -> None:
    state = access.allow(AccessState(), HOME, temporary=False)
    state = access.allow(state, NEW_HOME, temporary=False)
    assert state.dynamic_cidrs == [IPv4Network("203.0.113.9/32")]


def test_temporary_adds_without_removing() -> None:
    state = access.allow(AccessState(), HOME, temporary=False)
    state = access.allow(state, NEW_HOME, temporary=True)
    state = access.allow(state, NEW_HOME, temporary=True)
    assert state.dynamic_cidrs == [IPv4Network("198.51.100.7/32"), IPv4Network("203.0.113.9/32")]


def test_revoke_only_touches_dynamic_ranges(local_data: dict[str, Any]) -> None:
    local = parse_local(local_data)
    state = access.allow(AccessState(), NEW_HOME, temporary=False)
    assert access.revoke(state, local, IPv4Network("203.0.113.9/32")).dynamic_cidrs == []
    with pytest.raises(ValueError, match="stable range"):
        access.revoke(state, local, IPv4Network("192.0.2.0/24"))
    with pytest.raises(ValueError, match="not an allowed range"):
        access.revoke(state, local, IPv4Network("203.0.113.0/24"))


def test_effective_ranges_and_coverage(local_data: dict[str, Any]) -> None:
    local = parse_local(local_data)
    state = AccessState(dynamic_cidrs=[IPv4Network("198.51.100.7/32")])
    cidrs = access.effective_ssh_cidrs(local, state)
    assert cidrs == [IPv4Network("192.0.2.0/24"), IPv4Network("198.51.100.7/32")]
    assert access.covers(cidrs, IPv4Address("192.0.2.200"))
    assert not access.covers(cidrs, NEW_HOME)


def test_public_ip_parsing() -> None:
    assert access.parse_public_ip("8.8.8.8\n") == IPv4Address("8.8.8.8")
    for bad in ("10.0.0.1", "not an ip", "2001:4860::8888"):
        with pytest.raises(ValueError):
            access.parse_public_ip(bad)


def test_state_round_trip(tmp_path: Path) -> None:
    path = access.access_state_path(tmp_path / ".state")
    assert access.load_access_state(path) == AccessState()
    state = AccessState(dynamic_cidrs=[IPv4Network("198.51.100.7/32")])
    access.save_access_state(path, state)
    assert access.load_access_state(path) == state
    assert path.read_text(encoding="utf-8").startswith("# Managed by `ff-ops access`")
