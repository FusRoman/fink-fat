"""Source ranges allowed to SSH into the bastion.

Two lists are combined:

* ``admin_cidrs`` in ``infra.local.yaml`` — stable ranges written by hand (never
  modified by ff-ops);
* ``dynamic_cidrs`` in ``.state/access.yaml`` — addresses managed by
  ``ff-ops access allow-me`` / ``revoke``, e.g. a home IP that changes.

The state file is rewritten by ff-ops, so ``infra.local.yaml`` keeps its comments.
"""

from __future__ import annotations

from collections.abc import Iterable
from ipaddress import IPv4Address, IPv4Network, ip_address
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from finkfat_ops.config import ConfigError, LocalConfig
from finkfat_ops.runner import Command

IP_ECHO_URL = "https://api.ipify.org"
"""Service returning the caller's public IPv4 address as plain text."""


class AccessState(BaseModel):
    """Content of ``.state/access.yaml``.

    Attributes
    ----------
    dynamic_cidrs : list[IPv4Network]
        Ranges added by ``ff-ops access allow-me``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dynamic_cidrs: list[IPv4Network] = Field(default_factory=list)


def access_state_path(state_dir: Path) -> Path:
    """Return the location of the access state file inside ``state_dir``."""
    return state_dir / "access.yaml"


def load_access_state(path: Path) -> AccessState:
    """Read the access state; a missing file means no dynamic range.

    Raises
    ------
    ConfigError
        The file exists but is not a valid state.
    """
    if not path.exists():
        return AccessState()
    try:
        return AccessState.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except (yaml.YAMLError, ValidationError) as error:
        raise ConfigError([f"{path}: invalid access state ({error})"]) from error


def save_access_state(path: Path, state: AccessState) -> None:
    """Write the access state (creating ``.state/`` if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"dynamic_cidrs": [str(cidr) for cidr in state.dynamic_cidrs]}
    header = "# Managed by `ff-ops access`. Stable ranges belong in infra.local.yaml.\n"
    path.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def effective_ssh_cidrs(local: LocalConfig, state: AccessState) -> list[IPv4Network]:
    """Combine stable and dynamic ranges, without duplicates, in a stable order.

    Parameters
    ----------
    local : LocalConfig
        Holds the stable ``admin_cidrs``.
    state : AccessState
        Holds the dynamic ranges.

    Returns
    -------
    list[IPv4Network]
        Sorted unique ranges.
    """
    return sorted(set(local.admin_cidrs) | set(state.dynamic_cidrs))


def covers(cidrs: Iterable[IPv4Network], ip: IPv4Address) -> bool:
    """Tell whether ``ip`` belongs to one of ``cidrs``."""
    return any(ip in cidr for cidr in cidrs)


def public_ip_command() -> Command:
    """Build the command that asks a public echo service for the caller's IPv4 address."""
    return Command.of("curl", "-4", "-fsS", "--max-time", "10", IP_ECHO_URL, timeout_s=20)


def parse_public_ip(output: str) -> IPv4Address:
    """Parse the answer of :func:`public_ip_command`.

    Raises
    ------
    ValueError
        The answer is not a public IPv4 address.
    """
    ip = ip_address(output.strip())
    if not isinstance(ip, IPv4Address) or not ip.is_global:
        raise ValueError(f"'{output.strip()}' is not a public IPv4 address")
    return ip


def allow(state: AccessState, ip: IPv4Address, *, temporary: bool) -> AccessState:
    """Return the state after allowing ``ip``.

    Parameters
    ----------
    state : AccessState
        Current state.
    ip : IPv4Address
        Address to allow (as a ``/32``).
    temporary : bool
        Add it next to the existing dynamic ranges instead of replacing them.

    Returns
    -------
    AccessState
        The new state.
    """
    cidr = IPv4Network(f"{ip}/32")
    if temporary:
        kept = [c for c in state.dynamic_cidrs if c != cidr]
        return AccessState(dynamic_cidrs=[*kept, cidr])
    return AccessState(dynamic_cidrs=[cidr])


def revoke(state: AccessState, local: LocalConfig, cidr: IPv4Network) -> AccessState:
    """Return the state after removing a dynamic range.

    Parameters
    ----------
    state : AccessState
        Current state.
    local : LocalConfig
        Used to explain where a stable range must be removed.
    cidr : IPv4Network
        Range to remove.

    Returns
    -------
    AccessState
        The new state.

    Raises
    ------
    ValueError
        ``cidr`` is a stable range (edit ``infra.local.yaml``) or is not allowed at all.
    """
    if cidr in state.dynamic_cidrs:
        return AccessState(dynamic_cidrs=[c for c in state.dynamic_cidrs if c != cidr])
    if cidr in local.admin_cidrs:
        raise ValueError(f"{cidr} is a stable range: remove it from infra.local.yaml (admin_cidrs)")
    raise ValueError(f"{cidr} is not an allowed range")
