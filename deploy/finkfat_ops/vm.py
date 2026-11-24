"""Nova commands on the production VMs (list, console, reboot, stop, start).

All functions are pure: they build :class:`~finkfat_ops.runner.Command` values or interpret
JSON already returned by ``openstack``. VMs are addressed by their name
(``ffat-<env>-<role>``), which only ff-ops uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from finkfat_ops import naming
from finkfat_ops.naming import Role
from finkfat_ops.openstack import OpenStackCli
from finkfat_ops.runner import Command

CONSOLE_LINES = 100
"""Default number of console log lines shown."""


class Power(StrEnum):
    """Power actions on a VM, named after their ``openstack server`` sub-command."""

    REBOOT = "reboot"
    STOP = "stop"
    START = "start"


@dataclass(frozen=True)
class ServerInfo:
    """A VM as listed by ``openstack server list``.

    Attributes
    ----------
    id : str
        Server ID.
    name : str
        Server name.
    status : str
        Nova status, e.g. ``ACTIVE`` or ``SHUTOFF``.
    networks : str
        Nova's description of the addresses.
    """

    id: str
    name: str
    status: str
    networks: str = ""


def list_command(cli: OpenStackCli, env: str) -> Command:
    """``openstack server list --name ^ffat-<env>-`` (the VMs of this environment only)."""
    return cli.command("server", "list", "--name", f"^{naming.env_prefix(env)}-")


def parse_server_list(data: Any) -> list[ServerInfo]:
    """Build :class:`ServerInfo` values from ``openstack server list -f json``.

    Parameters
    ----------
    data : Any
        Decoded JSON output; ``None`` (dry-run) gives an empty list.

    Returns
    -------
    list[ServerInfo]
        The VMs, sorted by name.
    """
    servers = [
        ServerInfo(
            id=item["ID"],
            name=item.get("Name") or "",
            status=item.get("Status") or "",
            networks=_networks_text(item.get("Networks")),
        )
        for item in data or []
    ]
    return sorted(servers, key=lambda server: server.name)


def _networks_text(value: Any) -> str:
    """Render Nova's addresses (a mapping in recent CLIs, a string in older ones)."""
    if isinstance(value, dict):
        return "; ".join(f"{net}={', '.join(map(str, ips))}" for net, ips in value.items())
    return str(value or "")


def power_command(cli: OpenStackCli, env: str, role: Role, action: Power) -> Command:
    """``openstack server reboot|stop|start ffat-<env>-<role>``."""
    return cli.command("server", str(action), naming.server_name(env, role), json_output=False)


def console_command(cli: OpenStackCli, env: str, role: Role, lines: int) -> Command:
    """``openstack console log show --lines <n> ffat-<env>-<role>`` (boot messages)."""
    return cli.command(
        "console",
        "log",
        "show",
        "--lines",
        str(lines),
        naming.server_name(env, role),
        json_output=False,
    )
