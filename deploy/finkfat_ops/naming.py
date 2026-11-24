"""Names of every OpenStack resource and SSH host managed by ff-ops.

All names derive from the environment name (``prod``) so that ff-ops only ever touches
resources prefixed ``ffat-<env>-``; everything else in the OpenStack project is left
alone. Every function here is pure.
"""

from __future__ import annotations

from enum import StrEnum

PREFIX = "ffat"
"""Common prefix of every resource created by ff-ops."""


class Role(StrEnum):
    """The three production machines."""

    COMPUTE = "compute"
    DB = "db"
    EXPLORER = "explorer"


class Stack(StrEnum):
    """The three Heat stacks, in creation order."""

    NET = "net"
    STORAGE = "storage"
    SERVERS = "servers"


class SecurityGroup(StrEnum):
    """The security groups created by the ``net`` stack."""

    EXPLORER = "explorer"
    ADMIN_SSH = "admin-ssh"
    INTERNAL_SSH = "internal-ssh"
    DB = "db"
    COMPUTE = "compute"


def env_prefix(env: str) -> str:
    """Return the prefix shared by every resource of an environment.

    Parameters
    ----------
    env : str
        Environment name, e.g. ``"prod"``.

    Returns
    -------
    str
        ``"ffat-<env>"``.
    """
    return f"{PREFIX}-{env}"


def stack_name(env: str, stack: Stack) -> str:
    """Return the Heat stack name, e.g. ``ffat-prod-net``.

    Parameters
    ----------
    env : str
        Environment name.
    stack : Stack
        Which stack.
    """
    return f"{env_prefix(env)}-{stack}"


def network_name(env: str) -> str:
    """Return the private network name, e.g. ``ffat-prod-net``."""
    return f"{env_prefix(env)}-net"


def subnet_name(env: str) -> str:
    """Return the private subnet name, e.g. ``ffat-prod-subnet``."""
    return f"{env_prefix(env)}-subnet"


def router_name(env: str) -> str:
    """Return the router name, e.g. ``ffat-prod-router``."""
    return f"{env_prefix(env)}-router"


def security_group_name(env: str, group: SecurityGroup) -> str:
    """Return a security group name, e.g. ``ffat-prod-sg-db``.

    Parameters
    ----------
    env : str
        Environment name.
    group : SecurityGroup
        Which security group.
    """
    return f"{env_prefix(env)}-sg-{group}"


def server_name(env: str, role: Role) -> str:
    """Return a VM name, e.g. ``ffat-prod-compute``."""
    return f"{env_prefix(env)}-{role}"


def port_name(env: str, role: Role) -> str:
    """Return the name of a VM's fixed-IP Neutron port, e.g. ``ffat-prod-db-port``."""
    return f"{server_name(env, role)}-port"


def data_volume_name(env: str, volume: str) -> str:
    """Return a data volume name, e.g. ``ffat-prod-vol-pgdata``.

    Parameters
    ----------
    env : str
        Environment name.
    volume : str
        Short volume name from ``infra.yaml`` (``engine``, ``pgdata``).
    """
    return f"{env_prefix(env)}-vol-{volume}"


def root_volume_name(env: str, role: Role) -> str:
    """Return the boot (root) volume name of a VM, e.g. ``ffat-prod-db-root``."""
    return f"{server_name(env, role)}-root"


def ssh_host_alias(role: Role) -> str:
    """Return the host alias used in the generated ``ssh_config``, e.g. ``ffat-db``."""
    return f"{PREFIX}-{role}"
