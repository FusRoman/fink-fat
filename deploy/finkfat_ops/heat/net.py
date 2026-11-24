"""Template of the ``net`` stack: private network, router and security groups.

Logical resource names (keys of ``resources``) are stable identifiers used by Heat to
compute updates; renaming one makes Heat replace the resource.
"""

from __future__ import annotations

from collections.abc import Sequence
from ipaddress import IPv4Network
from typing import Any

from finkfat_ops import naming
from finkfat_ops.config import InfraConfig
from finkfat_ops.naming import SecurityGroup

PROTECTED_RESOURCES = frozenset({"net", "subnet", "router", "router_interface"})
"""Resources whose replacement or deletion would cut every VM off the network."""

SECURITY_GROUP_RESOURCES = {
    SecurityGroup.EXPLORER: "sg_explorer",
    SecurityGroup.ADMIN_SSH: "sg_admin_ssh",
    SecurityGroup.INTERNAL_SSH: "sg_internal_ssh",
    SecurityGroup.DB: "sg_db",
    SecurityGroup.COMPUTE: "sg_compute",
}
"""Logical resource name of each security group."""

ADMIN_SSH_RESOURCE = SECURITY_GROUP_RESOURCES[SecurityGroup.ADMIN_SSH]
"""The only resource that ``ff-ops access`` is allowed to change."""

SSH_PORT = 22
POSTGRES_PORT = 5432
WEB_PORTS = (80, 443)


def _tcp_from_cidr(port: int, cidr: IPv4Network | str) -> dict[str, Any]:
    """Build an ingress rule opening one TCP port to an IPv4 range."""
    return {
        "direction": "ingress",
        "ethertype": "IPv4",
        "protocol": "tcp",
        "port_range_min": port,
        "port_range_max": port,
        "remote_mode": "remote_ip_prefix",
        "remote_ip_prefix": str(cidr),
    }


def _tcp_from_group(port: int, group_resource: str) -> dict[str, Any]:
    """Build an ingress rule opening one TCP port to the members of another group."""
    return {
        "direction": "ingress",
        "ethertype": "IPv4",
        "protocol": "tcp",
        "port_range_min": port,
        "port_range_max": port,
        "remote_mode": "remote_group_id",
        "remote_group_id": {"get_resource": group_resource},
    }


def _security_group(env: str, group: SecurityGroup, description: str, rules: list) -> dict:
    """Build an ``OS::Neutron::SecurityGroup`` resource (ingress rules only).

    Neutron keeps its default "allow all egress" rules, since no egress rule is given.
    """
    return {
        "type": "OS::Neutron::SecurityGroup",
        "properties": {
            "name": naming.security_group_name(env, group),
            "description": description,
            "rules": rules,
        },
    }


def build_net_template(infra: InfraConfig, ssh_cidrs: Sequence[IPv4Network]) -> dict[str, Any]:
    """Build the HOT template of the ``net`` stack.

    Security groups:

    * ``explorer`` — 80 and 443 from anywhere (the public web site);
    * ``admin-ssh`` — 22 from ``ssh_cidrs`` only (on the bastion);
    * ``internal-ssh`` — 22 from members of ``explorer`` (bastion hops to compute and db);
    * ``db`` — 5432 from members of ``compute`` and ``explorer``;
    * ``compute`` — no rule: a label used by ``db``.

    Parameters
    ----------
    infra : InfraConfig
        Versioned structure (environment, network).
    ssh_cidrs : Sequence[IPv4Network]
        Source ranges allowed to SSH into the bastion (stable and dynamic ones).

    Returns
    -------
    dict[str, Any]
        The template, ready for :func:`finkfat_ops.heat.render.render_template`.

    Raises
    ------
    ValueError
        ``ssh_cidrs`` is empty or contains a range open to the whole Internet.
    """
    if not ssh_cidrs:
        raise ValueError("at least one SSH source range is required")
    if any(cidr.prefixlen == 0 for cidr in ssh_cidrs):
        raise ValueError("SSH must not be open to the whole Internet")
    env, network = infra.env, infra.network
    pool_start, pool_end = network.dhcp_pool()
    sg = SECURITY_GROUP_RESOURCES
    subnet_properties: dict[str, Any] = {
        "name": naming.subnet_name(env),
        "network": {"get_resource": "net"},
        "cidr": str(network.cidr),
        "ip_version": 4,
        "enable_dhcp": True,
        "allocation_pools": [{"start": str(pool_start), "end": str(pool_end)}],
    }
    if network.dns_nameservers:
        subnet_properties["dns_nameservers"] = [str(ip) for ip in network.dns_nameservers]

    resources: dict[str, Any] = {
        "net": {"type": "OS::Neutron::Net", "properties": {"name": naming.network_name(env)}},
        "subnet": {"type": "OS::Neutron::Subnet", "properties": subnet_properties},
        "router": {
            "type": "OS::Neutron::Router",
            "properties": {
                "name": naming.router_name(env),
                "external_gateway_info": {"network": network.external_network},
            },
        },
        "router_interface": {
            "type": "OS::Neutron::RouterInterface",
            "properties": {
                "router": {"get_resource": "router"},
                "subnet": {"get_resource": "subnet"},
            },
        },
        sg[SecurityGroup.EXPLORER]: _security_group(
            env,
            SecurityGroup.EXPLORER,
            "Public web access to the explorer (HTTP for certificate issuance, HTTPS).",
            [_tcp_from_cidr(port, "0.0.0.0/0") for port in WEB_PORTS],
        ),
        sg[SecurityGroup.ADMIN_SSH]: _security_group(
            env,
            SecurityGroup.ADMIN_SSH,
            "SSH into the bastion from the allowed operator ranges only.",
            [_tcp_from_cidr(SSH_PORT, cidr) for cidr in sorted(set(ssh_cidrs))],
        ),
        sg[SecurityGroup.INTERNAL_SSH]: _security_group(
            env,
            SecurityGroup.INTERNAL_SSH,
            "SSH from the bastion to the private machines.",
            [_tcp_from_group(SSH_PORT, sg[SecurityGroup.EXPLORER])],
        ),
        sg[SecurityGroup.COMPUTE]: _security_group(
            env,
            SecurityGroup.COMPUTE,
            "Label of the compute machine (no ingress rule).",
            [],
        ),
        sg[SecurityGroup.DB]: _security_group(
            env,
            SecurityGroup.DB,
            "Postgres from the compute and explorer machines only.",
            [
                _tcp_from_group(POSTGRES_PORT, sg[SecurityGroup.COMPUTE]),
                _tcp_from_group(POSTGRES_PORT, sg[SecurityGroup.EXPLORER]),
            ],
        ),
    }
    outputs = {
        "network_id": {"value": {"get_resource": "net"}},
        "subnet_id": {"value": {"get_resource": "subnet"}},
        "router_id": {"value": {"get_resource": "router"}},
    } | {f"{resource}_id": {"value": {"get_resource": resource}} for resource in sg.values()}
    return {
        "heat_template_version": infra.heat_template_version,
        "description": f"fink-fat {env}: private network, router and security groups",
        "resources": resources,
        "outputs": outputs,
    }
