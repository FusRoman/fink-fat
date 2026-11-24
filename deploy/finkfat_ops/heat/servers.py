"""Template of the ``servers`` stack: ports, boot volumes, VMs, attachments.

The stack consumes the outputs of the two other stacks as parameters
(:data:`NET_OUTPUTS`, the volume outputs of the storage stack), so it holds no ID itself and
can be destroyed and recreated at will: the data stays on the storage volumes.

For each enabled role:

* ``port_<role>`` — Neutron port with the role's fixed IP and security groups;
* ``root_<role>`` — boot volume created from the system image (named, so that it is easy
  to recognise; deleted with the VM);
* ``server_<role>`` — the VM, booting from ``root_<role>``; changing its ``user_data`` does
  not rebuild it (``IGNORE``) and changing its flavor resizes it (``RESIZE``);
* ``attach_<role>`` — attachment of the role's data volume, if it has one.

The explorer's public address belongs to the net stack (``fip_explorer``); this stack only
associates it with the explorer's port (``fip_explorer_assoc``). Destroying and recreating
this stack, or the explorer alone, therefore never changes the public address, hence the
DNS record.
"""

from __future__ import annotations

from typing import Any

from finkfat_ops import naming
from finkfat_ops.config import InfraConfig, RoleConfig
from finkfat_ops.heat.cloud_init import VOLUME_ID_PLACEHOLDER, build_user_data
from finkfat_ops.heat.net import (
    FLOATING_IP_ID_OUTPUT,
    PUBLIC_IP_OUTPUT,
    SECURITY_GROUP_RESOURCES,
)
from finkfat_ops.heat.storage import DataVolume, data_volumes
from finkfat_ops.naming import Role, SecurityGroup

NET_OUTPUTS = (
    "network_id",
    "subnet_id",
    *(f"{resource}_id" for resource in SECURITY_GROUP_RESOURCES.values()),
    FLOATING_IP_ID_OUTPUT,
    PUBLIC_IP_OUTPUT,
)
"""Outputs of the net stack consumed by this template, as parameters of the same name."""

ROLE_SECURITY_GROUPS = {
    Role.EXPLORER: (SecurityGroup.EXPLORER, SecurityGroup.ADMIN_SSH),
    Role.DB: (SecurityGroup.DB, SecurityGroup.INTERNAL_SSH),
    Role.COMPUTE: (SecurityGroup.COMPUTE, SecurityGroup.INTERNAL_SSH),
}
"""Security groups of each role's port (see the net stack for their rules)."""

FLOATING_IP_ASSOCIATION = "fip_explorer_assoc"
"""Association of the net stack's floating IP with the explorer's port."""

PROTECTED_ROLES = (Role.DB, Role.COMPUTE)
"""Roles whose VM and data attachment must not be destroyed silently (they hold the data)."""

EXPLORER_PUBLIC_IP_OUTPUT = PUBLIC_IP_OUTPUT
"""Output holding the public address of the bastion (passed through from the net stack)."""


def port_resource(role: Role) -> str:
    """Logical name of a role's port, e.g. ``port_db``."""
    return f"port_{role}"


def root_resource(role: Role) -> str:
    """Logical name of a role's boot volume, e.g. ``root_db``."""
    return f"root_{role}"


def server_resource(role: Role) -> str:
    """Logical name of a role's VM, e.g. ``server_db``."""
    return f"server_{role}"


def attachment_resource(role: Role) -> str:
    """Logical name of a role's data volume attachment, e.g. ``attach_db``."""
    return f"attach_{role}"


def server_id_output(role: Role) -> str:
    """Output holding the ID of a role's VM, e.g. ``server_db_id``."""
    return f"{server_resource(role)}_id"


def protected_resources(infra: InfraConfig) -> frozenset[str]:
    """Return the resources whose replacement or deletion needs ``--allow-replace``.

    The db and compute VMs and their data attachments (detaching a mounted disk risks
    corrupting it). The floating IP is protected in the net stack.

    Parameters
    ----------
    infra : InfraConfig
        Versioned structure.

    Returns
    -------
    frozenset[str]
        Logical resource names.
    """
    protected: set[str] = set()
    for role, cfg in infra.roles.items():
        if role in PROTECTED_ROLES:
            protected.add(server_resource(role))
            if cfg.volume is not None:
                protected.add(attachment_resource(role))
    return frozenset(protected)


def _parameters(volumes: list[DataVolume]) -> dict[str, Any]:
    """Declare the IDs coming from the net and storage stacks."""
    parameters = {
        name: {"type": "string", "description": f"Output {name} of the net stack."}
        for name in NET_OUTPUTS
    }
    for volume in volumes:
        parameters[volume.output] = {
            "type": "string",
            "description": f"ID of {volume.name} (output of the storage stack).",
        }
    return parameters


def _user_data(infra: InfraConfig, cfg: RoleConfig, volume: DataVolume | None) -> Any:
    """Build the ``user_data`` property: plain text, or a ``str_replace`` for the volume ID."""
    text = build_user_data(infra.ssh_user, cfg.volume)
    if volume is None:
        return text
    return {
        "str_replace": {
            "template": text,
            "params": {VOLUME_ID_PLACEHOLDER: {"get_param": volume.output}},
        }
    }


def _role_resources(
    infra: InfraConfig, role: Role, cfg: RoleConfig, volume: DataVolume | None
) -> dict[str, Any]:
    """Build the port, boot volume, VM and attachment of one enabled role."""
    env = infra.env
    resources: dict[str, Any] = {
        port_resource(role): {
            "type": "OS::Neutron::Port",
            "properties": {
                "name": naming.port_name(env, role),
                "network": {"get_param": "network_id"},
                "fixed_ips": [
                    {"subnet": {"get_param": "subnet_id"}, "ip_address": str(cfg.fixed_ip)}
                ],
                "security_groups": [
                    {"get_param": f"{SECURITY_GROUP_RESOURCES[group]}_id"}
                    for group in ROLE_SECURITY_GROUPS[role]
                ],
            },
        },
        root_resource(role): {
            "type": "OS::Cinder::Volume",
            "properties": {
                "name": naming.root_volume_name(env, role),
                "size": cfg.root_gb,
                "image": infra.os_image,
                "volume_type": infra.volume_type,
                "description": f"fink-fat {env}: system disk of the {role} machine",
            },
        },
        server_resource(role): {
            "type": "OS::Nova::Server",
            "properties": {
                "name": naming.server_name(env, role),
                "flavor": cfg.flavor,
                "key_name": infra.keypair,
                "availability_zone": infra.availability_zone,
                "networks": [{"port": {"get_resource": port_resource(role)}}],
                "block_device_mapping_v2": [
                    {
                        "boot_index": 0,
                        "volume_id": {"get_resource": root_resource(role)},
                        "delete_on_termination": False,
                    }
                ],
                "user_data_format": "RAW",
                "user_data": _user_data(infra, cfg, volume),
                "user_data_update_policy": "IGNORE",
                "flavor_update_policy": "RESIZE",
            },
        },
    }
    if volume is not None:
        resources[attachment_resource(role)] = {
            "type": "OS::Cinder::VolumeAttachment",
            "properties": {
                "instance_uuid": {"get_resource": server_resource(role)},
                "volume_id": {"get_param": volume.output},
            },
        }
    return resources


def build_servers_template(infra: InfraConfig) -> dict[str, Any]:
    """Build the HOT template of the ``servers`` stack.

    Disabled roles (``enabled: false``) get no resource: applying the template deletes
    their VM, while their data volume stays in the storage stack.

    Parameters
    ----------
    infra : InfraConfig
        Versioned structure.

    Returns
    -------
    dict[str, Any]
        The template, ready for :func:`finkfat_ops.heat.render.render_template`.
    """
    volumes = data_volumes(infra)
    volume_of = {volume.role: volume for volume in volumes}
    resources: dict[str, Any] = {}
    outputs: dict[str, Any] = {
        EXPLORER_PUBLIC_IP_OUTPUT: {
            "description": "Public address of the explorer (web site and SSH bastion)",
            "value": {"get_param": PUBLIC_IP_OUTPUT},
        }
    }
    for role, cfg in infra.roles.items():
        if not cfg.enabled:
            continue
        resources |= _role_resources(infra, role, cfg, volume_of.get(role))
        outputs[server_id_output(role)] = {
            "description": f"ID of the {role} VM",
            "value": {"get_resource": server_resource(role)},
        }
        if cfg.floating_ip:
            resources[FLOATING_IP_ASSOCIATION] = {
                "type": "OS::Neutron::FloatingIPAssociation",
                "properties": {
                    "floatingip_id": {"get_param": FLOATING_IP_ID_OUTPUT},
                    "port_id": {"get_resource": port_resource(role)},
                },
            }
    return {
        "heat_template_version": infra.heat_template_version,
        "description": f"fink-fat {infra.env}: machines and their disks",
        "parameters": _parameters(volumes),
        "resources": resources,
        "outputs": outputs,
    }


class MissingUpstreamError(ValueError):
    """An output needed by the servers stack is missing from the net or storage stack."""


def upstream_parameters(
    infra: InfraConfig, net_outputs: dict[str, Any], storage_outputs: dict[str, Any]
) -> dict[str, str]:
    """Collect the parameters of the servers stack from the other stacks' outputs (pure).

    Parameters
    ----------
    infra : InfraConfig
        Versioned structure.
    net_outputs : dict[str, Any]
        Outputs of the net stack.
    storage_outputs : dict[str, Any]
        Outputs of the storage stack.

    Returns
    -------
    dict[str, str]
        Every parameter of :func:`build_servers_template`.

    Raises
    ------
    MissingUpstreamError
        An output is missing or empty (stack absent, failed or outdated).
    """
    wanted = dict.fromkeys(NET_OUTPUTS, (net_outputs, "net")) | {
        volume.output: (storage_outputs, "storage") for volume in data_volumes(infra)
    }
    parameters: dict[str, str] = {}
    missing: list[str] = []
    for name, (outputs, stack) in wanted.items():
        value = outputs.get(name)
        if value:
            parameters[name] = str(value)
        else:
            missing.append(f"{name} ({stack})")
    if missing:
        raise MissingUpstreamError(
            f"missing outputs: {', '.join(missing)}; apply the net and storage stacks first"
        )
    return parameters
