"""Cinder volume commands and the parsing of their results.

All functions are pure: they build :class:`~finkfat_ops.runner.Command` values or
interpret JSON already returned by ``openstack``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from finkfat_ops.heat.storage import DataVolume
from finkfat_ops.openstack import OpenStackCli
from finkfat_ops.runner import Command


@dataclass(frozen=True)
class VolumeInfo:
    """A Cinder volume as listed by ``openstack volume list``.

    Attributes
    ----------
    id : str
        Volume ID.
    name : str
        Volume name.
    status : str
        Cinder status, e.g. ``available`` or ``in-use``.
    size_gb : int
        Current size.
    attached_to : str
        Cinder's description of the attachment (empty when detached).
    """

    id: str
    name: str
    status: str
    size_gb: int
    attached_to: str = ""


def list_by_name_command(cli: OpenStackCli, name: str) -> Command:
    """``openstack volume list --name <name>`` (every volume carrying that exact name)."""
    return cli.command("volume", "list", "--name", name)


def parse_volume_list(data: Any) -> list[VolumeInfo]:
    """Build :class:`VolumeInfo` values from ``openstack volume list -f json``.

    Parameters
    ----------
    data : Any
        Decoded JSON output (a list of objects with ``ID``, ``Name``, ``Status``,
        ``Size`` and ``Attached to``); ``None`` (dry-run) gives an empty list.

    Returns
    -------
    list[VolumeInfo]
        The volumes, in the order listed.
    """
    return [
        VolumeInfo(
            id=item["ID"],
            name=item.get("Name") or "",
            status=item.get("Status") or "",
            size_gb=int(item.get("Size") or 0),
            attached_to=item.get("Attached to") or "",
        )
        for item in data or []
    ]


def snapshot_name(volume: DataVolume, when: datetime) -> str:
    """Return a snapshot name, e.g. ``ffat-prod-vol-pgdata-20261124T093000Z``.

    Parameters
    ----------
    volume : DataVolume
        The snapshotted data volume.
    when : datetime
        Time of the snapshot, in UTC.
    """
    return f"{volume.name}-{when:%Y%m%dT%H%M%SZ}"


def snapshot_command(cli: OpenStackCli, volume_id: str, name: str) -> Command:
    """``openstack volume snapshot create --force``.

    ``--force`` allows the snapshot of a volume in use: the result is crash-consistent,
    which Postgres recovers from (its WAL lives on the same volume).

    Parameters
    ----------
    cli : OpenStackCli
        Command builder.
    volume_id : str
        Volume to snapshot.
    name : str
        Snapshot name (see :func:`snapshot_name`).
    """
    return cli.command(
        "volume",
        "snapshot",
        "create",
        "--volume",
        volume_id,
        "--force",
        "--description",
        "taken by ff-ops volume snapshot",
        name,
    )


def snapshot_show_command(cli: OpenStackCli, name: str) -> Command:
    """``openstack volume snapshot show <name>`` (status of a snapshot)."""
    return cli.command("volume", "snapshot", "show", name, json_output=False)


ORPHAN = "orphan"
"""Management label of a volume that no stack manages (``infra apply storage`` adopts it)."""


def management(
    volume: DataVolume,
    volume_id: str,
    stack_parameters: Mapping[str, str] | None,
    stack_outputs: Mapping[str, Any] | None,
) -> str:
    """Describe how the storage stack manages an existing volume (pure).

    Parameters
    ----------
    volume : DataVolume
        The data volume of ``infra.yaml``.
    volume_id : str
        ID of the Cinder volume carrying its name.
    stack_parameters : Mapping[str, str] | None
        Current parameters of the storage stack; ``None`` when it does not exist.
    stack_outputs : Mapping[str, Any] | None
        Current outputs of the storage stack; ``None`` when it does not exist.

    Returns
    -------
    str
        ``"adopted"`` (taken over by the stack), ``"created"`` (by the stack) or
        :data:`ORPHAN` (managed by no stack).
    """
    if (stack_parameters or {}).get(volume.parameter) == volume_id:
        return "adopted"
    if (stack_outputs or {}).get(volume.output) == volume_id:
        return "created"
    return ORPHAN
