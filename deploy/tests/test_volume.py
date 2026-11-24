"""Tests of finkfat_ops.volume (commands, parsing, management labels)."""

from __future__ import annotations

from datetime import UTC, datetime

from finkfat_ops.heat.storage import DataVolume
from finkfat_ops.naming import Role
from finkfat_ops.openstack import OpenStackCli
from finkfat_ops.volume import (
    ORPHAN,
    VolumeInfo,
    list_by_name_command,
    management,
    parse_volume_list,
    snapshot_command,
    snapshot_name,
)

CLI = OpenStackCli(cloud="test-cloud", binary="openstack")
PGDATA = DataVolume(Role.DB, "vol_pgdata", "ffat-prod-vol-pgdata", 500)


def test_list_by_name_command() -> None:
    assert list_by_name_command(CLI, "ffat-prod-vol-pgdata").argv == (
        *("openstack", "--os-cloud", "test-cloud"),
        *("volume", "list", "--name", "ffat-prod-vol-pgdata", "-f", "json"),
    )


def test_volume_list_parsing() -> None:
    data = [
        {"ID": "v1", "Name": "ffat-prod-vol-pgdata", "Status": "in-use", "Size": 500,
         "Attached to": "Attached to s1 on /dev/vdb "},
        {"ID": "v2", "Name": "ffat-prod-vol-engine", "Status": "available", "Size": 200,
         "Attached to": ""},
    ]  # fmt: skip
    assert parse_volume_list(data) == [
        VolumeInfo("v1", "ffat-prod-vol-pgdata", "in-use", 500, "Attached to s1 on /dev/vdb "),
        VolumeInfo("v2", "ffat-prod-vol-engine", "available", 200, ""),
    ]
    assert parse_volume_list(None) == []


def test_snapshot_name_and_command() -> None:
    name = snapshot_name(PGDATA, datetime(2026, 11, 24, 9, 30, 0, tzinfo=UTC))
    assert name == "ffat-prod-vol-pgdata-20261124T093000Z"
    argv = snapshot_command(CLI, "v1", name).argv
    assert argv[3:6] == ("volume", "snapshot", "create")
    assert argv[argv.index("--volume") + 1] == "v1"
    assert "--force" in argv
    assert argv[-3:] == (name, "-f", "json")


def test_management_labels() -> None:
    assert management(PGDATA, "v1", {"vol_pgdata_existing_id": "v1"}, None) == "adopted"
    assert management(PGDATA, "v1", {}, {"vol_pgdata_id": "v1"}) == "created"
    assert management(PGDATA, "v1", None, None) == ORPHAN
