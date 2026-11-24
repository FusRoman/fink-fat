"""First-boot preparation of the VMs (cloud-init ``user_data``).

One model for the three roles: Docker, time synchronisation, automatic security updates,
and, for the roles that have one, the mount of the data volume. The text never holds a
secret: ``user_data`` is readable by every process of the VM (metadata service) and by
anyone who can run ``openstack stack show``.

The data volume is found through ``/dev/disk/by-id/virtio-<first 20 characters of its
ID>``, a stable path (``/dev/vdb`` may change between boots). Its ID is only known when the
stack is applied, so the text carries :data:`VOLUME_ID_PLACEHOLDER`, replaced by Heat
(``str_replace``).
"""

from __future__ import annotations

from typing import Any

from finkfat_ops.config import VolumeConfig
from finkfat_ops.heat.render import dump_yaml

VOLUME_ID_PLACEHOLDER = "__FFAT_DATA_VOLUME_ID__"
"""Marker of the data volume ID in the ``user_data`` text."""

MOUNT_SCRIPT_PATH = "/usr/local/sbin/ffat-mount-volume"
"""Location of the mount script on the VM."""

VOLUME_WAIT_S = 900
"""How long the mount script waits for the data volume (Heat attaches it after the boot)."""

DOCKER_REPO = "https://download.docker.com/linux/rhel/docker-ce.repo"
"""Docker CE package repository for RHEL-compatible systems (AlmaLinux)."""

MOUNT_SCRIPT = f"""#!/bin/bash
# Mount a data volume by its Cinder ID: wait for the disk, create a file system only if the
# disk is blank, record it in /etc/fstab (by-id path, nofail) and mount it.
# Usage: ffat-mount-volume <volume id> <mount point>
set -euo pipefail
volume_id="$1"
mount_point="$2"
device="/dev/disk/by-id/virtio-${{volume_id:0:20}}"

for _ in $(seq {VOLUME_WAIT_S}); do
    [ -b "$device" ] && break
    sleep 1
done
if [ ! -b "$device" ]; then
    echo "ffat-mount-volume: $device did not appear after {VOLUME_WAIT_S} s" >&2
    exit 1
fi

if [ -z "$(blkid -o value -s TYPE "$device" || true)" ]; then
    mkfs.ext4 -q -L "$(basename "$mount_point")" "$device"
fi

mkdir -p "$mount_point"
if ! grep -q "^$device " /etc/fstab; then
    echo "$device $mount_point ext4 defaults,nofail,x-systemd.device-timeout=30s 0 2" \\
        >> /etc/fstab
fi
systemctl daemon-reload
mountpoint -q "$mount_point" || mount "$mount_point"
"""
"""Content of :data:`MOUNT_SCRIPT_PATH`. It never formats a disk that holds a file system."""

DOCKER_DAEMON_CONFIG = '{\n  "log-driver": "local"\n}\n'
"""``/etc/docker/daemon.json``: the ``local`` log driver rotates container logs."""


def build_user_data(ssh_user: str, volume: VolumeConfig | None) -> str:
    """Build the ``#cloud-config`` text of a role.

    Parameters
    ----------
    ssh_user : str
        Default user of the image, added to the ``docker`` group.
    volume : VolumeConfig | None
        Data volume to mount, or ``None`` (explorer).

    Returns
    -------
    str
        The cloud-config document. With a volume, it contains
        :data:`VOLUME_ID_PLACEHOLDER` where the volume ID belongs.
    """
    write_files: list[dict[str, Any]] = [
        {"path": "/etc/docker/daemon.json", "permissions": "0644", "content": DOCKER_DAEMON_CONFIG}
    ]
    runcmd: list[Any] = [
        ["sed", "-i", "s/^upgrade_type.*/upgrade_type = security/", "/etc/dnf/automatic.conf"],
        ["systemctl", "enable", "--now", "chronyd", "dnf-automatic-install.timer"],
        ["dnf", "config-manager", "--add-repo", DOCKER_REPO],
        [
            "dnf",
            "install",
            "-y",
            "docker-ce",
            "docker-ce-cli",
            "containerd.io",
            "docker-compose-plugin",
        ],
        ["systemctl", "enable", "--now", "docker"],
        ["usermod", "-aG", "docker", ssh_user],
    ]
    if volume is not None:
        write_files.append(
            {"path": MOUNT_SCRIPT_PATH, "permissions": "0755", "content": MOUNT_SCRIPT}
        )
        runcmd.append([MOUNT_SCRIPT_PATH, VOLUME_ID_PLACEHOLDER, volume.mount])
    document = {
        "package_update": True,
        "packages": ["chrony", "dnf-automatic", "dnf-plugins-core"],
        "write_files": write_files,
        "runcmd": runcmd,
    }
    return "#cloud-config\n" + dump_yaml(document)
