"""Tests of the servers stack template, of cloud-init and of the YAML rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from finkfat_ops.config import DeployPaths, load_deployment, parse_infra
from finkfat_ops.heat.cloud_init import (
    MOUNT_SCRIPT_PATH,
    VOLUME_ID_PLACEHOLDER,
    build_user_data,
)
from finkfat_ops.heat.net import FLOATING_IP, build_net_template
from finkfat_ops.heat.render import dump_yaml, render_template
from finkfat_ops.heat.servers import (
    FLOATING_IP_ASSOCIATION,
    NET_OUTPUTS,
    MissingUpstreamError,
    build_servers_template,
    protected_resources,
    upstream_parameters,
)
from finkfat_ops.heat.storage import build_storage_template

NET_OUTPUT_VALUES = {name: f"id-{name}" for name in NET_OUTPUTS}
STORAGE_OUTPUT_VALUES = {"vol_engine_id": "v-engine", "vol_pgdata_id": "v-pgdata"}


@pytest.fixture
def template(infra_data: dict[str, Any]) -> dict[str, Any]:
    return build_servers_template(parse_infra(infra_data))


def _props(template: dict[str, Any], resource: str) -> dict[str, Any]:
    return template["resources"][resource]["properties"]


def test_every_enabled_role_gets_port_disk_vm(template: dict[str, Any]) -> None:
    resources = template["resources"]
    for role in ("compute", "db", "explorer"):
        assert {f"port_{role}", f"root_{role}", f"server_{role}"} <= set(resources)
    assert {"attach_compute", "attach_db"} <= set(resources)
    assert "attach_explorer" not in resources


def test_port_has_fixed_ip_and_role_security_groups(template: dict[str, Any]) -> None:
    port = _props(template, "port_db")
    assert port["fixed_ips"] == [{"subnet": {"get_param": "subnet_id"}, "ip_address": "10.42.0.20"}]
    assert port["security_groups"] == [
        {"get_param": "sg_db_id"},
        {"get_param": "sg_internal_ssh_id"},
    ]
    explorer = _props(template, "port_explorer")["security_groups"]
    assert {"get_param": "sg_admin_ssh_id"} in explorer


def test_vm_boots_from_its_named_root_volume(template: dict[str, Any]) -> None:
    root = _props(template, "root_compute")
    assert root["name"] == "ffat-prod-compute-root" and root["size"] == 50
    assert root["image"].startswith("AlmaLinux-9")
    server = _props(template, "server_compute")
    assert server["flavor"] == "vd.64"
    assert server["block_device_mapping_v2"] == [
        {
            "boot_index": 0,
            "volume_id": {"get_resource": "root_compute"},
            "delete_on_termination": False,
        }
    ]
    assert server["user_data_update_policy"] == "IGNORE"
    assert server["flavor_update_policy"] == "RESIZE"


def test_data_volume_id_reaches_attachment_and_user_data(template: dict[str, Any]) -> None:
    assert _props(template, "attach_db") == {
        "instance_uuid": {"get_resource": "server_db"},
        "volume_id": {"get_param": "vol_pgdata_id"},
    }
    user_data = _props(template, "server_db")["user_data"]["str_replace"]
    assert user_data["params"] == {VOLUME_ID_PLACEHOLDER: {"get_param": "vol_pgdata_id"}}
    assert "/srv/pgdata" in user_data["template"]
    assert isinstance(_props(template, "server_explorer")["user_data"], str)


def test_only_the_floating_ip_association_lives_here(template: dict[str, Any]) -> None:
    assert FLOATING_IP not in template["resources"]
    assert _props(template, FLOATING_IP_ASSOCIATION) == {
        "floatingip_id": {"get_param": "explorer_floating_ip_id"},
        "port_id": {"get_resource": "port_explorer"},
    }
    assert template["outputs"]["explorer_public_ip"]["value"] == {"get_param": "explorer_public_ip"}


def test_disabled_roles_get_no_resource(infra_data: dict[str, Any]) -> None:
    infra_data["roles"]["explorer"]["enabled"] = False
    infra_data["roles"]["compute"]["enabled"] = False
    resources = build_servers_template(parse_infra(infra_data))["resources"]
    assert not any(name.endswith(("_explorer", "_compute")) for name in resources)
    assert FLOATING_IP_ASSOCIATION not in resources


def test_data_holders_and_public_address_are_protected(infra_data: dict[str, Any]) -> None:
    assert protected_resources(parse_infra(infra_data)) == {
        "server_db",
        "server_compute",
        "attach_db",
        "attach_compute",
    }


def test_every_parameter_is_declared(template: dict[str, Any], infra_data: dict[str, Any]) -> None:
    parameters = upstream_parameters(
        parse_infra(infra_data), NET_OUTPUT_VALUES, STORAGE_OUTPUT_VALUES
    )
    assert set(parameters) == set(template["parameters"])
    assert parameters["vol_pgdata_id"] == "v-pgdata"


def test_missing_upstream_outputs_are_named(infra_data: dict[str, Any]) -> None:
    with pytest.raises(MissingUpstreamError, match=r"vol_pgdata_id \(storage\)"):
        upstream_parameters(parse_infra(infra_data), NET_OUTPUT_VALUES, {"vol_engine_id": "v"})


def test_rendered_templates_hold_no_secret(deploy_dir: Path) -> None:
    deployment = load_deployment(DeployPaths(deploy_dir))
    infra = deployment.infra
    texts = [
        render_template(build_net_template(infra, deployment.local.admin_cidrs)),
        render_template(build_storage_template(infra)),
        render_template(build_servers_template(infra)),
    ]
    secrets = [
        value.get_secret_value()
        for value in deployment.secrets.__dict__.values()
        if hasattr(value, "get_secret_value") and value.get_secret_value()
    ]
    assert secrets
    assert not any(secret in text for secret in secrets for text in texts)


# ------------------------------------------------------------------------ cloud-init


def test_user_data_without_volume_installs_docker_only() -> None:
    text = build_user_data("almalinux", None)
    assert text.startswith("#cloud-config\n")
    document = yaml.safe_load(text)
    assert "chrony" in document["packages"]
    commands = [" ".join(c) for c in document["runcmd"]]
    assert any(c.startswith("dnf install -y docker-ce") for c in commands)
    assert "usermod -aG docker almalinux" in commands
    assert MOUNT_SCRIPT_PATH not in text and VOLUME_ID_PLACEHOLDER not in text


def test_user_data_with_volume_mounts_it_last(infra_data: dict[str, Any]) -> None:
    volume = parse_infra(infra_data).roles.db.volume
    document = yaml.safe_load(build_user_data("almalinux", volume))
    assert document["runcmd"][-1] == [MOUNT_SCRIPT_PATH, VOLUME_ID_PLACEHOLDER, "/srv/pgdata"]
    script = next(f for f in document["write_files"] if f["path"] == MOUNT_SCRIPT_PATH)
    assert script["permissions"] == "0755"
    assert 'blkid -o value -s TYPE "$device"' in script["content"]  # never reformat data
    assert "virtio-${volume_id:0:20}" in script["content"]
    assert "nofail" in script["content"]


def test_dump_yaml_round_trips_and_uses_blocks() -> None:
    document = {"b": "one line", "a": "two\nlines\n", "c": [1, True]}
    text = dump_yaml(document)
    assert yaml.safe_load(text) == document
    assert "a: |" in text and text.index("b:") < text.index("a:")
