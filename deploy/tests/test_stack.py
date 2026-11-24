"""Tests of finkfat_ops.stack: commands, parsing and the replacement guard."""

from __future__ import annotations

import json
from pathlib import Path

from finkfat_ops import stack
from finkfat_ops.openstack import OpenStackCli
from finkfat_ops.stack import ChangeSet, GuardViolation

CLI = OpenStackCli(cloud="test-cloud", binary="openstack")


def test_every_command_is_bound_to_the_cloud_profile() -> None:
    template = Path("/tmp/net.yaml")
    commands = [
        stack.show_command(CLI, "s"),
        stack.outputs_command(CLI, "s"),
        stack.preview_update_command(CLI, "s", template),
        stack.create_command(CLI, "s", template),
        stack.update_command(CLI, "s", template),
        stack.delete_command(CLI, "s"),
    ]
    assert all(c.argv[:3] == ("openstack", "--os-cloud", "test-cloud") for c in commands)


def test_preview_and_create_command_lines() -> None:
    preview = stack.preview_update_command(CLI, "ffat-prod-net", Path("/b/net.yaml"))
    assert preview.argv[3:] == (
        "stack",
        "update",
        "--dry-run",
        "--show-nested",
        "-t",
        "/b/net.yaml",
        "ffat-prod-net",
        "-f",
        "json",
    )
    create = stack.create_command(CLI, "ffat-prod-net", Path("/b/net.yaml"))
    assert "--wait" in create.argv and create.timeout_s == 1800
    delete = stack.delete_command(CLI, "ffat-prod-net")
    assert "--yes" in delete.argv and "-f" not in delete.argv


def test_change_set_accepts_objects_names_and_json_strings() -> None:
    data = {
        "added": [{"resource_name": "sg_new", "resource_type": "OS::Neutron::SecurityGroup"}],
        "updated": json.dumps([{"resource_name": "sg_admin_ssh"}]),
        "replaced": ["router"],
        "deleted": [],
        "unchanged": [{"resource_name": "net"}, {"resource_name": "subnet"}],
    }
    changes = stack.parse_change_set(data)
    assert changes == ChangeSet(
        added=("sg_new",),
        updated=("sg_admin_ssh",),
        replaced=("router",),
        unchanged=("net", "subnet"),
    )
    assert changes.changed() == {"sg_new", "sg_admin_ssh", "router"}
    assert not changes.is_empty


def test_no_change_is_empty() -> None:
    assert stack.parse_change_set({"unchanged": ["net"]}).is_empty
    assert stack.parse_change_set(None).is_empty


def test_outputs_from_show_and_from_output_show() -> None:
    from_show = [{"output_key": "network_id", "output_value": "abc", "description": ""}]
    from_output_show = {"network_id": {"output_key": "network_id", "output_value": "abc"}}
    assert stack.parse_outputs(from_show) == {"network_id": "abc"}
    assert stack.parse_outputs(from_output_show) == {"network_id": "abc"}
    assert stack.parse_outputs(None) == {}


def test_stack_show_parsing_and_health() -> None:
    info = stack.parse_stack_show(
        {
            "stack_name": "ffat-prod-net",
            "stack_status": "UPDATE_FAILED",
            "stack_status_reason": "Quota exceeded",
            "outputs": [],
        }
    )
    assert info.reason == "Quota exceeded" and not info.healthy
    ok = stack.parse_stack_show({"stack_name": "s", "stack_status": "CREATE_COMPLETE"})
    assert ok.healthy and ok.reason == ""


def test_creation_change_set_lists_every_resource() -> None:
    changes = stack.change_set_for_creation({"resources": {"b": {}, "a": {}}})
    assert changes.added == ("a", "b")


def test_guard_blocks_destruction_of_protected_resources() -> None:
    changes = ChangeSet(updated=("sg_db",), replaced=("router", "sg_x"), deleted=("net",))
    violations = stack.guard_violations(changes, {"net", "router", "subnet"})
    assert violations == [GuardViolation("router", "replaced"), GuardViolation("net", "deleted")]


def test_guard_honours_explicit_permission() -> None:
    changes = ChangeSet(replaced=("router",))
    assert stack.guard_violations(changes, {"router"}, allowed={"router"}) == []


def test_updates_in_place_are_never_blocked() -> None:
    changes = ChangeSet(updated=("net", "router"))
    assert stack.guard_violations(changes, {"net", "router"}) == []


def test_parameters_are_passed_sorted_before_the_stack_name() -> None:
    parameters = {"vol_pgdata_existing_id": "v1", "vol_engine_existing_id": ""}
    template = Path("/b/storage.yaml")
    for build in (stack.preview_update_command, stack.create_command, stack.update_command):
        argv = build(CLI, "ffat-prod-storage", template, parameters).argv
        assert argv[-7:] == (
            *("--parameter", "vol_engine_existing_id="),
            *("--parameter", "vol_pgdata_existing_id=v1"),
            *("ffat-prod-storage", "-f", "json"),
        )


def test_stack_show_reads_the_parameters() -> None:
    info = stack.parse_stack_show(
        {
            "stack_name": "s",
            "stack_status": "CREATE_COMPLETE",
            "parameters": {"OS::stack_id": "x", "vol_pgdata_existing_id": "v1"},
        }
    )
    assert info.parameters["vol_pgdata_existing_id"] == "v1"
    as_text = stack.parse_stack_show(
        {"stack_name": "s", "stack_status": "CREATE_COMPLETE", "parameters": '{"a": "b"}'}
    )
    assert as_text.parameters == {"a": "b"}


def test_creation_change_set_leaves_out_skipped_resources() -> None:
    changes = stack.change_set_for_creation({"resources": {"b": {}, "a": {}}}, skipped=["a"])
    assert changes.added == ("b",)
