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


# ------------------------------------------------------------- deployed template comparison

STORAGE = {
    "conditions": {"adopt_vol": {"not": {"equals": [{"get_param": "vol_existing_id"}, ""]}}},
    "resources": {
        "vol": {"type": "OS::Cinder::Volume", "condition": {"not": "adopt_vol"},
                "properties": {"size": 1}},
        "other": {"type": "OS::Heat::None", "properties": {"x": {"get_param": "p"}}},
    },
}  # fmt: skip


def test_identical_templates_change_nothing() -> None:
    assert stack.really_changed(STORAGE, STORAGE, {"p": "1"}, {"p": "1"}) == set()


def test_definition_and_presence_changes_are_detected() -> None:
    new = json.loads(json.dumps(STORAGE))
    new["resources"]["vol"]["properties"]["size"] = 2
    new["resources"]["extra"] = {"type": "OS::Heat::None"}
    del new["resources"]["other"]
    assert stack.really_changed(STORAGE, new, {}, {}) == {"vol", "extra", "other"}


def test_changed_parameters_reach_their_users_and_conditions() -> None:
    assert stack.really_changed(STORAGE, STORAGE, {"p": "1"}, {"p": "2"}) == {"other"}
    changed = stack.really_changed(
        STORAGE, STORAGE, {"vol_existing_id": ""}, {"vol_existing_id": "v1"}
    )
    assert changed == {"vol"}


def test_filter_keeps_additions_and_confirmed_changes() -> None:
    previewed = ChangeSet(added=("a",), updated=("u", "x"), replaced=("r",), deleted=("d",))
    kept, ignored = stack.filter_change_set(previewed, {"u"})
    assert kept.added == ("a",) and kept.deleted == ("d",)
    assert kept.updated == ("u",) and kept.replaced == ()
    assert ignored == ("r", "x")


def test_template_show_parsing() -> None:
    assert stack.parse_template({"resources": '{"a": {}}'}) == {"resources": {"a": {}}}
    assert stack.parse_template(None) == {}


# ---------------------------------------------------------------- in-place reclassification

OLD = {
    "resources": {
        "subnet": {
            "type": "OS::Neutron::Subnet",
            "properties": {"network": {"get_resource": "net"}, "cidr": "a"},
        }
    }
}
SCHEMA = {
    "properties": {"cidr": {"update_allowed": False}, "dns_nameservers": '{"update_allowed": true}'}
}


def _with(**properties: object) -> dict:
    new = json.loads(json.dumps(OLD))
    new["resources"]["subnet"]["properties"].update(properties)
    return new


def test_update_allowed_parsing() -> None:
    assert stack.parse_update_allowed(SCHEMA) == {"cidr": False, "dns_nameservers": True}
    assert stack.parse_update_allowed(None) == {}


def test_updatable_change_is_downgraded_to_an_update() -> None:
    changes = ChangeSet(replaced=("subnet",))
    candidates = stack.in_place_candidates(changes, OLD, _with(dns_nameservers=["x"]), {}, {})
    assert candidates == {"subnet": {"dns_nameservers"}}
    allowed = {"subnet": stack.parse_update_allowed(SCHEMA)}
    corrected, moved = stack.downgrade_replacements(changes, candidates, allowed)
    assert corrected.updated == ("subnet",) and corrected.replaced == () and moved == ("subnet",)


def test_non_updatable_change_stays_a_replacement() -> None:
    changes = ChangeSet(replaced=("subnet",))
    candidates = stack.in_place_candidates(
        changes, OLD, _with(cidr="b", dns_nameservers=["x"]), {}, {}
    )
    allowed = {"subnet": stack.parse_update_allowed(SCHEMA)}
    corrected, moved = stack.downgrade_replacements(changes, candidates, allowed)
    assert corrected.replaced == ("subnet",) and moved == ()


def test_unknown_or_structural_changes_are_never_downgraded() -> None:
    changes = ChangeSet(replaced=("subnet",))
    retyped = json.loads(json.dumps(OLD))
    retyped["resources"]["subnet"]["type"] = "OS::Other"
    assert stack.in_place_candidates(changes, OLD, retyped, {}, {}) == {}
    candidates = {"subnet": {"unknown_property"}}
    corrected, _ = stack.downgrade_replacements(changes, candidates, {"subnet": {}})
    assert corrected.replaced == ("subnet",)
