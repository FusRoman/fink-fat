"""Tests of the storage stack template and of the volume adoption decision."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from finkfat_ops.config import parse_infra
from finkfat_ops.heat.render import render_template
from finkfat_ops.heat.storage import (
    Adoption,
    AdoptionError,
    DataVolume,
    adoption_parameters,
    build_storage_template,
    data_volumes,
    protected_resources,
    resolve_adoptions,
)
from finkfat_ops.naming import Role


@pytest.fixture
def volumes(infra_data: dict[str, Any]) -> list[DataVolume]:
    return data_volumes(parse_infra(infra_data))


@pytest.fixture
def template(infra_data: dict[str, Any]) -> dict[str, Any]:
    return build_storage_template(parse_infra(infra_data))


def _pgdata(volumes: list[DataVolume]) -> DataVolume:
    return next(v for v in volumes if v.role == Role.DB)


def test_one_volume_per_role_with_data(volumes: list[DataVolume]) -> None:
    assert [(v.role, v.resource, v.name, v.size_gb) for v in volumes] == [
        (Role.COMPUTE, "vol_engine", "ffat-prod-vol-engine", 200),
        (Role.DB, "vol_pgdata", "ffat-prod-vol-pgdata", 500),
    ]


def test_disabled_role_keeps_its_volume(infra_data: dict[str, Any]) -> None:
    infra_data["roles"]["compute"]["enabled"] = False
    names = [v.name for v in data_volumes(parse_infra(infra_data))]
    assert "ffat-prod-vol-engine" in names


def test_volumes_are_retained_and_protected(
    template: dict[str, Any], infra_data: dict[str, Any]
) -> None:
    resources = template["resources"]
    assert set(resources) == {"vol_engine", "vol_pgdata"}
    assert protected_resources(parse_infra(infra_data)) == frozenset(resources)
    pgdata = resources["vol_pgdata"]
    assert pgdata["type"] == "OS::Cinder::Volume"
    assert pgdata["deletion_policy"] == "Retain"
    assert pgdata["properties"]["name"] == "ffat-prod-vol-pgdata"
    assert pgdata["properties"]["size"] == 500
    assert pgdata["properties"]["volume_type"] == "__DEFAULT__"


def test_adoption_is_driven_by_a_parameter(template: dict[str, Any]) -> None:
    assert template["parameters"]["vol_pgdata_existing_id"]["default"] == ""
    assert template["conditions"]["adopt_vol_pgdata"] == {
        "not": {"equals": [{"get_param": "vol_pgdata_existing_id"}, ""]}
    }
    assert template["resources"]["vol_pgdata"]["condition"] == {"not": "adopt_vol_pgdata"}
    assert template["outputs"]["vol_pgdata_id"]["value"] == {
        "if": [
            "adopt_vol_pgdata",
            {"get_param": "vol_pgdata_existing_id"},
            {"get_resource": "vol_pgdata"},
        ]
    }


def test_rendered_template_is_valid_yaml(template: dict[str, Any]) -> None:
    assert yaml.safe_load(render_template(template)) == template


def test_new_volumes_are_created(volumes: list[DataVolume]) -> None:
    assert resolve_adoptions(volumes, {}, None, None) == []


def test_orphan_volume_is_adopted(volumes: list[DataVolume]) -> None:
    adoptions = resolve_adoptions(volumes, {"ffat-prod-vol-pgdata": ["v1"]}, None, None)
    assert adoptions == [Adoption(_pgdata(volumes), "v1")]
    assert adoption_parameters(volumes, adoptions) == {
        "vol_engine_existing_id": "",
        "vol_pgdata_existing_id": "v1",
    }


def test_volume_created_by_the_stack_is_left_alone(volumes: list[DataVolume]) -> None:
    found = {"ffat-prod-vol-pgdata": ["v1"]}
    parameters = {"vol_pgdata_existing_id": ""}
    assert resolve_adoptions(volumes, found, parameters, {"vol_pgdata_id": "v1"}) == []


def test_adopted_volume_stays_adopted(volumes: list[DataVolume]) -> None:
    found = {"ffat-prod-vol-pgdata": ["v1"]}
    parameters = {"vol_pgdata_existing_id": "v1"}
    outputs = {"vol_pgdata_id": "v1"}
    assert resolve_adoptions(volumes, found, parameters, outputs) == [
        Adoption(_pgdata(volumes), "v1")
    ]


def test_duplicate_names_are_refused(volumes: list[DataVolume]) -> None:
    with pytest.raises(AdoptionError, match="2 volumes are named ffat-prod-vol-pgdata"):
        resolve_adoptions(volumes, {"ffat-prod-vol-pgdata": ["v1", "v2"]}, None, None)


def test_vanished_adopted_volume_is_refused(volumes: list[DataVolume]) -> None:
    with pytest.raises(AdoptionError, match="no longer exists"):
        resolve_adoptions(volumes, {}, {"vol_pgdata_existing_id": "v1"}, None)


def test_swapped_adopted_volume_is_refused(volumes: list[DataVolume]) -> None:
    found = {"ffat-prod-vol-pgdata": ["v2"]}
    with pytest.raises(AdoptionError, match="check which one holds the data"):
        resolve_adoptions(volumes, found, {"vol_pgdata_existing_id": "v1"}, None)
