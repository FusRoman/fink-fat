"""Template of the ``storage`` stack: the persistent data volumes, and their re-adoption.

Every data volume is an ``OS::Cinder::Volume`` with ``deletion_policy: Retain``: deleting
the stack (or the resource) keeps the volume and its data. Such a kept volume is then no
longer managed by any stack. To manage it again without creating a new, empty volume, the
template gives each volume a parameter ``<resource>_existing_id``:

* empty (the default) — Heat creates the volume;
* set to a volume ID — Heat creates nothing (the resource's condition is false) and the
  stack output ``<resource>_id`` returns that ID: the existing volume is *adopted*.

The template itself never depends on the cloud; :func:`resolve_adoptions` decides the
parameter values from what the cloud reports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from finkfat_ops import naming
from finkfat_ops.config import InfraConfig
from finkfat_ops.naming import Role


@dataclass(frozen=True)
class DataVolume:
    """A persistent data volume, as described by ``infra.yaml``.

    Attributes
    ----------
    role : Role
        Machine the volume is attached to.
    resource : str
        Logical resource name in the template, e.g. ``vol_pgdata``.
    name : str
        Cinder volume name, e.g. ``ffat-prod-vol-pgdata``.
    size_gb : int
        Size requested at creation.
    """

    role: Role
    resource: str
    name: str
    size_gb: int

    @property
    def parameter(self) -> str:
        """Name of the template parameter holding the ID of an adopted volume."""
        return f"{self.resource}_existing_id"

    @property
    def condition(self) -> str:
        """Name of the template condition that is true when the volume is adopted."""
        return f"adopt_{self.resource}"

    @property
    def output(self) -> str:
        """Name of the stack output holding the volume ID (created or adopted)."""
        return f"{self.resource}_id"


def data_volumes(infra: InfraConfig) -> list[DataVolume]:
    """List the data volumes of every role, enabled or not (a disabled role keeps its data).

    Parameters
    ----------
    infra : InfraConfig
        Versioned structure.

    Returns
    -------
    list[DataVolume]
        One entry per role that has a data volume, in role order.
    """
    return [
        DataVolume(
            role=role,
            resource=f"vol_{cfg.volume.name.replace('-', '_')}",
            name=naming.data_volume_name(infra.env, cfg.volume.name),
            size_gb=cfg.volume.size_gb,
        )
        for role, cfg in infra.roles.items()
        if cfg.volume is not None
    ]


def protected_resources(infra: InfraConfig) -> frozenset[str]:
    """Return the logical names of the volumes, which must never be replaced or deleted.

    Parameters
    ----------
    infra : InfraConfig
        Versioned structure.

    Returns
    -------
    frozenset[str]
        Every volume resource of the template.
    """
    return frozenset(volume.resource for volume in data_volumes(infra))


def build_storage_template(infra: InfraConfig) -> dict[str, Any]:
    """Build the HOT template of the ``storage`` stack.

    Parameters
    ----------
    infra : InfraConfig
        Versioned structure (environment, roles, volume type).

    Returns
    -------
    dict[str, Any]
        The template, ready for :func:`finkfat_ops.heat.render.render_template`.
    """
    volumes = data_volumes(infra)
    parameters = {
        volume.parameter: {
            "type": "string",
            "default": "",
            "description": (
                f"ID of an existing {volume.name} to adopt instead of creating one "
                "(set by ff-ops; empty to create the volume)."
            ),
        }
        for volume in volumes
    }
    conditions = {
        volume.condition: {"not": {"equals": [{"get_param": volume.parameter}, ""]}}
        for volume in volumes
    }
    resources = {
        volume.resource: {
            "type": "OS::Cinder::Volume",
            "deletion_policy": "Retain",
            "condition": {"not": volume.condition},
            "properties": {
                "name": volume.name,
                "size": volume.size_gb,
                "volume_type": infra.volume_type,
                "description": f"fink-fat {infra.env}: data of the {volume.role} machine",
            },
        }
        for volume in volumes
    }
    outputs = {
        volume.output: {
            "description": f"ID of {volume.name} (created or adopted)",
            "value": {
                "if": [
                    volume.condition,
                    {"get_param": volume.parameter},
                    {"get_resource": volume.resource},
                ]
            },
        }
        for volume in volumes
    }
    return {
        "heat_template_version": infra.heat_template_version,
        "description": f"fink-fat {infra.env}: persistent data volumes (kept on deletion)",
        "parameters": parameters,
        "conditions": conditions,
        "resources": resources,
        "outputs": outputs,
    }


# --------------------------------------------------------------------------- adoption


@dataclass(frozen=True)
class Adoption:
    """An existing volume that the stack takes over instead of creating a new one.

    Attributes
    ----------
    volume : DataVolume
        The data volume of ``infra.yaml``.
    volume_id : str
        ID of the existing Cinder volume.
    """

    volume: DataVolume
    volume_id: str


class AdoptionError(ValueError):
    """The volumes found in the cloud do not allow a safe decision."""


def resolve_adoptions(
    volumes: Sequence[DataVolume],
    found: Mapping[str, Sequence[str]],
    stack_parameters: Mapping[str, str] | None,
    stack_outputs: Mapping[str, Any] | None,
) -> list[Adoption]:
    """Decide which existing volumes the stack must adopt.

    For each data volume:

    * no Cinder volume has its name — none adopted, the stack creates it;
    * the stack already adopted that very volume — it stays adopted;
    * the stack created that very volume — nothing to adopt;
    * otherwise (typically after ``infra destroy storage``, whose ``Retain`` kept the
      volume) — the volume is adopted.

    Parameters
    ----------
    volumes : Sequence[DataVolume]
        Data volumes of ``infra.yaml``.
    found : Mapping[str, Sequence[str]]
        IDs of the Cinder volumes carrying each volume name.
    stack_parameters : Mapping[str, str] | None
        Current parameters of the storage stack; ``None`` when it does not exist.
    stack_outputs : Mapping[str, Any] | None
        Current outputs of the storage stack; ``None`` when it does not exist.

    Returns
    -------
    list[Adoption]
        The volumes to adopt, in the order of ``volumes``.

    Raises
    ------
    AdoptionError
        Several volumes share a name, or a volume adopted by the stack has disappeared:
        guessing could attach an empty volume in place of the data.
    """
    parameters = stack_parameters or {}
    outputs = stack_outputs or {}
    adoptions: list[Adoption] = []
    for volume in volumes:
        ids = list(found.get(volume.name, ()))
        adopted_id = parameters.get(volume.parameter) or ""
        if len(ids) > 1:
            raise AdoptionError(
                f"{len(ids)} volumes are named {volume.name} ({', '.join(ids)}): "
                "rename or delete the extra ones (after checking their content), then retry"
            )
        if not ids:
            if adopted_id:
                raise AdoptionError(
                    f"volume {volume.name} ({adopted_id}), adopted by the storage stack, "
                    "no longer exists: restore it from a snapshot, or clear the adoption "
                    "deliberately before creating an empty volume"
                )
            continue
        (volume_id,) = ids
        if adopted_id and adopted_id != volume_id:
            raise AdoptionError(
                f"the storage stack adopted {volume.name} {adopted_id}, but the volume of "
                f"that name is now {volume_id}: check which one holds the data"
            )
        created_id = None if adopted_id else outputs.get(volume.output)
        if volume_id != created_id:
            adoptions.append(Adoption(volume, volume_id))
    return adoptions


def adoption_parameters(
    volumes: Sequence[DataVolume], adoptions: Sequence[Adoption]
) -> dict[str, str]:
    """Return the value of every ``<resource>_existing_id`` parameter.

    Every parameter is given (empty when the volume is created), so that an update never
    silently keeps a stale value.

    Parameters
    ----------
    volumes : Sequence[DataVolume]
        Data volumes of ``infra.yaml``.
    adoptions : Sequence[Adoption]
        Output of :func:`resolve_adoptions`.

    Returns
    -------
    dict[str, str]
        Parameter name to volume ID (or ``""``).
    """
    adopted = {adoption.volume.resource: adoption.volume_id for adoption in adoptions}
    return {volume.parameter: adopted.get(volume.resource, "") for volume in volumes}
