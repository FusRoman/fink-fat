"""Heat stack commands, parsing of their results, and the replacement guard.

All functions are pure: they build :class:`~finkfat_ops.runner.Command` values or
interpret JSON already returned by ``openstack``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finkfat_ops.openstack import WAIT_TIMEOUT_S, OpenStackCli
from finkfat_ops.runner import Command

CHANGE_KINDS = ("added", "updated", "replaced", "deleted", "unchanged")
"""Categories of ``openstack stack update --dry-run``, in display order."""

DESTRUCTIVE_KINDS = ("replaced", "deleted")
"""Changes that destroy a resource (a replacement deletes, then recreates)."""


# ------------------------------------------------------------------------ commands


def show_command(cli: OpenStackCli, stack: str) -> Command:
    """``openstack stack show <stack>`` (status and outputs)."""
    return cli.command("stack", "show", stack)


def template_command(cli: OpenStackCli, stack: str) -> Command:
    """``openstack stack template show <stack>``: the template currently deployed."""
    return cli.command("stack", "template", "show", stack)


def resource_type_command(cli: OpenStackCli, resource_type: str) -> Command:
    """``openstack orchestration resource type show <type>``: Heat's schema of a type."""
    return cli.command("orchestration", "resource", "type", "show", resource_type)


def outputs_command(cli: OpenStackCli, stack: str) -> Command:
    """``openstack stack output show <stack> --all``."""
    return cli.command("stack", "output", "show", stack, "--all")


def _parameter_args(parameters: Mapping[str, str]) -> list[str]:
    """Turn template parameters into ``--parameter key=value`` arguments (sorted)."""
    return [
        arg for key in sorted(parameters) for arg in ("--parameter", f"{key}={parameters[key]}")
    ]


def preview_update_command(
    cli: OpenStackCli, stack: str, template: Path, parameters: Mapping[str, str] | None = None
) -> Command:
    """``openstack stack update --dry-run``: list the changes without applying them."""
    return cli.command(
        "stack",
        "update",
        "--dry-run",
        "--show-nested",
        "-t",
        str(template),
        *_parameter_args(parameters or {}),
        stack,
    )


def create_command(
    cli: OpenStackCli, stack: str, template: Path, parameters: Mapping[str, str] | None = None
) -> Command:
    """``openstack stack create --wait`` (rollback on failure is Heat's default)."""
    return cli.command(
        "stack",
        "create",
        "--wait",
        "-t",
        str(template),
        *_parameter_args(parameters or {}),
        stack,
        timeout_s=WAIT_TIMEOUT_S,
    )


def update_command(
    cli: OpenStackCli, stack: str, template: Path, parameters: Mapping[str, str] | None = None
) -> Command:
    """``openstack stack update --wait``: apply the template; only differences change.

    Every parameter of the template must be given: an update without ``--existing``
    resets the missing ones to their default.
    """
    return cli.command(
        "stack",
        "update",
        "--wait",
        "-t",
        str(template),
        *_parameter_args(parameters or {}),
        stack,
        timeout_s=WAIT_TIMEOUT_S,
    )


def delete_command(cli: OpenStackCli, stack: str) -> Command:
    """``openstack stack delete --yes --wait`` (resources with ``Retain`` survive)."""
    return cli.command(
        "stack", "delete", "--yes", "--wait", stack, json_output=False, timeout_s=WAIT_TIMEOUT_S
    )


# ------------------------------------------------------------------------- parsing


@dataclass(frozen=True)
class StackInfo:
    """Status of an existing stack.

    Attributes
    ----------
    name : str
        Stack name.
    status : str
        Heat status, e.g. ``CREATE_COMPLETE`` or ``UPDATE_FAILED``.
    reason : str
        Heat's explanation of the status (the error, for a failed stack).
    outputs : dict[str, Any]
        Output values by key.
    parameters : dict[str, str]
        Current template parameters (Heat's own ``OS::*`` entries included).
    """

    name: str
    status: str
    reason: str
    outputs: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, str] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        """Whether the last operation on the stack completed successfully."""
        return self.status.endswith("_COMPLETE") and not self.status.startswith("DELETE")


def _decode(value: Any) -> Any:
    """Decode a value that ``openstack -f json`` may have printed as a JSON string."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def parse_outputs(data: Any) -> dict[str, Any]:
    """Turn stack outputs (``stack show`` list or ``output show --all`` mapping) into a dict.

    Parameters
    ----------
    data : Any
        Either a list of ``{"output_key", "output_value"}`` objects or a mapping from
        output key to such an object.

    Returns
    -------
    dict[str, Any]
        Output values by key.
    """
    data = _decode(data) or []
    entries = data.values() if isinstance(data, dict) else data
    outputs = {}
    for entry in entries:
        item = _decode(entry)
        outputs[item["output_key"]] = item.get("output_value")
    return outputs


def parse_stack_show(data: dict[str, Any]) -> StackInfo:
    """Build a :class:`StackInfo` from ``openstack stack show -f json``.

    Parameters
    ----------
    data : dict[str, Any]
        Decoded JSON output.

    Returns
    -------
    StackInfo
        Name, status, reason, outputs and parameters.
    """
    parameters = _decode(data.get("parameters")) or {}
    return StackInfo(
        name=data["stack_name"],
        status=data["stack_status"],
        reason=data.get("stack_status_reason") or "",
        outputs=parse_outputs(data.get("outputs")),
        parameters={str(key): str(value) for key, value in parameters.items()},
    )


@dataclass(frozen=True)
class ChangeSet:
    """Resource changes announced by ``openstack stack update --dry-run``.

    Each attribute holds logical resource names (the keys of ``resources``).

    Attributes
    ----------
    added, updated, replaced, deleted, unchanged : tuple[str, ...]
        Resources per kind of change.
    """

    added: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    replaced: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Whether nothing would change."""
        return not (self.added or self.updated or self.replaced or self.deleted)

    def changed(self) -> set[str]:
        """Names of every resource that would be added, updated, replaced or deleted."""
        return {*self.added, *self.updated, *self.replaced, *self.deleted}


def _resource_name(item: Any) -> str:
    """Extract the logical name of one dry-run entry (a resource object or a plain name)."""
    item = _decode(item)
    return item["resource_name"] if isinstance(item, dict) else str(item)


def parse_change_set(data: Any) -> ChangeSet:
    """Build a :class:`ChangeSet` from ``openstack stack update --dry-run -f json``.

    Parameters
    ----------
    data : Any
        Decoded JSON output: a mapping from change kind to a list of resources.

    Returns
    -------
    ChangeSet
        Sorted resource names per kind; unknown kinds are ignored.
    """
    data = _decode(data) or {}
    return ChangeSet(
        **{
            kind: tuple(sorted(_resource_name(item) for item in (_decode(data.get(kind)) or [])))
            for kind in CHANGE_KINDS
        }
    )


def change_set_for_creation(template: dict[str, Any], skipped: Iterable[str] = ()) -> ChangeSet:
    """Describe the creation of a new stack as a change set (every resource added).

    Parameters
    ----------
    template : dict[str, Any]
        The template to create.
    skipped : Iterable[str]
        Resources whose condition is false for this creation (e.g. adopted volumes).

    Returns
    -------
    ChangeSet
        The resources of the template that will be created, in ``added``.
    """
    created = set(template.get("resources", {})) - set(skipped)
    return ChangeSet(added=tuple(sorted(created)))


def parse_template(data: Any) -> dict[str, Any]:
    """Decode the output of ``openstack stack template show -f json``.

    Parameters
    ----------
    data : Any
        Decoded JSON output; nested sections may come as JSON strings.

    Returns
    -------
    dict[str, Any]
        The template; ``{}`` for an empty output (dry-run).
    """
    return {key: _decode(value) for key, value in (_decode(data) or {}).items()}


def _parameters_used(node: Any) -> set[str]:
    """Collect the names of every ``get_param`` inside a template fragment."""
    if isinstance(node, dict):
        found = {str(node["get_param"])} if "get_param" in node else set()
        for value in node.values():
            found |= _parameters_used(value)
        return found
    if isinstance(node, list):
        return set().union(*(_parameters_used(item) for item in node))
    return set()


def _conditions_used(condition: Any) -> set[str]:
    """Collect the condition names a resource's ``condition`` entry refers to."""
    if isinstance(condition, str):
        return {condition}
    if isinstance(condition, dict):
        return set().union(set(), *(_conditions_used(value) for value in condition.values()))
    if isinstance(condition, list):
        return set().union(set(), *(_conditions_used(item) for item in condition))
    return set()


def really_changed(
    deployed: dict[str, Any],
    template: dict[str, Any],
    deployed_parameters: dict[str, str],
    parameters: dict[str, str],
) -> set[str]:
    """List the resources that a template and its parameters actually change (pure).

    A resource changes when its definition differs from the deployed one, when it appears
    or disappears, or when it uses (directly, or through its ``condition``) a parameter
    whose value differs.

    ``openstack stack update --dry-run`` also reports as updated or replaced every resource
    that merely references another resource with ``get_resource``, even when nothing
    changed. This function is the reference used to discard those false reports. A real
    replacement still shows up through the resource whose own definition changed.

    Parameters
    ----------
    deployed : dict[str, Any]
        Template currently deployed (:func:`parse_template`).
    template : dict[str, Any]
        Template about to be applied.
    deployed_parameters : dict[str, str]
        Current parameter values of the stack.
    parameters : dict[str, str]
        Parameter values about to be passed.

    Returns
    -------
    set[str]
        Logical names of the resources that really change.
    """
    old = deployed.get("resources") or {}
    new = template.get("resources") or {}
    changed_parameters = _changed_parameters(deployed_parameters, parameters)
    changed = set(old.keys() ^ new.keys())
    for name in old.keys() & new.keys():
        used = _resource_parameters(new[name], template)
        if old[name] != new[name] or used & changed_parameters:
            changed.add(name)
    return changed


def _changed_parameters(deployed: dict[str, str], parameters: dict[str, str]) -> set[str]:
    """Names of the parameters whose value differs from the deployed one."""
    return {name for name, value in parameters.items() if deployed.get(name) != value}


def _resource_parameters(definition: dict[str, Any], template: dict[str, Any]) -> set[str]:
    """Parameters a resource uses, in its properties or through its condition."""
    conditions = template.get("conditions") or {}
    used = _parameters_used(definition.get("properties"))
    for condition in _conditions_used(definition.get("condition")):
        used |= _parameters_used(conditions.get(condition))
    return used


def parse_update_allowed(data: Any) -> dict[str, bool]:
    """Read which properties of a resource type can change in place.

    Parameters
    ----------
    data : Any
        Decoded JSON of ``openstack orchestration resource type show -f json``.

    Returns
    -------
    dict[str, bool]
        Property name to its ``update_allowed`` flag; ``{}`` for an empty output.
    """
    properties = _decode((_decode(data) or {}).get("properties")) or {}
    allowed = {}
    for name, schema in properties.items():
        decoded = _decode(schema)
        if isinstance(decoded, dict):
            allowed[name] = bool(decoded.get("update_allowed"))
    return allowed


def in_place_candidates(
    changes: ChangeSet,
    deployed: dict[str, Any],
    template: dict[str, Any],
    deployed_parameters: dict[str, str],
    parameters: dict[str, str],
) -> dict[str, set[str]]:
    """Find the replacements that may in fact be in-place updates, with their changed properties.

    A replacement announced by Heat's preview qualifies when the resource's type and every
    other key of its definition are unchanged, only literal property values differ, and it
    uses no changed parameter. Heat's preview can announce such a replacement because of a
    reference to another resource (``get_resource``), which it compares unresolved.

    Parameters
    ----------
    changes : ChangeSet
        Confirmed changes (after :func:`filter_change_set`).
    deployed : dict[str, Any]
        Template currently deployed.
    template : dict[str, Any]
        Template about to be applied.
    deployed_parameters : dict[str, str]
        Current parameter values of the stack.
    parameters : dict[str, str]
        Parameter values about to be passed.

    Returns
    -------
    dict[str, set[str]]
        Resource name to the names of its changed properties.
    """
    old = deployed.get("resources") or {}
    new = template.get("resources") or {}
    changed_parameters = _changed_parameters(deployed_parameters, parameters)
    candidates = {}
    for name in changes.replaced:
        before, after = old.get(name), new.get(name)
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        if {k: v for k, v in before.items() if k != "properties"} != {
            k: v for k, v in after.items() if k != "properties"
        }:
            continue
        if _resource_parameters(after, template) & changed_parameters:
            continue
        old_props, new_props = before.get("properties") or {}, after.get("properties") or {}
        differing = {
            prop
            for prop in old_props.keys() | new_props.keys()
            if old_props.get(prop) != new_props.get(prop)
        }
        if differing:
            candidates[name] = differing
    return candidates


def downgrade_replacements(
    changes: ChangeSet,
    candidates: dict[str, set[str]],
    update_allowed: dict[str, dict[str, bool]],
) -> tuple[ChangeSet, tuple[str, ...]]:
    """Turn into updates the replacements whose changed properties all update in place.

    This is the rule Heat itself applies during a real update: a resource is replaced only
    when a changed property is not ``update_allowed`` in the schema of its type.

    Parameters
    ----------
    changes : ChangeSet
        Confirmed changes.
    candidates : dict[str, set[str]]
        Output of :func:`in_place_candidates`.
    update_allowed : dict[str, dict[str, bool]]
        Resource name to the ``update_allowed`` flags of its type (:func:`parse_update_allowed`).

    Returns
    -------
    tuple[ChangeSet, tuple[str, ...]]
        The corrected changes, and the resources moved from replaced to updated (sorted).
    """
    downgraded = sorted(
        name
        for name, properties in candidates.items()
        if name in changes.replaced
        and all(update_allowed.get(name, {}).get(prop, False) for prop in properties)
    )
    if not downgraded:
        return changes, ()
    corrected = ChangeSet(
        added=changes.added,
        updated=tuple(sorted({*changes.updated, *downgraded})),
        replaced=tuple(name for name in changes.replaced if name not in downgraded),
        deleted=changes.deleted,
        unchanged=changes.unchanged,
    )
    return corrected, tuple(downgraded)


def filter_change_set(changes: ChangeSet, changed: set[str]) -> tuple[ChangeSet, tuple[str, ...]]:
    """Discard the updates and replacements that :func:`really_changed` does not confirm.

    Added and deleted resources are always kept (they follow from the templates and
    conditions themselves).

    Parameters
    ----------
    changes : ChangeSet
        Changes announced by ``openstack stack update --dry-run``.
    changed : set[str]
        Resources that really change.

    Returns
    -------
    tuple[ChangeSet, tuple[str, ...]]
        The confirmed changes, and the names discarded (sorted).
    """
    ignored = sorted({*changes.updated, *changes.replaced} - changed)
    kept = ChangeSet(
        added=changes.added,
        updated=tuple(name for name in changes.updated if name in changed),
        replaced=tuple(name for name in changes.replaced if name in changed),
        deleted=changes.deleted,
        unchanged=tuple(sorted({*changes.unchanged, *ignored})),
    )
    return kept, tuple(ignored)


# ------------------------------------------------------------------- replacement guard


@dataclass(frozen=True)
class GuardViolation:
    """A protected resource that a change would destroy.

    Attributes
    ----------
    resource : str
        Logical resource name.
    kind : str
        ``replaced`` or ``deleted``.
    """

    resource: str
    kind: str


def guard_violations(
    changes: ChangeSet, protected: Iterable[str], allowed: Iterable[str] = ()
) -> list[GuardViolation]:
    """List the protected resources that ``changes`` would replace or delete.

    Parameters
    ----------
    changes : ChangeSet
        The previewed changes.
    protected : Iterable[str]
        Logical names that must not be destroyed (volumes, db and compute VMs,
        floating IP, network…).
    allowed : Iterable[str]
        Protected names the operator explicitly accepted to destroy
        (``--allow-replace``).

    Returns
    -------
    list[GuardViolation]
        One entry per destroyed protected resource not in ``allowed``; empty when safe.
    """
    blocked = set(protected) - set(allowed)
    return [
        GuardViolation(resource, kind)
        for kind in DESTRUCTIVE_KINDS
        for resource in getattr(changes, kind)
        if resource in blocked
    ]
