"""Heat stack commands, parsing of their results, and the replacement guard.

All functions are pure: they build :class:`~finkfat_ops.runner.Command` values or
interpret JSON already returned by ``openstack``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
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


def outputs_command(cli: OpenStackCli, stack: str) -> Command:
    """``openstack stack output show <stack> --all``."""
    return cli.command("stack", "output", "show", stack, "--all")


def preview_update_command(cli: OpenStackCli, stack: str, template: Path) -> Command:
    """``openstack stack update --dry-run``: list the changes without applying them."""
    return cli.command("stack", "update", "--dry-run", "--show-nested", "-t", str(template), stack)


def create_command(cli: OpenStackCli, stack: str, template: Path) -> Command:
    """``openstack stack create --wait`` (rollback on failure is Heat's default)."""
    return cli.command(
        "stack", "create", "--wait", "-t", str(template), stack, timeout_s=WAIT_TIMEOUT_S
    )


def update_command(cli: OpenStackCli, stack: str, template: Path) -> Command:
    """``openstack stack update --wait``: apply the template; only differences change."""
    return cli.command(
        "stack", "update", "--wait", "-t", str(template), stack, timeout_s=WAIT_TIMEOUT_S
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
    """

    name: str
    status: str
    reason: str
    outputs: dict[str, Any] = field(default_factory=dict)

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
        Name, status, reason and outputs.
    """
    return StackInfo(
        name=data["stack_name"],
        status=data["stack_status"],
        reason=data.get("stack_status_reason") or "",
        outputs=parse_outputs(data.get("outputs")),
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


def change_set_for_creation(template: dict[str, Any]) -> ChangeSet:
    """Describe the creation of a new stack as a change set (every resource added).

    Parameters
    ----------
    template : dict[str, Any]
        The template to create.

    Returns
    -------
    ChangeSet
        All resources of the template in ``added``.
    """
    return ChangeSet(added=tuple(sorted(template.get("resources", {}))))


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
