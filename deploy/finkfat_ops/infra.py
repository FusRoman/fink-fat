"""Orchestration of the Heat stacks: render, plan, apply, show, destroy.

The decisions (which template, what is protected, whether a change is safe) are made
by pure functions of :mod:`finkfat_ops.heat` and :mod:`finkfat_ops.stack`; this module
sequences the commands and the operator's confirmations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finkfat_ops import naming
from finkfat_ops import stack as stack_ops
from finkfat_ops.access import AccessState, effective_ssh_cidrs
from finkfat_ops.config import Deployment, DeployPaths
from finkfat_ops.heat.net import PROTECTED_RESOURCES as NET_PROTECTED
from finkfat_ops.heat.net import build_net_template
from finkfat_ops.heat.render import render_template
from finkfat_ops.naming import Stack
from finkfat_ops.openstack import OpenStackCli, is_not_found, parse_json
from finkfat_ops.runner import CommandError, Runner
from finkfat_ops.stack import ChangeSet, GuardViolation, StackInfo


class InfraError(RuntimeError):
    """A stack operation was refused or failed."""


@dataclass(frozen=True)
class StackSpec:
    """Everything needed to deploy one stack.

    Attributes
    ----------
    stack : Stack
        Which stack.
    name : str
        Heat stack name, e.g. ``ffat-prod-net``.
    template : dict[str, Any]
        The HOT template.
    protected : frozenset[str]
        Logical resources that must never be replaced or deleted silently.
    """

    stack: Stack
    name: str
    template: dict[str, Any]
    protected: frozenset[str]


IMPLEMENTED_STACKS = (Stack.NET,)
"""Stacks that ff-ops can build so far."""


def build_stack_spec(stack: Stack, deployment: Deployment, access: AccessState) -> StackSpec:
    """Build the template and protection list of a stack (pure).

    Parameters
    ----------
    stack : Stack
        Which stack.
    deployment : Deployment
        Validated configuration.
    access : AccessState
        Dynamic SSH ranges.

    Returns
    -------
    StackSpec
        The stack specification.

    Raises
    ------
    InfraError
        The stack is not implemented yet.
    """
    name = naming.stack_name(deployment.infra.env, stack)
    if stack == Stack.NET:
        cidrs = effective_ssh_cidrs(deployment.local, access)
        return StackSpec(stack, name, build_net_template(deployment.infra, cidrs), NET_PROTECTED)
    raise InfraError(f"stack '{stack}' is not implemented yet")


def describe_changes(changes: ChangeSet) -> list[str]:
    """Format a change set as one line per kind of change (pure).

    Parameters
    ----------
    changes : ChangeSet
        The previewed changes.

    Returns
    -------
    list[str]
        Lines such as ``"+ added: net, subnet"``; ``["no change"]`` when empty.
    """
    if changes.is_empty:
        return ["no change"]
    symbols = {"added": "+", "updated": "~", "replaced": "!", "deleted": "-"}
    return [
        f"{symbol} {kind}: {', '.join(getattr(changes, kind))}"
        for kind, symbol in symbols.items()
        if getattr(changes, kind)
    ]


def describe_violations(spec: StackSpec, violations: list[GuardViolation]) -> str:
    """Explain why the replacement guard refused a change (pure)."""
    listed = ", ".join(f"{v.resource} ({v.kind})" for v in violations)
    names = " ".join(f"--allow-replace {v.resource}" for v in violations)
    return (
        f"refused: this change would destroy protected resources of {spec.name}: {listed}. "
        f"If this is really intended, re-run with {names} (after a snapshot of any data volume)."
    )


@dataclass
class InfraSession:
    """What a stack operation needs to talk to the cloud and to the operator.

    Attributes
    ----------
    deployment : Deployment
        Validated configuration.
    paths : DeployPaths
        File locations (templates are written to ``.build/``).
    cli : OpenStackCli
        Command builder bound to the cloud profile.
    runner : Runner
        Runs the commands.
    echo : Callable[[str], None]
        Prints a line for the operator.
    confirm : Callable[[str], bool]
        Asks the operator a yes/no question.
    """

    deployment: Deployment
    paths: DeployPaths
    cli: OpenStackCli
    runner: Runner
    echo: Callable[[str], None]
    confirm: Callable[[str], bool]


def render(session: InfraSession, spec: StackSpec) -> Path:
    """Write the template of ``spec`` to ``.build/<stack>.yaml`` and return its path."""
    session.paths.build_dir.mkdir(parents=True, exist_ok=True)
    path = session.paths.build_dir / f"{spec.stack}.yaml"
    path.write_text(render_template(spec.template), encoding="utf-8")
    return path


def get_stack(session: InfraSession, name: str) -> StackInfo | None:
    """Return the status of a stack, or ``None`` if it does not exist (or in dry-run).

    Raises
    ------
    CommandError
        The query failed for another reason than "not found".
    """
    result = session.runner.run(stack_ops.show_command(session.cli, name), check=False)
    if is_not_found(result):
        return None
    if not result.ok:
        raise CommandError(stack_ops.show_command(session.cli, name), result.stderr.strip(), result)
    data = parse_json(result)
    return None if data is None else stack_ops.parse_stack_show(data)


@dataclass(frozen=True)
class Plan:
    """Outcome of :func:`plan`.

    Attributes
    ----------
    spec : StackSpec
        The stack.
    template_path : Path
        Rendered template.
    existing : StackInfo | None
        Current stack, or ``None`` when it would be created.
    changes : ChangeSet
        What would change.
    """

    spec: StackSpec
    template_path: Path
    existing: StackInfo | None
    changes: ChangeSet


def plan(session: InfraSession, spec: StackSpec) -> Plan:
    """Render the template and preview the changes, without modifying anything.

    Parameters
    ----------
    session : InfraSession
        Cloud access.
    spec : StackSpec
        The stack.

    Returns
    -------
    Plan
        The previewed changes (every resource added for a new stack).
    """
    path = render(session, spec)
    existing = get_stack(session, spec.name)
    if existing is None:
        return Plan(spec, path, None, stack_ops.change_set_for_creation(spec.template))
    result = session.runner.run(stack_ops.preview_update_command(session.cli, spec.name, path))
    return Plan(spec, path, existing, stack_ops.parse_change_set(parse_json(result)))


def apply(
    session: InfraSession,
    spec: StackSpec,
    *,
    allow_replace: Iterable[str] = (),
    only: Iterable[str] | None = None,
    assume_yes: bool = False,
) -> StackInfo | None:
    """Create or update a stack after a preview, the guard and the operator's consent.

    Parameters
    ----------
    session : InfraSession
        Cloud access.
    spec : StackSpec
        The stack.
    allow_replace : Iterable[str]
        Protected resources the operator accepts to replace or delete.
    only : Iterable[str] | None
        If given, refuse unless every changed resource is in this set (used by
        ``ff-ops access`` to touch nothing but the SSH rules).
    assume_yes : bool
        Skip the confirmation question.

    Returns
    -------
    StackInfo | None
        The stack after the operation (``None`` in dry-run).

    Raises
    ------
    InfraError
        Guard violation, change outside ``only``, operator refusal, or failed stack.
    """
    planned = plan(session, spec)
    for line in describe_changes(planned.changes):
        session.echo(f"  {line}")
    if planned.existing is not None and planned.changes.is_empty:
        session.echo(f"{spec.name} is up to date")
        return planned.existing
    if only is not None:
        unexpected = planned.changes.changed() - set(only)
        if unexpected:
            raise InfraError(
                f"refused: {spec.name} has other pending changes "
                f"({', '.join(sorted(unexpected))}); "
                f"review them with `ff-ops infra plan {spec.stack}` first"
            )
    violations = stack_ops.guard_violations(planned.changes, spec.protected, allow_replace)
    if violations:
        raise InfraError(describe_violations(spec, violations))
    verb = "create" if planned.existing is None else "update"
    if not assume_yes and not session.confirm(f"{verb} {spec.name}?"):
        raise InfraError("cancelled")
    command = (stack_ops.create_command if planned.existing is None else stack_ops.update_command)(
        session.cli, spec.name, planned.template_path
    )
    session.echo(f"{verb} {spec.name} (this waits until Heat is done)…")
    try:
        session.runner.run(command)
    except CommandError as error:
        raise InfraError(_failure_report(session, spec.name, error)) from error
    final = get_stack(session, spec.name)
    if final is not None and not final.healthy:
        raise InfraError(f"{spec.name} is {final.status}: {final.reason}")
    return final


def _failure_report(session: InfraSession, name: str, error: CommandError) -> str:
    """Explain a failed create/update with the stack status Heat recorded, if any.

    Heat often answers a bare "Internal Error" to the CLI while the stack itself keeps the
    real reason (``stack_status_reason``).
    """
    try:
        info = get_stack(session, name)
    except CommandError:
        info = None
    if info is None:
        return (
            f"{error}\n  {name} was not created: Heat refused the request before creating "
            "anything. Run the command again with `openstack --debug` to see Heat's answer."
        )
    return f"{error}\n  {name} is {info.status}: {info.reason or '(no reason given)'}"


def destroy(session: InfraSession, spec: StackSpec, *, assume_yes: bool = False) -> None:
    """Delete a stack after the operator retypes its name.

    Resources with ``deletion_policy: Retain`` (data volumes) survive the deletion.

    Raises
    ------
    InfraError
        The confirmation does not match.
    """
    if not assume_yes and not session.confirm(f"delete {spec.name} and every resource it owns?"):
        raise InfraError("cancelled")
    session.runner.run(stack_ops.delete_command(session.cli, spec.name))
