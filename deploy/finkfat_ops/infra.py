"""Orchestration of the Heat stacks: render, plan, apply, show, destroy.

The decisions (which template, what is protected, whether a change is safe) are made
by pure functions of :mod:`finkfat_ops.heat` and :mod:`finkfat_ops.stack`; this module
sequences the commands and the operator's confirmations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any

from finkfat_ops import naming
from finkfat_ops import ssh as ssh_ops
from finkfat_ops import stack as stack_ops
from finkfat_ops import volume as volume_ops
from finkfat_ops.access import AccessState, effective_ssh_cidrs
from finkfat_ops.config import Deployment, DeployPaths
from finkfat_ops.heat.net import PROTECTED_RESOURCES as NET_PROTECTED
from finkfat_ops.heat.net import build_net_template
from finkfat_ops.heat.render import render_template
from finkfat_ops.heat.servers import (
    EXPLORER_PUBLIC_IP_OUTPUT,
    MissingUpstreamError,
    build_servers_template,
    upstream_parameters,
)
from finkfat_ops.heat.servers import protected_resources as servers_protected
from finkfat_ops.heat.storage import (
    Adoption,
    AdoptionError,
    DataVolume,
    adoption_parameters,
    build_storage_template,
    data_volumes,
    resolve_adoptions,
)
from finkfat_ops.heat.storage import protected_resources as storage_protected
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
    parameters : Mapping[str, str]
        Template parameters passed to every create, preview and update.
    adoptions : tuple[Adoption, ...]
        Existing volumes the stack takes over instead of creating them (storage only).
    """

    stack: Stack
    name: str
    template: dict[str, Any]
    protected: frozenset[str]
    parameters: Mapping[str, str] = field(default_factory=dict)
    adoptions: tuple[Adoption, ...] = ()


IMPLEMENTED_STACKS = (Stack.NET, Stack.STORAGE, Stack.SERVERS)
"""Stacks that ff-ops can build so far."""


def build_stack_spec(
    stack: Stack,
    deployment: Deployment,
    access: AccessState,
    adoptions: Iterable[Adoption] = (),
    upstream: Mapping[str, str] | None = None,
) -> StackSpec:
    """Build the template, parameters and protection list of a stack (pure).

    Parameters
    ----------
    stack : Stack
        Which stack.
    deployment : Deployment
        Validated configuration.
    access : AccessState
        Dynamic SSH ranges (net stack).
    adoptions : Iterable[Adoption]
        Existing volumes to adopt (storage stack), from :func:`resolve_spec`.
    upstream : Mapping[str, str] | None
        Parameters taken from the net and storage outputs (servers stack), from
        :func:`resolve_spec`; ``None`` to build the template alone (``infra render``).

    Returns
    -------
    StackSpec
        The stack specification.

    Raises
    ------
    InfraError
        The stack is not implemented yet.
    """
    infra = deployment.infra
    name = naming.stack_name(infra.env, stack)
    if stack == Stack.NET:
        cidrs = effective_ssh_cidrs(deployment.local, access)
        template = build_net_template(infra, cidrs, deployment.local.dns_nameservers)
        return StackSpec(stack, name, template, NET_PROTECTED)
    if stack == Stack.STORAGE:
        adopted = tuple(adoptions)
        return StackSpec(
            stack,
            name,
            build_storage_template(infra),
            storage_protected(infra),
            parameters=adoption_parameters(data_volumes(infra), adopted),
            adoptions=adopted,
        )
    if stack == Stack.SERVERS:
        return StackSpec(
            stack,
            name,
            build_servers_template(infra),
            servers_protected(infra),
            parameters=dict(upstream or {}),
        )
    raise InfraError(f"stack '{stack}' is not implemented yet")


def resolve_spec(session: InfraSession, stack: Stack, access: AccessState) -> StackSpec:
    """Build a stack specification, with the decisions that need the cloud's state.

    For the storage stack, the existing volumes are looked up by name and compared with
    the stack's current parameters and outputs to decide which ones to adopt (see
    :func:`finkfat_ops.heat.storage.resolve_adoptions`). For the servers stack, the
    outputs of the net and storage stacks become its parameters. The net stack needs no
    lookup.

    Parameters
    ----------
    session : InfraSession
        Cloud access.
    stack : Stack
        Which stack.
    access : AccessState
        Dynamic SSH ranges (net stack).

    Returns
    -------
    StackSpec
        The stack specification.

    Raises
    ------
    InfraError
        The volumes found do not allow a safe adoption decision, or the net or storage
        stack is missing or unhealthy (servers stack).
    """
    if stack == Stack.SERVERS:
        return build_stack_spec(
            stack, session.deployment, access, upstream=_upstream_parameters(session)
        )
    if stack != Stack.STORAGE:
        return build_stack_spec(stack, session.deployment, access)
    volumes = data_volumes(session.deployment.infra)
    existing = get_stack(session, naming.stack_name(session.deployment.infra.env, stack))
    found = {v.name: [info.id for info in find_volumes(session, v.name)] for v in volumes}
    try:
        adoptions = resolve_adoptions(
            volumes,
            found,
            existing.parameters if existing else None,
            existing.outputs if existing else None,
        )
    except AdoptionError as error:
        raise InfraError(f"refused: {error}") from error
    return build_stack_spec(stack, session.deployment, access, adoptions)


def _upstream_parameters(session: InfraSession) -> dict[str, str]:
    """Read the outputs of the net and storage stacks as parameters of the servers stack."""
    env = session.deployment.infra.env
    outputs = {}
    for upstream in (Stack.NET, Stack.STORAGE):
        name = naming.stack_name(env, upstream)
        info = get_stack(session, name)
        if info is None:
            raise InfraError(
                f"refused: {name} does not exist; create it first (ff-ops infra apply {upstream})"
            )
        if not info.healthy:
            raise InfraError(
                f"refused: {info.name} is {info.status}; fix it first (ff-ops infra show)"
            )
        outputs[upstream] = info.outputs
    try:
        return upstream_parameters(
            session.deployment.infra, outputs[Stack.NET], outputs[Stack.STORAGE]
        )
    except MissingUpstreamError as error:
        raise InfraError(f"refused: {error}") from error


def write_ssh_access(
    session: InfraSession, before: StackInfo | None, after: StackInfo
) -> Path | None:
    """Write ``.state/ssh_config`` from the servers stack and forget obsolete host keys.

    Parameters
    ----------
    session : InfraSession
        Cloud access and configuration.
    before : StackInfo | None
        Servers stack before the apply (``None`` if it did not exist).
    after : StackInfo
        Servers stack after the apply.

    Returns
    -------
    Path | None
        The written ``ssh_config``, or ``None`` when the stack exposes no public address
        (dry-run).
    """
    address = after.outputs.get(EXPLORER_PUBLIC_IP_OUTPUT)
    if not address:
        return None
    state_dir = session.paths.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    known_hosts = ssh_ops.known_hosts_path(state_dir)
    config = ssh_ops.ssh_config_path(state_dir)
    identity = session.deployment.local.ssh_identity_file
    infra = session.deployment.infra
    config.write_text(
        ssh_ops.build_ssh_config(infra, IPv4Address(str(address)), known_hosts, identity),
        encoding="utf-8",
    )
    if known_hosts.exists():
        before_outputs = before.outputs if before is not None else None
        for role in ssh_ops.recreated_roles(infra, before_outputs, after.outputs):
            session.runner.run(ssh_ops.forget_host_command(known_hosts, role), check=False)
    return config


def find_volumes(session: InfraSession, name: str) -> list[volume_ops.VolumeInfo]:
    """Return every Cinder volume carrying exactly ``name`` (empty in dry-run)."""
    result = session.runner.run(volume_ops.list_by_name_command(session.cli, name))
    return [info for info in volume_ops.parse_volume_list(parse_json(result)) if info.name == name]


def describe_adoptions(spec: StackSpec) -> list[str]:
    """Format the adoptions of a stack as one line each (pure).

    Parameters
    ----------
    spec : StackSpec
        The stack.

    Returns
    -------
    list[str]
        Lines such as ``"= adopted: ffat-prod-vol-pgdata (<id>)"``; empty without adoption.
    """
    return [f"= adopted: {a.volume.name} ({a.volume_id})" for a in spec.adoptions]


def volume_of_role(deployment: Deployment, role: naming.Role) -> DataVolume:
    """Return the data volume of a role (pure).

    Raises
    ------
    InfraError
        The role has no data volume.
    """
    for volume in data_volumes(deployment.infra):
        if volume.role == role:
            return volume
    raise InfraError(f"the {role} machine has no data volume")


def describe_changes(changes: ChangeSet, *, creating: bool = False) -> list[str]:
    """Format a change set as one line per kind of change (pure).

    Parameters
    ----------
    changes : ChangeSet
        The previewed changes.
    creating : bool
        The stack does not exist yet: an empty change set then means that the stack
        will be created without resources of its own (every volume adopted).

    Returns
    -------
    list[str]
        Lines such as ``"+ added: net, subnet"``; ``["no change"]`` (or
        ``["nothing to create"]`` for a creation) when empty.
    """
    if changes.is_empty:
        return ["nothing to create" if creating else "no change"]
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
    ignored : tuple[str, ...]
        Resources that Heat's preview reported as updated or replaced although neither
        their definition nor their parameters change (references to other resources).
    in_place : tuple[str, ...]
        Resources that Heat's preview reported as replaced although every property that
        changes is updatable in place: counted as updated.
    """

    spec: StackSpec
    template_path: Path
    existing: StackInfo | None
    changes: ChangeSet
    ignored: tuple[str, ...] = ()
    in_place: tuple[str, ...] = ()


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

    Notes
    -----
    For an existing stack, the deployed template is compared with the new one first
    (:func:`finkfat_ops.stack.really_changed`): when nothing changes, Heat is not asked
    at all; otherwise its preview is kept only where the comparison confirms it.
    """
    path = render(session, spec)
    existing = get_stack(session, spec.name)
    if existing is None:
        skipped = [adoption.volume.resource for adoption in spec.adoptions]
        return Plan(spec, path, None, stack_ops.change_set_for_creation(spec.template, skipped))
    deployed = stack_ops.parse_template(
        parse_json(session.runner.run(stack_ops.template_command(session.cli, spec.name)))
    )
    changed = stack_ops.really_changed(
        deployed, spec.template, existing.parameters, dict(spec.parameters)
    )
    if deployed and not changed:
        return Plan(spec, path, existing, stack_ops.ChangeSet())
    result = session.runner.run(
        stack_ops.preview_update_command(session.cli, spec.name, path, spec.parameters)
    )
    previewed = stack_ops.parse_change_set(parse_json(result))
    if not deployed:
        return Plan(spec, path, existing, previewed)
    kept, ignored = stack_ops.filter_change_set(previewed, changed)
    candidates = stack_ops.in_place_candidates(
        kept, deployed, spec.template, existing.parameters, dict(spec.parameters)
    )
    allowed = _update_allowed(session, spec.template, candidates)
    kept, in_place = stack_ops.downgrade_replacements(kept, candidates, allowed)
    return Plan(spec, path, existing, kept, ignored, in_place)


def _update_allowed(
    session: InfraSession, template: dict[str, Any], candidates: Mapping[str, Any]
) -> dict[str, dict[str, bool]]:
    """Read from Heat which properties of each candidate's type update in place."""
    resources = template.get("resources") or {}
    schemas: dict[str, dict[str, bool]] = {}
    allowed = {}
    for name in candidates:
        resource_type = resources[name]["type"]
        if resource_type not in schemas:
            result = session.runner.run(stack_ops.resource_type_command(session.cli, resource_type))
            schemas[resource_type] = stack_ops.parse_update_allowed(parse_json(result))
        allowed[name] = schemas[resource_type]
    return allowed


def describe_ignored(planned: Plan) -> list[str]:
    """Format the preview entries corrected or discarded by :func:`plan` (pure).

    Parameters
    ----------
    planned : Plan
        The plan.

    Returns
    -------
    list[str]
        One line per kind of correction; none when Heat's preview was kept as is.
    """
    lines = []
    if planned.in_place:
        lines.append(
            f"(Heat's preview said {', '.join(planned.in_place)} would be replaced, but every "
            "property that changes is updatable in place: counted as updated)"
        )
    if planned.ignored:
        lines.append(
            f"(Heat's preview also listed {', '.join(planned.ignored)}: they only reference "
            "changed resources, their own definition is unchanged; ignored)"
        )
    return lines


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
    changes = describe_changes(planned.changes, creating=planned.existing is None)
    for line in [*describe_adoptions(spec), *changes, *describe_ignored(planned)]:
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
        session.cli, spec.name, planned.template_path, spec.parameters
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
