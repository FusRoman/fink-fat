"""Command-line interface of ff-ops (wiring only: no logic lives here)."""

from __future__ import annotations

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from ipaddress import IPv4Address, IPv4Network
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from finkfat_ops import access as access_ops
from finkfat_ops import infra as infra_ops
from finkfat_ops import naming
from finkfat_ops import ssh as ssh_ops
from finkfat_ops import vm as vm_ops
from finkfat_ops import volume as volume_ops
from finkfat_ops.checks import Finding, Level, check_sensitive_files
from finkfat_ops.config import ConfigError, Deployment, DeployPaths, load_deployment
from finkfat_ops.heat.net import ADMIN_SSH_RESOURCE
from finkfat_ops.heat.storage import data_volumes
from finkfat_ops.naming import Role, Stack
from finkfat_ops.openstack import (
    OpenStackCli,
    ProjectMismatchError,
    default_openstack_binary,
    parse_json,
    profile_password,
    verify_project,
)
from finkfat_ops.runner import (
    CommandError,
    CommandResult,
    DryRunRunner,
    Runner,
    SubprocessRunner,
)

DEFAULT_DEPLOY_DIR = Path(__file__).resolve().parents[1]
"""The ``deploy/`` directory this package lives in."""

console = Console(soft_wrap=True)
error_console = Console(soft_wrap=True, stderr=True)
"""Remote commands' error output (``ff-ops ssh … -- <cmd>``)."""

app = typer.Typer(
    name="ff-ops",
    help="Manage the fink-fat production instance on OpenStack. See deploy/README.md.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(
    help="Validate and inspect the deployment configuration.", no_args_is_help=True
)
app.add_typer(config_app, name="config")
infra_app = typer.Typer(
    help="Heat stacks: render, preview, apply, inspect and destroy.", no_args_is_help=True
)
app.add_typer(infra_app, name="infra")
access_app = typer.Typer(
    help="Source ranges allowed to SSH into the bastion.", no_args_is_help=True
)
app.add_typer(access_app, name="access")
volume_app = typer.Typer(help="Persistent data volumes: list and snapshot.", no_args_is_help=True)
app.add_typer(volume_app, name="volume")
vm_app = typer.Typer(help="The VMs: list, console log, reboot, stop, start.", no_args_is_help=True)
app.add_typer(vm_app, name="vm")


@dataclass
class AppContext:
    """State shared by every command.

    Attributes
    ----------
    paths : DeployPaths
        Deployment file locations.
    runner : Runner
        Executes external commands (real, dry-run or fake).
    dry_run : bool
        Commands are printed, not run; nothing local is written either.
    openstack_binary : str
        The ``openstack`` executable.
    confirm : Callable[[str], bool]
        Asks the operator a yes/no question.
    prompt : Callable[[str], str]
        Asks the operator to type a value.
    ask_secret : Callable[[str], str]
        Asks the operator to type a secret, without echoing it.
    """

    paths: DeployPaths
    runner: Runner
    dry_run: bool = False
    openstack_binary: str = field(default_factory=default_openstack_binary)
    confirm: Callable[[str], bool] = lambda question: typer.confirm(question, default=False)
    prompt: Callable[[str], str] = lambda question: typer.prompt(question, default="")
    ask_secret: Callable[[str], str] = lambda question: typer.prompt(question, hide_input=True)


@app.callback()
def main_callback(
    ctx: typer.Context,
    deploy_dir: Annotated[
        Path, typer.Option(help="The deploy/ directory holding infra.yaml and secrets.env.")
    ] = DEFAULT_DEPLOY_DIR,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print external commands instead of running them.")
    ] = False,
) -> None:
    """Build the shared context, unless a test already injected one."""
    if ctx.obj is None:
        runner: Runner = DryRunRunner(echo=_print_plain) if dry_run else SubprocessRunner()
        ctx.obj = AppContext(
            paths=DeployPaths(deploy_dir.resolve()), runner=runner, dry_run=dry_run
        )


def _print_plain(line: str) -> None:
    """Print a line verbatim (no rich markup: ``[dry-run]`` must stay visible)."""
    console.print(line, markup=False, highlight=False)


_LEVEL_STYLE = {Level.OK: "[green]✔[/]", Level.WARN: "[yellow]⚠[/]", Level.ERROR: "[red]✘[/]"}


def _print_findings(findings: list[Finding]) -> None:
    """Print findings, each followed by its fix when there is one."""
    for finding in findings:
        console.print(f"{_LEVEL_STYLE[finding.level]} {finding.message}")
        if finding.fix and finding.level != Level.OK:
            console.print(f"    → {finding.fix}", style="dim")


def _deployment_table(deployment: Deployment) -> Table:
    """Summarise the validated deployment as a table (no secret is shown)."""
    infra, local = deployment.infra, deployment.local
    table = Table(
        title=f"fink-fat {infra.env} — project {local.openstack.project} ({local.openstack.cloud})"
    )
    for column in ("role", "enabled", "flavor", "fixed IP", "data volume", "public"):
        table.add_column(column)
    for role, cfg in infra.roles.items():
        volume = f"{cfg.volume.size_gb} GB → {cfg.volume.mount}" if cfg.volume else "—"
        table.add_row(
            str(role),
            "yes" if cfg.enabled else "no",
            cfg.flavor,
            str(cfg.fixed_ip),
            volume,
            "floating IP" if cfg.floating_ip else "—",
        )
    return table


@config_app.command("check")
def config_check(ctx: typer.Context) -> None:
    """Validate infra.yaml, infra.local.yaml and secrets.env without contacting anything.

    Also checks that the gitignored files are not tracked by git and are private (600).

    Example: ff-ops config check
    """
    app_ctx: AppContext = ctx.obj
    failed = False
    try:
        deployment = load_deployment(app_ctx.paths)
    except ConfigError as error:
        failed = True
        for problem in error.problems:
            console.print(f"[red]✘[/] {problem}")
    else:
        console.print(_deployment_table(deployment))
        local = deployment.local
        console.print(f"SSH allowed from: {', '.join(str(c) for c in local.admin_cidrs)}")
        console.print(f"explorer domain: {local.explorer_domain or '(none — self-signed HTTPS)'}")
        console.print("[green]✔[/] configuration is valid")
    findings = check_sensitive_files(app_ctx.paths, app_ctx.runner)
    _print_findings(findings)
    if failed or any(f.level == Level.ERROR for f in findings):
        raise typer.Exit(code=1)


# ----------------------------------------------------------------------------- helpers


@contextmanager
def _fail_nicely() -> Generator[None]:
    """Turn the expected errors into a red message and exit status 1."""
    try:
        yield
    except ConfigError as error:
        for problem in error.problems:
            console.print(f"[red]✘[/] {problem}")
        raise typer.Exit(code=1) from error
    except (
        infra_ops.InfraError,
        CommandError,
        ProjectMismatchError,
        ValueError,
    ) as error:
        console.print(f"[red]✘[/] {error}")
        raise typer.Exit(code=1) from error


def _session(app_ctx: AppContext, *, verify: bool = True) -> infra_ops.InfraSession:
    """Load the configuration and, unless ``verify`` is false, check the cloud project.

    The password of a password-based profile is asked here, once, only when the cloud is
    really contacted (not for ``verify=False`` nor in dry-run mode).
    """
    deployment = load_deployment(app_ctx.paths)
    target = deployment.local.openstack
    contacts_cloud = verify and not app_ctx.dry_run
    password = profile_password(
        target.ask_password and contacts_cloud, os.environ, app_ctx.ask_secret
    )
    cli = OpenStackCli(cloud=target.cloud, binary=app_ctx.openstack_binary, password=password)
    if verify:
        project_id = verify_project(cli, app_ctx.runner, target.project)
        if project_id:
            label = (
                project_id if target.project == project_id else f"{target.project} ({project_id})"
            )
            console.print(f"[green]✔[/] project {label}")
    return infra_ops.InfraSession(
        deployment=deployment,
        paths=app_ctx.paths,
        cli=cli,
        runner=app_ctx.runner,
        echo=console.print,
        confirm=app_ctx.confirm,
    )


def _access_state(app_ctx: AppContext) -> access_ops.AccessState:
    """Read ``.state/access.yaml``."""
    return access_ops.load_access_state(access_ops.access_state_path(app_ctx.paths.state_dir))


def _spec(
    session: infra_ops.InfraSession, app_ctx: AppContext, stack: Stack
) -> infra_ops.StackSpec:
    """Build the specification of ``stack`` from the configuration alone (no cloud call)."""
    return infra_ops.build_stack_spec(stack, session.deployment, _access_state(app_ctx))


def _resolved_spec(
    session: infra_ops.InfraSession, app_ctx: AppContext, stack: Stack
) -> infra_ops.StackSpec:
    """Build the specification of ``stack``, with the decisions that need the cloud."""
    return infra_ops.resolve_spec(session, stack, _access_state(app_ctx))


StackArgument = Annotated[Stack, typer.Argument(help="Which stack (net, storage, servers).")]
OptionalStackArgument = Annotated[
    Stack | None, typer.Argument(help="Which stack (net, storage, servers).")
]


# ------------------------------------------------------------------------------- infra


@infra_app.command("render")
def infra_render(ctx: typer.Context, stack: OptionalStackArgument = None) -> None:
    """Write the Heat template of a stack to deploy/.build/ (contacts nothing).

    The stack must be named: without it, this help is shown and nothing is written.

    Example: ff-ops infra render net
    """
    if stack is None:
        _print_help(ctx)
    app_ctx: AppContext = ctx.obj
    with _fail_nicely():
        session = _session(app_ctx, verify=False)
        path = infra_ops.render(session, _spec(session, app_ctx, stack))
        console.print(f"[green]✔[/] {path}")
        console.print(f"    check it with: openstack orchestration template validate -t {path}")


@infra_app.command("plan")
def infra_plan(ctx: typer.Context, stack: OptionalStackArgument = None) -> None:
    """Preview what applying a stack would change, without changing anything.

    Without a stack, the three stacks are previewed in order (net, storage, servers);
    a stack that cannot be previewed is reported and the next one is still shown.

    Examples: ff-ops infra plan    ·    ff-ops infra plan servers
    """
    app_ctx: AppContext = ctx.obj
    failed = False
    with _fail_nicely():
        session = _session(app_ctx)
        for current in [stack] if stack is not None else list(Stack):
            try:
                _print_plan(session, app_ctx, current)
            except (infra_ops.InfraError, CommandError) as error:
                name = naming.stack_name(session.deployment.infra.env, current)
                console.print(f"{name}: [red]✘[/] {error}")
                failed = True
    if failed:
        raise typer.Exit(code=1)


def _print_help(ctx: typer.Context) -> NoReturn:
    """Print the help of the current command and stop (used when a stack is not named)."""
    console.print(ctx.get_help(), markup=False, highlight=False)
    raise typer.Exit(code=0)


def _print_plan(session: infra_ops.InfraSession, app_ctx: AppContext, stack: Stack) -> None:
    """Preview one stack and print its changes, adoptions and guard violations."""
    planned = infra_ops.plan(session, _resolved_spec(session, app_ctx, stack))
    state = "will be created" if planned.existing is None else planned.existing.status
    console.print(f"{planned.spec.name}: {state}")
    lines = [
        *infra_ops.describe_adoptions(planned.spec),
        *infra_ops.describe_changes(planned.changes, creating=planned.existing is None),
        *infra_ops.describe_ignored(planned),
    ]
    for line in lines:
        console.print(f"  {line}", markup=False)
    violations = infra_ops.stack_ops.guard_violations(planned.changes, planned.spec.protected)
    if violations:
        console.print(f"[red]✘[/] {infra_ops.describe_violations(planned.spec, violations)}")


@infra_app.command("apply")
def infra_apply(
    ctx: typer.Context,
    stack: OptionalStackArgument = None,
    allow_replace: Annotated[
        list[str] | None,
        typer.Option(help="Protected resource that may be replaced or deleted (repeatable)."),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation.")] = False,
) -> None:
    """Create or update a stack: preview, replacement guard, confirmation, then apply.

    The stack must be named: without it, this help is shown and nothing is applied.

    Example: ff-ops infra apply net
    """
    if stack is None:
        _print_help(ctx)
    app_ctx: AppContext = ctx.obj
    with _fail_nicely():
        session = _session(app_ctx)
        spec = _resolved_spec(session, app_ctx, stack)
        for name in allow_replace or []:
            if app_ctx.prompt(f"type '{name}' to confirm it may be destroyed") != name:
                raise infra_ops.InfraError("cancelled")
        before = infra_ops.get_stack(session, spec.name) if stack == Stack.SERVERS else None
        final = infra_ops.apply(session, spec, allow_replace=allow_replace or (), assume_yes=yes)
        if final is not None:
            console.print(f"[green]✔[/] {final.name}: {final.status}")
        if stack == Stack.SERVERS and final is not None:
            _report_ssh_access(infra_ops.write_ssh_access(session, before, final))


def _report_ssh_access(config: Path | None) -> None:
    """Tell where the SSH configuration was written and what to do next."""
    if config is None:
        return
    console.print(f"[green]✔[/] {config}")
    console.print(
        "    next: ff-ops init (waits for cloud-init, then checks each machine)", style="dim"
    )


@infra_app.command("show")
def infra_show(ctx: typer.Context) -> None:
    """Show the status and outputs of every stack.

    Example: ff-ops infra show
    """
    app_ctx: AppContext = ctx.obj
    with _fail_nicely():
        session = _session(app_ctx)
        table = Table(title="Heat stacks")
        for column in ("stack", "status", "reason / outputs"):
            table.add_column(column)
        env = session.deployment.infra.env
        for stack in Stack:
            name = naming.stack_name(env, stack)
            info = infra_ops.get_stack(session, name)
            if info is None:
                table.add_row(name, "absent", "")
                continue
            detail = info.reason if not info.healthy else ", ".join(sorted(info.outputs))
            style = "green" if info.healthy else "red"
            table.add_row(name, f"[{style}]{info.status}[/]", detail)
        console.print(table)


@infra_app.command("destroy")
def infra_destroy(ctx: typer.Context, stack: StackArgument) -> None:
    """Delete a stack and every resource it owns (retyping its name is required).

    Data volumes (storage stack, deletion policy Retain) survive the deletion.

    Example: ff-ops infra destroy servers
    """
    app_ctx: AppContext = ctx.obj
    with _fail_nicely():
        session = _session(app_ctx)
        spec = _spec(session, app_ctx, stack)
        if app_ctx.prompt(f"type '{spec.name}' to delete it") != spec.name:
            raise infra_ops.InfraError("cancelled")
        infra_ops.destroy(session, spec, assume_yes=True)
        console.print(f"[green]✔[/] {spec.name} deleted")
        if stack == Stack.STORAGE:
            console.print(
                "    the data volumes are kept (Retain): `ff-ops infra apply storage` adopts "
                "them again, `ff-ops volume list` shows them",
                style="dim",
            )


# ------------------------------------------------------------------------------ access


def _detect_public_ip(app_ctx: AppContext) -> IPv4Address:
    """Ask the public echo service for this machine's IPv4 address."""
    result = app_ctx.runner.run(access_ops.public_ip_command())
    return access_ops.parse_public_ip(result.stdout)


@access_app.command("list")
def access_list(ctx: typer.Context) -> None:
    """List the ranges allowed to SSH into the bastion and check the current IP.

    Example: ff-ops access list
    """
    app_ctx: AppContext = ctx.obj
    with _fail_nicely():
        deployment = load_deployment(app_ctx.paths)
        state = _access_state(app_ctx)
        table = Table(title="SSH into the bastion is allowed from")
        table.add_column("range")
        table.add_column("managed by")
        for cidr in deployment.local.admin_cidrs:
            table.add_row(str(cidr), "infra.local.yaml (stable)")
        for cidr in state.dynamic_cidrs:
            table.add_row(str(cidr), "ff-ops access (dynamic)")
        console.print(table)
        if app_ctx.dry_run:
            return
        ip = _detect_public_ip(app_ctx)
        cidrs = access_ops.effective_ssh_cidrs(deployment.local, state)
        if access_ops.covers(cidrs, ip):
            console.print(f"[green]✔[/] your current IP {ip} is allowed")
        else:
            console.print(f"[yellow]⚠[/] your current IP {ip} is not allowed")
            console.print("    → ff-ops access allow-me", style="dim")


def _apply_access(app_ctx: AppContext, new_state: access_ops.AccessState, yes: bool) -> None:
    """Apply the SSH rules of ``new_state`` to the net stack, then record the state."""
    session = _session(app_ctx)
    spec = infra_ops.build_stack_spec(Stack.NET, session.deployment, new_state)
    infra_ops.apply(session, spec, only={ADMIN_SSH_RESOURCE}, assume_yes=yes)
    if app_ctx.dry_run:
        _print_plain("[dry-run] .state/access.yaml left unchanged")
        return
    access_ops.save_access_state(access_ops.access_state_path(app_ctx.paths.state_dir), new_state)
    console.print("[green]✔[/] SSH rules updated")


@access_app.command("allow-me")
def access_allow_me(
    ctx: typer.Context,
    temporary: Annotated[
        bool, typer.Option(help="Add this IP without removing the previous dynamic ones.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation.")] = False,
) -> None:
    """Allow this machine's current public IP to SSH into the bastion.

    By default the new IP replaces the previous dynamic one (stable ranges of
    infra.local.yaml are never touched).

    Example: ff-ops access allow-me --temporary
    """
    app_ctx: AppContext = ctx.obj
    with _fail_nicely():
        ip = _detect_public_ip(app_ctx)
        console.print(f"current public IP: {ip}")
        new_state = access_ops.allow(_access_state(app_ctx), ip, temporary=temporary)
        _apply_access(app_ctx, new_state, yes)


@access_app.command("revoke")
def access_revoke(
    ctx: typer.Context,
    cidr: Annotated[str, typer.Argument(help="Dynamic range to remove, e.g. 198.51.100.7/32.")],
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation.")] = False,
) -> None:
    """Remove a dynamic range from the SSH rules of the bastion.

    Example: ff-ops access revoke 198.51.100.7/32
    """
    app_ctx: AppContext = ctx.obj
    with _fail_nicely():
        deployment = load_deployment(app_ctx.paths)
        new_state = access_ops.revoke(_access_state(app_ctx), deployment.local, IPv4Network(cidr))
        _apply_access(app_ctx, new_state, yes)


# ------------------------------------------------------------------------------ volume


@volume_app.command("list")
def volume_list(ctx: typer.Context) -> None:
    """List the data volumes: size, status, attachment and how the storage stack manages them.

    Example: ff-ops volume list
    """
    app_ctx: AppContext = ctx.obj
    with _fail_nicely():
        session = _session(app_ctx)
        infra = session.deployment.infra
        existing = infra_ops.get_stack(session, naming.stack_name(infra.env, Stack.STORAGE))
        parameters = existing.parameters if existing else None
        outputs = existing.outputs if existing else None
        table = Table(title="Data volumes")
        for column in ("role", "volume", "size", "status", "attached to", "managed", "ID"):
            table.add_column(column)
        labels = []
        for volume in data_volumes(infra):
            found = infra_ops.find_volumes(session, volume.name)
            if not found:
                table.add_row(str(volume.role), volume.name, f"({volume.size_gb} GB)", "absent")
            for info in found:
                label = volume_ops.management(volume, info.id, parameters, outputs)
                labels.append(label)
                table.add_row(
                    str(volume.role),
                    volume.name,
                    f"{info.size_gb} GB",
                    info.status,
                    info.attached_to or "—",
                    label,
                    info.id,
                )
        console.print(table)
        if volume_ops.ORPHAN in labels:
            console.print(
                f"[yellow]⚠[/] '{volume_ops.ORPHAN}' volumes are managed by no stack: "
                "`ff-ops infra apply storage` adopts them"
            )


@volume_app.command("snapshot")
def volume_snapshot(
    ctx: typer.Context,
    role: Annotated[Role, typer.Argument(help="Machine whose data volume is snapshotted.")],
) -> None:
    """Take a Cinder snapshot of a role's data volume, even while it is attached.

    The snapshot is crash-consistent (like a power cut): Postgres recovers from it. Take
    one before any risky operation.

    Example: ff-ops volume snapshot db
    """
    app_ctx: AppContext = ctx.obj
    with _fail_nicely():
        session = _session(app_ctx)
        volume = infra_ops.volume_of_role(session.deployment, role)
        found = infra_ops.find_volumes(session, volume.name)
        if app_ctx.dry_run and not found:
            found = [volume_ops.VolumeInfo(f"<{volume.name} id>", volume.name, "", 0)]
        if len(found) != 1:
            raise infra_ops.InfraError(
                f"expected exactly one volume named {volume.name}, found {len(found)}"
            )
        name = volume_ops.snapshot_name(volume, datetime.now(UTC))
        result = app_ctx.runner.run(volume_ops.snapshot_command(session.cli, found[0].id, name))
        snapshot = parse_json(result) or {}
        console.print(
            f"[green]✔[/] snapshot {name} requested ({snapshot.get('id', 'dry-run')}, "
            f"status {snapshot.get('status', '—')})"
        )
        check = volume_ops.snapshot_show_command(session.cli, name).display()
        console.print(f"    check it reaches 'available': {check}", style="dim", markup=False)


# ---------------------------------------------------------------------- ssh and init


def _ssh_config(app_ctx: AppContext, *, refresh: bool) -> Path:
    """Return the generated ``ssh_config``, writing it from the servers stack if needed."""
    config = ssh_ops.ssh_config_path(app_ctx.paths.state_dir)
    if config.exists() and not refresh:
        return config
    session = _session(app_ctx)
    name = naming.stack_name(session.deployment.infra.env, Stack.SERVERS)
    info = infra_ops.get_stack(session, name)
    if info is None or not info.healthy:
        raise infra_ops.InfraError(f"{name} is not deployed: run ff-ops infra apply servers")
    written = infra_ops.write_ssh_access(session, info, info)
    if written is None:
        raise infra_ops.InfraError(f"{name} exposes no public address yet")
    return written


@app.command("ssh", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def ssh(
    ctx: typer.Context,
    role: Annotated[Role, typer.Argument(help="Machine to connect to.")],
    refresh: Annotated[
        bool, typer.Option(help="Rewrite .state/ssh_config from the servers stack first.")
    ] = False,
) -> None:
    """Open a shell on a machine (through the bastion), or run a command after `--`.

    Examples: ff-ops ssh db    ·    ff-ops ssh compute -- docker ps
    """
    app_ctx: AppContext = ctx.obj
    with _fail_nicely():
        config = _ssh_config(app_ctx, refresh=refresh)
        result = app_ctx.runner.run(ssh_ops.ssh_command(config, role, ctx.args), check=False)
        if result.stdout:
            console.print(result.stdout, end="", markup=False, highlight=False)
        if result.stderr:
            error_console.print(result.stderr, end="", markup=False, highlight=False)
        if result.returncode:
            raise typer.Exit(code=result.returncode)


@app.command("init")
def init(
    ctx: typer.Context,
    roles: Annotated[
        list[Role] | None, typer.Argument(help="Machines to check (default: every enabled one).")
    ] = None,
) -> None:
    """Wait for the first boot (cloud-init) of the machines, then check them.

    Checks Docker, the data volume mount and the clock synchronisation. The bastion
    (explorer) is checked first, since the other machines are reached through it.

    Example: ff-ops init db
    """
    app_ctx: AppContext = ctx.obj
    failed = False
    with _fail_nicely():
        infra = load_deployment(app_ctx.paths).infra
        config = _ssh_config(app_ctx, refresh=False)
        wanted = roles or [role for role, cfg in infra.roles.items() if cfg.enabled]
        ordered = sorted(set(wanted), key=lambda role: role != Role.EXPLORER)
        for role in ordered:
            console.print(f"[bold]{role}[/]")
            for check in ssh_ops.init_checks(infra, role):
                command = ssh_ops.ssh_command(
                    config, role, check.command, timeout_s=check.timeout_s
                )
                try:
                    result = app_ctx.runner.run(command, check=False)
                except CommandError as error:
                    result = CommandResult(returncode=1, stderr=str(error).splitlines()[0])
                finding = ssh_ops.evaluate_check(check, result)
                _print_findings([finding])
                failed = failed or finding.level == Level.ERROR
                if finding.level == Level.ERROR:
                    break
    if failed:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------------- vm


@vm_app.command("list")
def vm_list(ctx: typer.Context) -> None:
    """List the VMs of this environment with their status and addresses.

    Example: ff-ops vm list
    """
    app_ctx: AppContext = ctx.obj
    with _fail_nicely():
        session = _session(app_ctx)
        result = app_ctx.runner.run(vm_ops.list_command(session.cli, session.deployment.infra.env))
        table = Table(title="VMs")
        for column in ("name", "status", "addresses", "ID"):
            table.add_column(column)
        for server in vm_ops.parse_server_list(parse_json(result)):
            style = "green" if server.status == "ACTIVE" else "yellow"
            table.add_row(server.name, f"[{style}]{server.status}[/]", server.networks, server.id)
        console.print(table)


@vm_app.command("console")
def vm_console(
    ctx: typer.Context,
    role: Annotated[Role, typer.Argument(help="Machine whose console log is shown.")],
    lines: Annotated[int, typer.Option(help="Number of lines.")] = vm_ops.CONSOLE_LINES,
) -> None:
    """Show the end of a VM's console log (boot and cloud-init messages, even without SSH).

    Example: ff-ops vm console db --lines 200
    """
    app_ctx: AppContext = ctx.obj
    with _fail_nicely():
        session = _session(app_ctx)
        command = vm_ops.console_command(session.cli, session.deployment.infra.env, role, lines)
        console.print(app_ctx.runner.run(command).stdout, markup=False, highlight=False)


def _power(ctx: typer.Context, role: Role, action: vm_ops.Power, yes: bool) -> None:
    """Reboot, stop or start a VM, after confirmation for reboot and stop."""
    app_ctx: AppContext = ctx.obj
    with _fail_nicely():
        session = _session(app_ctx)
        name = naming.server_name(session.deployment.infra.env, role)
        if action != vm_ops.Power.START and not yes and not app_ctx.confirm(f"{action} {name}?"):
            raise infra_ops.InfraError("cancelled")
        command = vm_ops.power_command(session.cli, session.deployment.infra.env, role, action)
        app_ctx.runner.run(command)
        console.print(f"[green]✔[/] {action} {name} requested (ff-ops vm list shows its status)")


YesOption = Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation.")]
RoleArgument = Annotated[Role, typer.Argument(help="Which machine.")]


@vm_app.command("reboot")
def vm_reboot(ctx: typer.Context, role: RoleArgument, yes: YesOption = False) -> None:
    """Reboot a VM (soft reboot).

    Example: ff-ops vm reboot explorer
    """
    _power(ctx, role, vm_ops.Power.REBOOT, yes)


@vm_app.command("stop")
def vm_stop(ctx: typer.Context, role: RoleArgument, yes: YesOption = False) -> None:
    """Shut a VM down (its disks are kept).

    Example: ff-ops vm stop compute
    """
    _power(ctx, role, vm_ops.Power.STOP, yes)


@vm_app.command("start")
def vm_start(ctx: typer.Context, role: RoleArgument) -> None:
    """Start a stopped VM.

    Example: ff-ops vm start compute
    """
    _power(ctx, role, vm_ops.Power.START, yes=True)


def main() -> None:
    """Entry point of the ``ff-ops`` script."""
    app()
