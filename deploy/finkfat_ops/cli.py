"""Command-line interface of ff-ops (wiring only: no logic lives here)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from finkfat_ops.checks import Finding, Level, check_sensitive_files
from finkfat_ops.config import ConfigError, Deployment, DeployPaths, load_deployment
from finkfat_ops.runner import DryRunRunner, Runner, SubprocessRunner

DEFAULT_DEPLOY_DIR = Path(__file__).resolve().parents[1]
"""The ``deploy/`` directory this package lives in."""

console = Console()

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


@dataclass
class AppContext:
    """State shared by every command.

    Attributes
    ----------
    paths : DeployPaths
        Deployment file locations.
    runner : Runner
        Executes external commands (real, dry-run or fake).
    """

    paths: DeployPaths
    runner: Runner


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
        runner: Runner = DryRunRunner(echo=console.print) if dry_run else SubprocessRunner()
        ctx.obj = AppContext(paths=DeployPaths(deploy_dir.resolve()), runner=runner)


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


def main() -> None:
    """Entry point of the ``ff-ops`` script."""
    app()
