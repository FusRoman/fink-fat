"""Diagnosis of the operator's workstation (``ff-ops doctor``).

Every check gives a :class:`~finkfat_ops.checks.Finding` carrying what to do when it fails:
the tools (``openstack`` and its Heat plugin, ``ssh``, ``gh``), the configuration files and
their permissions, the cloud profile and project, the reachability of every OpenStack
endpoint used by ff-ops, the SSH key, and whether the current public address may reach the
bastion.

Building the commands and evaluating their results are pure; :func:`local_checks` and
:func:`cloud_checks` run them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv4Network
from pathlib import Path
from typing import Any

from finkfat_ops import access
from finkfat_ops.checks import Finding, Level, evaluate_file_mode
from finkfat_ops.openstack import OpenStackCli, parse_json
from finkfat_ops.runner import Command, CommandError, CommandResult, Runner

ENDPOINT_TYPES = {
    "identity": "Keystone",
    "compute": "Nova",
    "network": "Neutron",
    "volumev3": "Cinder",
    "block-storage": "Cinder",
    "orchestration": "Heat",
}
"""Service types of the catalog ff-ops talks to, with their usual names."""

REACH_TIMEOUT_S = 10
"""Longest wait for an endpoint to answer."""

USER_CLOUDS_YAML = Path("~/.config/openstack/clouds.yaml")
"""Usual location of the cloud profiles."""


# ------------------------------------------------------------------------------ tools


def heat_plugin_command(binary: str) -> Command:
    """``openstack stack list --help``: fails when the Heat plugin is missing (pure)."""
    return Command.of(binary, "stack", "list", "--help", timeout_s=60)


def gh_auth_command() -> Command:
    """``gh auth status`` (pure)."""
    return Command.of("gh", "auth", "status", timeout_s=30)


def ssh_version_command() -> Command:
    """``ssh -V`` (pure)."""
    return Command.of("ssh", "-V", timeout_s=10)


def evaluate_tool(label: str, result: CommandResult, fix: str) -> Finding:
    """Turn the result of a tool probe into a finding (pure)."""
    if result.ok:
        return Finding(Level.OK, label)
    detail = (result.stderr or result.stdout).strip().splitlines()
    reason = detail[-1] if detail else f"exit status {result.returncode}"
    return Finding(Level.ERROR, f"{label}: {reason}", fix)


def evaluate_identity_file(path: Path | None, mode: int | None) -> Finding:
    """Check the private key named by ``ssh_identity_file`` (pure).

    Parameters
    ----------
    path : Path | None
        The key, expanded; ``None`` when ``ssh`` chooses it.
    mode : int | None
        Its ``st_mode``; ``None`` when the file does not exist.
    """
    if path is None:
        return Finding(Level.OK, "SSH key: ssh's defaults (no ssh_identity_file)")
    if mode is None:
        return Finding(
            Level.ERROR,
            f"SSH key {path} not found",
            "fix ssh_identity_file in infra.local.yaml (private key of the Nova keypair)",
        )
    if mode & 0o077:
        return Finding(Level.ERROR, f"SSH key {path} is readable by others", f"chmod 600 {path}")
    return Finding(Level.OK, f"SSH key {path}")


def evaluate_public_ip(ip: IPv4Address, cidrs: Sequence[IPv4Network]) -> Finding:
    """Check that the current public address may SSH into the bastion (pure)."""
    if access.covers(cidrs, ip):
        return Finding(Level.OK, f"public IP {ip} is allowed to SSH into the bastion")
    return Finding(
        Level.WARN,
        f"public IP {ip} is not in the allowed SSH ranges",
        "ff-ops access allow-me (or go through sshuttle, which exits from the lab range)",
    )


# ------------------------------------------------------------------------------ cloud


@dataclass(frozen=True)
class Endpoint:
    """A public endpoint of the cloud catalog.

    Attributes
    ----------
    service : str
        Usual name, e.g. ``Heat``.
    url : str
        Public URL.
    """

    service: str
    url: str


def parse_catalog(data: Any) -> list[Endpoint]:
    """Read the public endpoints ff-ops uses from ``openstack catalog list -f json`` (pure).

    Depending on the client version, the endpoints of a service are a list of objects or
    one text block (a region line, then lines such as ``public: <url>``); both are
    accepted.
    """
    endpoints: dict[str, Endpoint] = {}
    for entry in data or []:
        name = ENDPOINT_TYPES.get(str(entry.get("Type", "")))
        if name is None or name in endpoints:
            continue
        for url in _public_urls(entry.get("Endpoints")):
            endpoints[name] = Endpoint(name, url)
            break
    return sorted(endpoints.values(), key=lambda endpoint: endpoint.service)


def _public_urls(raw: Any) -> Iterable[str]:
    """Extract the public URLs of a catalog entry's ``Endpoints``."""
    if isinstance(raw, str):
        yield from re.findall(r"public:\s*(\S+)", raw)
        return
    for item in raw or []:
        if isinstance(item, dict) and item.get("interface") == "public" and item.get("url"):
            yield str(item["url"])


def reach_command(url: str) -> Command:
    """Contact an endpoint; any HTTP answer, even an error, proves it is reachable (pure)."""
    return Command.of(
        "curl",
        "--silent",
        "--output",
        "/dev/null",
        "--connect-timeout",
        "5",
        "--max-time",
        str(REACH_TIMEOUT_S),
        url,
        timeout_s=REACH_TIMEOUT_S + 5,
    )


def evaluate_reach(endpoint: Endpoint, result: CommandResult) -> Finding:
    """Turn the result of :func:`reach_command` into a finding (pure)."""
    if result.ok:
        return Finding(Level.OK, f"{endpoint.service} reachable ({endpoint.url})")
    fix = (
        "start sshuttle: this endpoint is only reachable from the lab network"
        if endpoint.service == "Heat"
        else "check your network (sshuttle?) and the cloud's status"
    )
    return Finding(Level.ERROR, f"{endpoint.service} unreachable ({endpoint.url})", fix)


# -------------------------------------------------------------------------------- I/O


def _probe(runner: Runner, command: Command) -> CommandResult:
    """Run a probe; a missing program or a timeout becomes a failed result."""
    try:
        return runner.run(command, check=False)
    except CommandError as error:
        return CommandResult(returncode=127, stderr=str(error).splitlines()[0])


def local_checks(runner: Runner, openstack_binary: str, identity_file: str | None) -> list[Finding]:
    """Check the tools, the SSH key and ``clouds.yaml``'s mode (I/O; contacts nothing remote).

    Parameters
    ----------
    runner : Runner
        Runs the probes.
    openstack_binary : str
        The ``openstack`` executable.
    identity_file : str | None
        ``ssh_identity_file`` of ``infra.local.yaml``.
    """
    findings = [
        evaluate_tool(
            "openstack CLI with the Heat plugin",
            _probe(runner, heat_plugin_command(openstack_binary)),
            "cd deploy && pdm install, then run ff-ops through `pdm run`",
        ),
        evaluate_tool("ssh", _probe(runner, ssh_version_command()), "install OpenSSH"),
        evaluate_tool(
            "gh logged in (ff-ops images list)",
            _probe(runner, gh_auth_command()),
            "gh auth login (scope read:packages)",
        ),
    ]
    key = Path(identity_file).expanduser() if identity_file else None
    mode = key.stat().st_mode if key is not None and key.is_file() else None
    findings.append(evaluate_identity_file(key, mode))
    clouds = USER_CLOUDS_YAML.expanduser()
    if clouds.is_file():
        findings.append(evaluate_file_mode(clouds, clouds.stat().st_mode))
    return findings


def cloud_checks(runner: Runner, cli: OpenStackCli) -> list[Finding]:
    """Check that every OpenStack endpoint ff-ops uses answers (I/O).

    The project itself is checked beforehand by the caller
    (:func:`finkfat_ops.openstack.verify_project`).
    """
    listing = _probe(runner, cli.command("catalog", "list"))
    if not listing.ok:
        return [evaluate_tool("service catalog", listing, "check your clouds.yaml profile")]
    endpoints = parse_catalog(parse_json(listing))
    if not endpoints:
        return [Finding(Level.WARN, "the service catalog lists none of the expected endpoints")]
    return [evaluate_reach(e, _probe(runner, reach_command(e.url))) for e in endpoints]


def public_ip_check(runner: Runner, cidrs: Sequence[IPv4Network]) -> Finding:
    """Check the current public address against the allowed SSH ranges (I/O)."""
    result = _probe(runner, access.public_ip_command())
    try:
        ip = access.parse_public_ip(result.stdout) if result.ok else None
    except ValueError:
        ip = None
    if ip is None:
        return Finding(Level.WARN, "could not detect the public IP", "check the Internet access")
    return evaluate_public_ip(ip, cidrs)
