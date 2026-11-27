"""State of the production instance in one report (``ff-ops status``).

For each enabled role, one SSH command prints labelled sections (deployment record,
containers, disk usage and, on the db, the last nightly sync); the explorer's certificate
is read from this machine with ``curl``; the VM states come from ``openstack server list``.

Building the commands, parsing the outputs and deciding what deserves a warning are pure
(:func:`status_script`, ``parse_*``, :func:`build_report`); :func:`collect_role` is the only
I/O, and an unreachable machine gives a row in error instead of an exception.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finkfat_ops.checks import Finding, Level
from finkfat_ops.config import InfraConfig
from finkfat_ops.deploy import (
    POSTGRES_DB,
    POSTGRES_USER,
    RECORD_FILE,
    REMOTE_DIR,
    DeployRecord,
    compose_script,
    parse_deploy_record,
)
from finkfat_ops.naming import Role, server_name
from finkfat_ops.runner import Command, CommandError, Runner
from finkfat_ops.ssh import ssh_command
from finkfat_ops.vm import ServerInfo

SECTION_MARK = "@@ffat-section "
"""Prefix of the lines separating the sections of :func:`status_script`'s output."""

DISK_WARN_PERCENT = 80
"""Disk usage from which a warning is given."""

CERT_WARN_DAYS = 20
"""Certificate lifetime below which a warning is given (Caddy renews at 30 days)."""

SYNC_WARN_HOURS = 36.0
"""Age of the last nightly sync from which a warning is given (one night was missed)."""

STATUS_TIMEOUT_S = 60.0


# ---------------------------------------------------------------------------- commands


def status_script(role: Role, infra: InfraConfig) -> str:
    """Build the remote shell script printing a role's state in sections (pure).

    Sections: ``record`` (``deployed.json``), ``ps`` (``docker compose ps`` as JSON),
    ``df`` (usage of the data volume, or of ``/``), and on the db ``sync`` (the
    ``fink_fat_sync_state`` row, empty before the first nightly sync).
    """
    # One ssh call per machine instead of four: each part prints a marker line, then its
    # output; parse_sections splits the whole output back on the markers.
    volume = infra.roles.of(role).volume
    mount = volume.mount if volume else "/"
    parts = [
        # Nothing (and no error) when the role was never deployed.
        f"echo '{SECTION_MARK}record'; cat {REMOTE_DIR}/{RECORD_FILE} 2>/dev/null",
        # --all: stopped and crashed containers too, which are exactly what must be seen.
        f"echo '{SECTION_MARK}ps'; ({compose_script('ps', '--all', '--format', 'json')}) "
        "2>/dev/null",
        # -P: one line per file system, whatever the device name's length (easy to parse).
        f"echo '{SECTION_MARK}df'; df -P {mount}",
    ]
    if role == Role.DB:
        # psql inside the postgres container: -A unaligned, -t no header, -F "|" field
        # separator, so the row reads "61001|2026-11-27 03:00:00+00". The table is empty
        # before the first nightly sync and absent before the first migration: both give
        # an empty section (errors are silenced).
        query = "SELECT last_synced_night, synced_at FROM fink_fat_sync_state"
        psql = (
            f"{compose_script('exec', '-T', 'postgres')} psql -U {POSTGRES_USER} "
            f'-d {POSTGRES_DB} -AtF "|" -c "{query}"'
        )
        parts.append(f"echo '{SECTION_MARK}sync'; ({psql}) 2>/dev/null")
    # Parts are joined with ";" (not "&&"): one that fails does not hide the others. The
    # final `true` makes the script succeed, so a failed ssh call only means "unreachable".
    return "; ".join(parts) + "; true"


def status_command(config: Path, role: Role, infra: InfraConfig) -> Command:
    """Run :func:`status_script` on a machine (pure)."""
    return ssh_command(config, role, (status_script(role, infra),), timeout_s=STATUS_TIMEOUT_S)


def certificate_command(domain: str) -> Command:
    """Connect to the explorer over HTTPS, verbosely, to read its certificate (pure)."""
    return Command.of(
        "curl", "--silent", "--verbose", "--head", "--output", "/dev/null", f"https://{domain}/"
    )


# ----------------------------------------------------------------------------- parsing


@dataclass(frozen=True)
class ServiceState:
    """A container as listed by ``docker compose ps``.

    Attributes
    ----------
    name : str
        Compose service, e.g. ``stream``.
    state : str
        ``running``, ``restarting``, ``exited``…
    health : str
        ``healthy``, ``unhealthy``, ``starting``, or empty without a health check.
    status : str
        Docker's human description, e.g. ``Up 3 days (healthy)``.
    """

    name: str
    state: str
    health: str = ""
    status: str = ""

    @property
    def ok(self) -> bool:
        """Running, and not reported unhealthy."""
        return self.state == "running" and self.health != "unhealthy"


@dataclass(frozen=True)
class SyncState:
    """Last nightly swap of the trajectory tables.

    Attributes
    ----------
    night : int
        Last synced night.
    synced_at : datetime
        When it was synced.
    """

    night: int
    synced_at: datetime


def parse_sections(text: str) -> dict[str, str]:
    """Split :func:`status_script`'s output into its sections (pure)."""
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith(SECTION_MARK):
            current = sections.setdefault(line[len(SECTION_MARK) :].strip(), [])
        elif current is not None:
            current.append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def parse_services(text: str) -> list[ServiceState]:
    """Read ``docker compose ps --format json`` (pure).

    Compose prints either one JSON array or one JSON object per line, depending on its
    version; both are accepted. Unreadable lines are ignored.
    """
    items: list[Any] = []
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            items = json.loads(stripped)
        except ValueError:
            items = []
    else:
        for line in stripped.splitlines():
            try:
                items.append(json.loads(line))
            except ValueError:
                continue
    return sorted(
        (
            ServiceState(
                name=str(item.get("Service", "?")),
                state=str(item.get("State", "")),
                health=str(item.get("Health", "")),
                status=str(item.get("Status", "")),
            )
            for item in items
            if isinstance(item, dict)
        ),
        key=lambda service: service.name,
    )


def parse_disk_percent(text: str) -> int | None:
    """Read the usage percentage from ``df -P`` (pure); ``None`` if absent."""
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return None
    match = re.search(r"\s(\d+)%\s", f" {lines[-1]} ")
    return int(match.group(1)) if match else None


def parse_sync(text: str) -> SyncState | None:
    """Read the ``fink_fat_sync_state`` row printed by ``psql -At`` (pure)."""
    night, _, synced_at = text.strip().partition("|")
    try:
        return SyncState(int(night), datetime.fromisoformat(synced_at.strip()))
    except ValueError:
        return None


def parse_certificate_expiry(verbose: str) -> datetime | None:
    """Read the certificate's end of validity from ``curl --verbose`` (pure)."""
    match = re.search(r"expire date: (\w{3} +\d+ [\d:]+ \d{4}) GMT", verbose)
    if match is None:
        return None
    expiry = datetime.strptime(" ".join(match.group(1).split()), "%b %d %H:%M:%S %Y")
    return expiry.replace(tzinfo=UTC)


# ------------------------------------------------------------------------------ report


@dataclass(frozen=True)
class RoleStatus:
    """What was observed on one machine.

    Attributes
    ----------
    role : Role
        The machine.
    vm_status : str
        Nova status, or ``absent``.
    services : list[ServiceState]
        Its containers.
    record : DeployRecord | None
        Its ``deployed.json``; ``None`` if it was never deployed.
    disk_percent : int | None
        Usage of its data volume (``/`` for the explorer).
    error : str | None
        Why the machine could not be read, if it could not.
    """

    role: Role
    vm_status: str
    services: list[ServiceState] = field(default_factory=list)
    record: DeployRecord | None = None
    disk_percent: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class StatusReport:
    """The whole report.

    Attributes
    ----------
    roles : list[RoleStatus]
        One entry per enabled role.
    sync : SyncState | None
        Last nightly sync, read on the db.
    certificate_expiry : datetime | None
        End of validity of the explorer's certificate.
    findings : list[Finding]
        Warnings and errors, each with what to do.
    drifted : list[Role]
        Machines whose configuration differs from ``deploy/``.
    """

    roles: list[RoleStatus]
    sync: SyncState | None
    certificate_expiry: datetime | None
    findings: list[Finding]
    drifted: list[Role] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        """No finding in error."""
        return all(finding.level != Level.ERROR for finding in self.findings)


def vm_states(infra: InfraConfig, servers: Sequence[ServerInfo]) -> dict[Role, str]:
    """Map each role to its VM's Nova status, ``absent`` when no VM has its name (pure)."""
    by_name = {server.name: server.status for server in servers}
    return {
        role: by_name.get(server_name(infra.env, role), "absent") for role, _ in infra.roles.items()
    }


def role_status(role: Role, vm_status: str, output: str | None, error: str | None) -> RoleStatus:
    """Build a role's status from its :func:`status_script` output (pure)."""
    if output is None:
        return RoleStatus(role, vm_status, error=error or "not reachable")
    sections = parse_sections(output)
    return RoleStatus(
        role=role,
        vm_status=vm_status,
        services=parse_services(sections.get("ps", "")),
        record=parse_deploy_record(sections.get("record", "")),
        disk_percent=parse_disk_percent(sections.get("df", "")),
    )


def has_drift(status: RoleStatus, local_fingerprint: str | None) -> bool:
    """Whether a deployed machine runs another configuration than ``deploy/`` (pure)."""
    record = status.record
    return (
        record is not None and local_fingerprint is not None and record.config != local_fingerprint
    )


def _role_findings(
    status: RoleStatus, expected_tag: str, local_fingerprint: str | None
) -> list[Finding]:
    """Warnings and errors about one machine (pure)."""
    role = status.role
    if status.vm_status != "ACTIVE":
        return [
            Finding(
                Level.ERROR,
                f"{role}: VM {status.vm_status}",
                f"ff-ops vm list; ff-ops vm start {role}",
            )
        ]
    if status.error:
        return [Finding(Level.ERROR, f"{role}: {status.error}", f"ff-ops ssh {role}")]
    if status.record is None:
        return [Finding(Level.WARN, f"{role}: never deployed", f"ff-ops deploy {role}")]
    findings = [
        Finding(
            Level.ERROR,
            f"{role}: {service.name} is {service.health or service.state} ({service.status})",
            f"ff-ops logs {role} {service.name}",
        )
        for service in status.services
        if not service.ok
    ]
    if not status.services:
        findings.append(Finding(Level.ERROR, f"{role}: no container", f"ff-ops deploy {role}"))
    if role != Role.DB and status.record.tag != expected_tag:
        findings.append(
            Finding(
                Level.WARN,
                f"{role}: runs {status.record.tag}, infra.yaml says {expected_tag}",
                f"ff-ops deploy {role}",
            )
        )
    if has_drift(status, local_fingerprint):
        findings.append(
            Finding(
                Level.WARN,
                f"{role}: configuration differs from deploy/ (drift)",
                f"ff-ops deploy {role}",
            )
        )
    if status.disk_percent is not None and status.disk_percent >= DISK_WARN_PERCENT:
        findings.append(
            Finding(
                Level.WARN,
                f"{role}: disk {status.disk_percent}% full",
                "free space, or raise the volume's size_gb in infra.yaml"
                if role != Role.EXPLORER
                else f"ff-ops ssh {role} -- docker system prune",
            )
        )
    return findings


@dataclass(frozen=True)
class Expected:
    """What ``deploy/`` says should be running.

    Attributes
    ----------
    tag : str
        ``images.tag`` of ``infra.yaml``.
    fingerprints : Mapping[Role, str]
        Fingerprint of what ``ff-ops deploy`` would send now, per role.
    domain : str | None
        ``explorer_domain``.
    """

    tag: str
    fingerprints: Mapping[Role, str]
    domain: str | None


def build_report(
    statuses: Sequence[RoleStatus],
    expected: Expected,
    *,
    sync: SyncState | None,
    certificate_expiry: datetime | None,
    now: datetime,
) -> StatusReport:
    """Decide what deserves attention (pure).

    Parameters
    ----------
    statuses : Sequence[RoleStatus]
        What was observed on each enabled machine.
    expected : Expected
        What should be running.
    sync : SyncState | None
        Last nightly sync.
    certificate_expiry : datetime | None
        End of validity of the explorer's certificate, if it could be read.
    now : datetime
        Current time (timezone-aware).

    Returns
    -------
    StatusReport
        Errors: VM not active, machine unreachable, container down or unhealthy, certificate
        unreadable. Warnings: never deployed, other tag, configuration drift, disk filling
        up, certificate close to expiry, no nightly sync for :data:`SYNC_WARN_HOURS`.
    """
    domain = expected.domain
    findings: list[Finding] = []
    for status in statuses:
        findings += _role_findings(status, expected.tag, expected.fingerprints.get(status.role))
    db_up = any(s.role == Role.DB and s.services and not s.error for s in statuses)
    if db_up:
        if sync is None:
            findings.append(
                Finding(Level.WARN, "no nightly sync yet", "ff-ops logs compute stream")
            )
        elif (now - sync.synced_at).total_seconds() > SYNC_WARN_HOURS * 3600:
            findings.append(
                Finding(
                    Level.WARN,
                    f"last nightly sync is {format_age(now, sync.synced_at)} old",
                    "ff-ops logs compute stream",
                )
            )
    if domain:
        if certificate_expiry is None:
            findings.append(
                Finding(
                    Level.ERROR,
                    f"https://{domain}/: no valid certificate",
                    "ff-ops logs explorer caddy",
                )
            )
        elif (certificate_expiry - now).days < CERT_WARN_DAYS:
            findings.append(
                Finding(
                    Level.WARN,
                    f"certificate expires in {(certificate_expiry - now).days} days",
                    "ff-ops logs explorer caddy (Caddy should have renewed it)",
                )
            )
    drifted = [s.role for s in statuses if has_drift(s, expected.fingerprints.get(s.role))]
    return StatusReport(list(statuses), sync, certificate_expiry, findings, drifted)


def format_age(now: datetime, then: datetime) -> str:
    """Write the time elapsed since ``then`` as ``5 h`` or ``3 d`` (pure)."""
    hours = (now - then).total_seconds() / 3600
    return f"{hours:.0f} h" if hours < 48 else f"{hours / 24:.0f} d"


# --------------------------------------------------------------------------------- I/O


def collect_role(
    runner: Runner, config: Path, role: Role, infra: InfraConfig, vm_status: str
) -> tuple[RoleStatus, str]:
    """Read a machine's state over SSH (I/O).

    Returns
    -------
    tuple[RoleStatus, str]
        The status, and the raw output (the db's ``sync`` section is read from it).
    """
    # A stopped or missing VM cannot answer: no ssh attempt (it would wait for its timeout).
    if vm_status != "ACTIVE":
        return RoleStatus(role, vm_status), ""
    # Never raise: an unreachable machine becomes a row in error, and the other machines
    # are still shown.
    try:
        result = runner.run(status_command(config, role, infra), check=False)
    except CommandError as error:  # timeout, or ssh missing
        return role_status(role, vm_status, None, str(error).splitlines()[0]), ""
    # The script itself always succeeds (final `true`): a failure here is ssh's own.
    if not result.ok:
        detail = (result.stderr or result.stdout).strip().splitlines()
        return role_status(role, vm_status, None, detail[-1] if detail else "ssh failed"), ""
    return role_status(role, vm_status, result.stdout, None), result.stdout


def read_certificate_expiry(runner: Runner, domain: str) -> datetime | None:
    """Read the explorer's certificate expiry from this machine (I/O)."""
    try:
        result = runner.run(certificate_command(domain), check=False)
    except CommandError:
        return None
    return parse_certificate_expiry(result.stderr) if result.ok else None
