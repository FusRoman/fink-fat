"""Configuration of the production deployment.

Three files, merged into one :class:`Deployment` (see ``deploy/README.md``, "Configuration"):

* ``infra.yaml`` — versioned structure (flavors, network, roles). Nothing personal.
* ``infra.local.yaml`` — gitignored: cloud profile, project, allowed SSH ranges, domain.
* ``secrets.env`` — gitignored: passwords and tokens.

Parsing and validation are pure (``parse_*`` functions take already-read data);
only the ``load_*`` functions touch the filesystem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from ipaddress import IPv4Address, IPv4Network
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

from finkfat_ops.naming import Role

LOCAL_ONLY_KEYS = frozenset(
    {"openstack", "admin_cidrs", "explorer_domain", "ssh_identity_file", "dns_nameservers"}
)
"""Keys that belong in ``infra.local.yaml`` and must never appear in the versioned file."""

MIN_ROOT_GB = 20
"""Smallest boot volume accepted (the flavors' own root disk size)."""

MAX_PREFIXLEN = 24
"""Smallest private subnet accepted (a /24)."""

DHCP_POOL_OFFSET = 100
"""Offset of the first DHCP address; fixed IPs must stay below it."""

_NAME_PATTERN = r"^[a-z][a-z0-9-]{0,30}$"
_HOSTNAME_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")


class ConfigError(Exception):
    """One or more configuration problems, each as a human-readable line."""

    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


class _Strict(BaseModel):
    """Base model: unknown keys are errors (a typo must never be silently ignored)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- infra.yaml


class VolumeConfig(_Strict):
    """A persistent data volume attached to a role.

    Attributes
    ----------
    name : str
        Short name, used in the volume name ``ffat-<env>-vol-<name>``.
    size_gb : int
        Size in GB (can only grow).
    mount : str
        Absolute mount point on the VM, under ``/srv/``.
    """

    name: str = Field(pattern=_NAME_PATTERN)
    size_gb: int = Field(ge=1)
    mount: str = Field(pattern=r"^/srv/[a-z0-9_-]+$")


class RoleConfig(_Strict):
    """One production machine.

    Attributes
    ----------
    enabled : bool
        ``False`` removes the VM on the next apply; its data volume is kept.
    flavor : str
        OpenStack flavor name.
    root_gb : int
        Size of the boot volume.
    fixed_ip : IPv4Address
        Fixed private IP inside the private subnet.
    volume : VolumeConfig | None
        Persistent data volume, if the role has one.
    floating_ip : bool
        Whether the VM gets the public floating IP (bastion only).
    """

    enabled: bool = True
    flavor: str = Field(min_length=1)
    root_gb: int = Field(ge=MIN_ROOT_GB)
    fixed_ip: IPv4Address
    volume: VolumeConfig | None = None
    floating_ip: bool = False


class RolesConfig(_Strict):
    """The three roles; all three must be described."""

    compute: RoleConfig
    db: RoleConfig
    explorer: RoleConfig

    def items(self) -> list[tuple[Role, RoleConfig]]:
        """Return ``(role, config)`` pairs in a stable order."""
        return [(Role.COMPUTE, self.compute), (Role.DB, self.db), (Role.EXPLORER, self.explorer)]

    def of(self, role: Role) -> RoleConfig:
        """Return the configuration of ``role``."""
        return dict(self.items())[role]


class NetworkConfig(_Strict):
    """The private network.

    Attributes
    ----------
    cidr : IPv4Network
        Private subnet range.
    external_network : str
        External network used for the router gateway and floating IP.
    """

    cidr: IPv4Network
    external_network: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_size(self) -> Self:
        """Require at least a /24, so that fixed IPs and the DHCP pool never overlap."""
        if self.cidr.prefixlen > MAX_PREFIXLEN:
            raise ValueError(f"cidr {self.cidr} is too small: use a /{MAX_PREFIXLEN} or larger")
        return self

    def dhcp_pool(self) -> tuple[IPv4Address, IPv4Address]:
        """Return the first and last address handed out by DHCP.

        Fixed IPs live below the pool (offsets 2 to 99), so that no other port can
        ever receive them.

        Returns
        -------
        tuple[IPv4Address, IPv4Address]
            Start and end of the DHCP allocation pool.
        """
        return (
            self.cidr.network_address + DHCP_POOL_OFFSET,
            self.cidr.broadcast_address - 1,
        )


class ImagesConfig(_Strict):
    """Docker images.

    Attributes
    ----------
    registry : str
        Registry and owner, e.g. ``ghcr.io/fusroman``.
    tag : str
        Default image tag deployed (``ff-ops update --tag`` overrides it).
    """

    registry: str = Field(min_length=1)
    tag: str = Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


class InfraConfig(_Strict):
    """Content of the versioned ``infra.yaml``."""

    env: str = Field(pattern=r"^[a-z][a-z0-9]{0,15}$")
    keypair: str = Field(min_length=1)
    heat_template_version: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    availability_zone: str = Field(min_length=1)
    os_image: str = Field(min_length=1)
    ssh_user: str = Field(pattern=r"^[a-z_][a-z0-9_-]{0,31}$")
    volume_type: str = Field(min_length=1)
    network: NetworkConfig
    images: ImagesConfig
    roles: RolesConfig

    @field_validator("heat_template_version", mode="before")
    @classmethod
    def _date_to_text(cls, value: Any) -> Any:
        """Accept an unquoted YAML date (``2021-04-16`` is parsed as a ``date``)."""
        return value.isoformat() if isinstance(value, date) else value

    @model_validator(mode="after")
    def _check_roles(self) -> Self:
        """Cross-field checks between roles and the network (see :func:`role_problems`)."""
        problems = role_problems(self.network, self.roles)
        if problems:
            raise ValueError("; ".join(problems))
        return self


def role_problems(network: NetworkConfig, roles: RolesConfig) -> list[str]:
    """List the inconsistencies between the roles and the private network.

    Checked rules:

    * each fixed IP is a usable host of the subnet, not its gateway (first host), and
      lies below the DHCP pool;
    * fixed IPs and data volume names are unique;
    * compute and db have a data volume (they hold the data);
    * exactly the explorer carries the floating IP (it is the bastion).

    Parameters
    ----------
    network : NetworkConfig
        The private network configuration.
    roles : RolesConfig
        The three roles.

    Returns
    -------
    list[str]
        One message per problem; empty when consistent.
    """
    problems: list[str] = []
    hosts = network.cidr.hosts()
    gateway = next(hosts, None)
    ips: dict[IPv4Address, Role] = {}
    volumes: dict[str, Role] = {}
    for role, cfg in roles.items():
        if cfg.fixed_ip not in network.cidr or cfg.fixed_ip in (
            network.cidr.network_address,
            network.cidr.broadcast_address,
        ):
            problems.append(f"roles.{role}.fixed_ip {cfg.fixed_ip} is not a host of {network.cidr}")
        elif cfg.fixed_ip == gateway:
            problems.append(f"roles.{role}.fixed_ip {cfg.fixed_ip} is the router gateway address")
        elif cfg.fixed_ip >= network.dhcp_pool()[0]:
            problems.append(
                f"roles.{role}.fixed_ip {cfg.fixed_ip} is inside the DHCP pool "
                f"(starting at {network.dhcp_pool()[0]})"
            )
        if cfg.fixed_ip in ips:
            problems.append(f"roles.{role}.fixed_ip duplicates roles.{ips[cfg.fixed_ip]}.fixed_ip")
        ips.setdefault(cfg.fixed_ip, role)
        if cfg.volume is not None:
            if cfg.volume.name in volumes:
                problems.append(
                    f"roles.{role}.volume.name '{cfg.volume.name}' "
                    f"is also used by roles.{volumes[cfg.volume.name]}"
                )
            volumes.setdefault(cfg.volume.name, role)
        elif role in (Role.COMPUTE, Role.DB):
            problems.append(f"roles.{role} needs a data volume (it holds production data)")
        if cfg.floating_ip != (role == Role.EXPLORER):
            problems.append(
                f"roles.{role}.floating_ip must be {role == Role.EXPLORER} "
                "(only the explorer, the SSH bastion, is public)"
            )
    return problems


# --------------------------------------------------------------------- infra.local.yaml


class OpenStackTarget(_Strict):
    """Which OpenStack project ff-ops talks to.

    Attributes
    ----------
    cloud : str
        Profile name in ``~/.config/openstack/clouds.yaml`` (passed as ``--os-cloud``).
    project : str
        Expected project name; ff-ops refuses to act on any other project.
    ask_password : bool
        The profile authenticates with a password kept out of ``clouds.yaml``: ff-ops asks
        for it once per command (or reads ``OS_PASSWORD``) and passes it to ``openstack``.
    """

    cloud: str = Field(min_length=1)
    project: str = Field(min_length=1)
    ask_password: bool = False


class LocalConfig(_Strict):
    """Content of the gitignored ``infra.local.yaml``.

    Attributes
    ----------
    openstack : OpenStackTarget
        Cloud profile and expected project.
    admin_cidrs : list[IPv4Network]
        Source ranges allowed to SSH into the bastion.
    explorer_domain : str | None
        Public domain name of the explorer; ``None`` for no HTTPS certificate.
    ssh_identity_file : str | None
        Private key matching the Nova keypair (``~`` allowed); ``None`` lets ``ssh`` use
        its default keys and agent.
    dns_nameservers : list[IPv4Address]
        DNS servers handed out by DHCP on the private subnet. Empty keeps Neutron's
        default (its DHCP server), which does not resolve external names on every cloud.
    """

    openstack: OpenStackTarget
    admin_cidrs: list[IPv4Network] = Field(min_length=1)
    explorer_domain: str | None = None
    ssh_identity_file: str | None = Field(default=None, min_length=1)
    dns_nameservers: list[IPv4Address] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Self:
        """Reject an SSH rule open to the whole Internet and malformed domain names."""
        for cidr in self.admin_cidrs:
            if cidr.prefixlen == 0:
                raise ValueError(
                    f"admin_cidrs contains {cidr}: SSH must not be open to the whole Internet"
                )
        if self.explorer_domain is not None and not _HOSTNAME_RE.match(self.explorer_domain):
            raise ValueError(f"explorer_domain '{self.explorer_domain}' is not a valid domain name")
        return self


# -------------------------------------------------------------------------- secrets.env


class Secrets(_Strict):
    """Content of the gitignored ``secrets.env``.

    Values are :class:`~pydantic.SecretStr`, so they never appear in ``repr`` or logs.

    Attributes
    ----------
    POSTGRES_PASSWORD : SecretStr
        Password of the Postgres user used by stream and explorer.
    PGADMIN_PASSWORD : SecretStr
        Password of the pgAdmin web UI.
    FINK_FAT_KAFKA_PASSWORD : SecretStr | None
        SASL password of the Fink Kafka account, if it has one.
    GHCR_READ_TOKEN : SecretStr | None
        ``read:packages`` token, only needed if the images are private.
    """

    POSTGRES_PASSWORD: SecretStr = Field(min_length=12)
    PGADMIN_PASSWORD: SecretStr = Field(min_length=12)
    FINK_FAT_KAFKA_PASSWORD: SecretStr | None = None
    GHCR_READ_TOKEN: SecretStr | None = None


def parse_env_text(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines (``.env`` syntax subset).

    Blank lines and ``#`` comments are skipped; a value may be wrapped in single or
    double quotes; an empty value is treated as absent.

    Parameters
    ----------
    text : str
        File content.

    Returns
    -------
    dict[str, str]
        The non-empty assignments.

    Raises
    ------
    ConfigError
        A line is not ``KEY=VALUE`` or a key is repeated.
    """
    values: dict[str, str] = {}
    problems: list[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            problems.append(f"line {number}: expected KEY=VALUE")
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key in values:
            problems.append(f"line {number}: {key} is set twice")
        if value:
            values[key] = value
    if problems:
        raise ConfigError(problems)
    return values


# ----------------------------------------------------------------------- parse / merge


def _format_validation_error(source: str, error: ValidationError) -> list[str]:
    """Turn a pydantic error into one ``source: field.path: message`` line per problem.

    Secret values are never echoed back (``input`` is deliberately not printed).
    """
    lines = []
    for item in error.errors():
        where = ".".join(str(part) for part in item["loc"]) or "(root)"
        message = item["msg"].removeprefix("Value error, ")
        lines.append(f"{source}: {where}: {message}")
    return lines


def parse_infra(data: Any, source: str = "infra.yaml") -> InfraConfig:
    """Validate the content of ``infra.yaml``.

    Parameters
    ----------
    data : Any
        Parsed YAML.
    source : str
        File name used in error messages.

    Raises
    ------
    ConfigError
        Invalid content, or a local-only key found in the versioned file.
    """
    if not isinstance(data, dict):
        raise ConfigError([f"{source}: expected a mapping at the top level"])
    leaked = sorted(LOCAL_ONLY_KEYS & data.keys())
    if leaked:
        raise ConfigError(
            [
                f"{source}: '{key}' must live in infra.local.yaml (gitignored), "
                "not in the versioned file"
                for key in leaked
            ]
        )
    try:
        return InfraConfig.model_validate(data)
    except ValidationError as error:
        raise ConfigError(_format_validation_error(source, error)) from error


def parse_local(data: Any, source: str = "infra.local.yaml") -> LocalConfig:
    """Validate the content of ``infra.local.yaml``.

    Raises
    ------
    ConfigError
        Invalid content.
    """
    if not isinstance(data, dict):
        raise ConfigError([f"{source}: expected a mapping at the top level"])
    try:
        return LocalConfig.model_validate(data)
    except ValidationError as error:
        raise ConfigError(_format_validation_error(source, error)) from error


def parse_secrets(values: dict[str, str], source: str = "secrets.env") -> Secrets:
    """Validate the assignments of ``secrets.env``.

    Raises
    ------
    ConfigError
        Unknown key (typo), missing or too short required secret.
    """
    try:
        return Secrets.model_validate(values)
    except ValidationError as error:
        raise ConfigError(_format_validation_error(source, error)) from error


@dataclass(frozen=True)
class Deployment:
    """The whole validated configuration.

    Attributes
    ----------
    infra : InfraConfig
        Versioned structure.
    local : LocalConfig
        Operator-specific settings.
    secrets : Secrets
        Passwords and tokens.
    """

    infra: InfraConfig
    local: LocalConfig
    secrets: Secrets


# ------------------------------------------------------------------------------- I/O


@dataclass(frozen=True)
class DeployPaths:
    """Locations of the files ff-ops reads and writes, relative to ``deploy/``.

    Attributes
    ----------
    root : Path
        The ``deploy/`` directory.
    """

    root: Path

    @property
    def infra(self) -> Path:
        """Versioned structure file."""
        return self.root / "infra.yaml"

    @property
    def infra_local(self) -> Path:
        """Gitignored local settings."""
        return self.root / "infra.local.yaml"

    @property
    def secrets(self) -> Path:
        """Gitignored secrets."""
        return self.root / "secrets.env"

    @property
    def build_dir(self) -> Path:
        """Generated Heat templates."""
        return self.root / ".build"

    @property
    def state_dir(self) -> Path:
        """Generated SSH configuration and known hosts."""
        return self.root / ".state"

    def sensitive_files(self) -> list[Path]:
        """Files that must never be tracked by git."""
        return [self.infra_local, self.secrets]


def read_yaml(path: Path) -> Any:
    """Read a YAML file.

    Raises
    ------
    ConfigError
        Missing file or invalid YAML.
    """
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError([f"{path.name}: file not found ({path})"]) from error
    except yaml.YAMLError as error:
        raise ConfigError([f"{path.name}: invalid YAML: {error}"]) from error


def read_env_file(path: Path) -> dict[str, str]:
    """Read a ``KEY=VALUE`` file (see :func:`parse_env_text`).

    Raises
    ------
    ConfigError
        Missing file or invalid line.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConfigError([f"{path.name}: file not found ({path})"]) from error
    try:
        return parse_env_text(text)
    except ConfigError as error:
        raise ConfigError([f"{path.name}: {p}" for p in error.problems]) from error


def load_deployment(paths: DeployPaths) -> Deployment:
    """Read and validate the three configuration files.

    All files are checked before failing, so every problem is reported at once.

    Raises
    ------
    ConfigError
        Every problem found across the three files.
    """
    problems: list[str] = []
    infra = local = secrets = None
    try:
        infra = parse_infra(read_yaml(paths.infra), paths.infra.name)
    except ConfigError as error:
        problems += error.problems
    try:
        local = parse_local(read_yaml(paths.infra_local), paths.infra_local.name)
    except ConfigError as error:
        problems += error.problems
    try:
        secrets = parse_secrets(read_env_file(paths.secrets), paths.secrets.name)
    except ConfigError as error:
        problems += error.problems
    if problems or infra is None or local is None or secrets is None:
        raise ConfigError(problems)
    return Deployment(infra=infra, local=local, secrets=secrets)
