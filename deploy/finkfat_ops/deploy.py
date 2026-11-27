"""Deployment of the services on the machines (``ff-ops deploy``).

Each role receives one *bundle* in ``/opt/finkfat`` on its VM:

* ``compose.yml`` — the role's Docker Compose file (``deploy/compose/<role>.yml``);
* ``config/`` — the production fink-fat configuration (compute and explorer);
* ``Caddyfile`` — generated from ``explorer_domain`` (explorer);
* ``.env`` — mode ``600``, only the variables and secrets this role needs;
* ``deployed.json`` — written last, once the health checks pass: tag, configuration
  fingerprint and date of the running deployment (read by ``ff-ops status``).

The bundle travels as one compressed tar archive on the standard input of ``ssh``: no
secret ever appears on a command line, and the machines need nothing beyond ``tar`` and
``base64``. Then ``docker compose pull`` and ``up -d`` start the services, and role-specific
health checks decide whether the deployment succeeded.

Everything above :func:`deploy_role` is pure: it builds texts, archives and
:class:`~finkfat_ops.runner.Command` values. :func:`deploy_role` runs them.

How one ``ff-ops deploy`` unfolds (``cli.deploy`` → :func:`deploy_role` per role)::

    cli.deploy
      load the configuration, pick the roles (ordered_roles: db → compute → explorer)
      ask "Proceed?"
      for each role:
        deploy_role
          1. read deploy/compose/<role>.yml and deploy/config/*      (static_files, read_files)
          2. ssh: create /opt/finkfat (+ chown of the compute data)  (prepare_command)
          3. ssh: read /proc/meminfo — db only                       (memory_command)
          4. build the bundle and its .env, pack it                  (build_bundle, pack)
          5. ssh: unpack the archive received on stdin               (upload_command)
          6. ssh: docker login — only with GHCR_READ_TOKEN           (login_command)
          7. ssh: docker compose pull, then up -d                    (compose_command)
          8. ssh: health checks of the role                          (health_checks)
             explorer: + HTTPS check from the workstation            (public_check_command)
          9. ssh: write deployed.json — only if no check failed     (record_command)
        a role in error stops the loop: the next roles are not touched

Every ``ssh …`` above is built by :func:`finkfat_ops.ssh.ssh_command` and goes through the
bastion (``.state/ssh_config``); the remote part is one shell script, run by the remote
user's shell.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import shlex
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from finkfat_ops import images
from finkfat_ops.checks import Finding, Level
from finkfat_ops.config import Deployment, DeployPaths, InfraConfig, Secrets
from finkfat_ops.naming import Role
from finkfat_ops.runner import Command, CommandError, CommandResult, Runner
from finkfat_ops.ssh import RemoteCheck, evaluate_check, ssh_command

REMOTE_DIR = "/opt/finkfat"
"""Directory of the bundle on every machine."""

ENV_FILE = ".env"
"""Name of the per-role environment file (read by Docker Compose)."""

RECORD_FILE = "deployed.json"
"""Name of the deployment record, written once the health checks pass."""

ENGINE_UID = 10001
"""UID of the ``finkfat`` user of the engine image (``dockerfile``), owner of the data volume."""

POSTGRES_USER = "finkfat"
"""Postgres role used by the stream and the explorer."""

POSTGRES_DB = "finkfat"
"""Database of the production instance."""

POSTGRES_PORT = 5432

PGADMIN_EMAIL = "admin@example.org"
"""pgAdmin login name (only a login: pgAdmin requires the form of an e-mail address)."""

DEPLOYMENT_ORDER = (Role.DB, Role.COMPUTE, Role.EXPLORER)
"""Safe order: the database first, then its writer, then its reader."""

PULL_TIMEOUT_S = 900.0
"""Longest ``docker compose pull`` (first pull of the images on a fresh VM)."""

UP_TIMEOUT_S = 900.0
"""Longest ``docker compose up -d`` (the stream may need its 10 min grace period to stop)."""

DRY_RUN_MEMORY_BYTES = 32 * 1024**3
"""Memory assumed for the Postgres settings in dry-run mode, where nothing is read."""

_MIB = 1024**2


# ------------------------------------------------------------------------------- files


def static_files(role: Role, paths: DeployPaths) -> dict[str, Path]:
    """Map the bundle paths of a role to the versioned files they come from (pure).

    Parameters
    ----------
    role : Role
        The machine.
    paths : DeployPaths
        Locations of ``deploy/compose`` and ``deploy/config``.

    Returns
    -------
    dict[str, Path]
        ``compose.yml`` for every role; the engine configuration for the compute and the
        explorer (both must use the same one); ``stream.yaml`` for the compute.
    """
    files = {"compose.yml": paths.compose_dir / f"{role}.yml"}
    if role in (Role.COMPUTE, Role.EXPLORER):
        files["config/engine_config.yaml"] = paths.fink_config_dir / "engine_config.yaml"
    if role == Role.COMPUTE:
        # Gitignored (broker address and consumer group): it exists on the operator's
        # workstation only, copied from config/stream.example.yaml.
        files["config/stream.yaml"] = paths.stream_config
    return files


def read_files(sources: Mapping[str, Path]) -> dict[str, bytes]:
    """Read the files of :func:`static_files` (I/O).

    Raises
    ------
    ValueError
        A file is missing.
    """
    missing = [path for path in sources.values() if not path.is_file()]
    if missing:
        raise ValueError(missing_files_message(missing))
    return {name: path.read_bytes() for name, path in sources.items()}


def missing_files(roles: Sequence[Role], paths: DeployPaths) -> list[Path]:
    """List the files that deploying ``roles`` needs and that do not exist (reads the disk).

    ``ff-ops deploy`` calls it before touching any machine, so that a missing file of the
    compute is reported before the db is redeployed; ``config check`` and ``doctor`` too.
    """
    needed = {path for role in roles for path in static_files(role, paths).values()}
    return sorted(path for path in needed if not path.is_file())


def missing_files_message(missing: Sequence[Path]) -> str:
    """One-line error naming the missing files, with how to create them (reads the disk)."""
    return f"missing deployment file(s): {', '.join(map(_missing_hint, missing))}"


def file_findings(roles: Sequence[Role], paths: DeployPaths) -> list[Finding]:
    """Check that every file needed to deploy ``roles`` exists (reads the disk).

    Returns
    -------
    list[Finding]
        One ERROR per missing file, with the command creating it from its template (a
        local file such as ``stream.yaml``) or restoring it from git (a versioned file);
        otherwise a single OK finding.
    """
    missing = missing_files(roles, paths)
    if not missing:
        return [Finding(Level.OK, "deployment files present (compose/, config/)")]
    return [
        Finding(Level.ERROR, f"{path.relative_to(paths.root)} is missing", _create_fix(path))
        for path in missing
    ]


def example_of(path: Path) -> Path:
    """Return the versioned template of a local file, e.g. ``stream.example.yaml`` (pure)."""
    return path.with_name(f"{path.stem}.example{path.suffix}")


def _missing_hint(path: Path) -> str:
    """Name a missing file, with how to create it when it has a template."""
    template = example_of(path)
    return f"{path} (copy {template.name} and fill it in)" if template.is_file() else str(path)


def _create_fix(path: Path) -> str:
    """Command creating a missing file: from its template if it has one, else from git."""
    template = example_of(path)
    if template.is_file():
        return f"cp {template} {path} && chmod 600 {path}, then fill it in"
    return f"git checkout -- {path}"


def render_caddyfile(domain: str | None) -> str:
    """Build the Caddyfile of the explorer (pure).

    Parameters
    ----------
    domain : str | None
        Public name of the explorer. Caddy then obtains and renews a Let's Encrypt
        certificate on its own and redirects HTTP to HTTPS. Without a domain no
        certificate can be obtained, so the explorer is served over plain HTTP on port 80.

    Returns
    -------
    str
        Caddyfile text forwarding every request to the ``app`` service.
    """
    return (
        "# Generated by ff-ops from explorer_domain (infra.local.yaml). Do not edit.\n"
        f"{domain or ':80'} {{\n"
        "    encode zstd gzip\n"
        "    reverse_proxy app:8080\n"
        "}\n"
    )


def generated_files(role: Role, deployment: Deployment) -> dict[str, bytes]:
    """Return the bundle files generated from the configuration (pure): the Caddyfile."""
    if role != Role.EXPLORER:
        return {}
    return {"Caddyfile": render_caddyfile(deployment.local.explorer_domain).encode()}


def config_fingerprint(files: Mapping[str, bytes]) -> str:
    """Fingerprint the deployed configuration (pure).

    Parameters
    ----------
    files : Mapping[str, bytes]
        Bundle files, without the ``.env`` (secrets are not fingerprinted, so the
        fingerprint can be shown and stored freely).

    Returns
    -------
    str
        The first 12 hexadecimal digits of a SHA-256 over the sorted names and contents.
    """
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode() + b"\0" + hashlib.sha256(files[name]).digest())
    return digest.hexdigest()[:12]


def local_fingerprint(role: Role, deployment: Deployment, paths: DeployPaths) -> str:
    """Fingerprint of the configuration a deployment of ``role`` would send now (reads files).

    Compared with the fingerprint of ``deployed.json``, it tells whether the machine runs
    the configuration of ``deploy/`` (``ff-ops status``).
    """
    versioned = read_files(static_files(role, paths))
    return config_fingerprint({**versioned, **generated_files(role, deployment)})


# ------------------------------------------------------------------------- environment


def parse_meminfo(text: str) -> int:
    """Read the total memory in bytes from ``/proc/meminfo`` (pure).

    Raises
    ------
    ValueError
        No ``MemTotal`` line.
    """
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.MULTILINE)
    if match is None:
        raise ValueError("could not read the machine's memory (/proc/meminfo has no MemTotal)")
    return int(match.group(1)) * 1024


def postgres_settings(memory_bytes: int) -> dict[str, str]:
    """Size Postgres for the machine's memory (pure).

    Parameters
    ----------
    memory_bytes : int
        Total memory of the db machine.

    Returns
    -------
    dict[str, str]
        ``shared_buffers`` (25 %), ``effective_cache_size`` (75 %), ``maintenance_work_mem``
        (5 %, at most 2 GB), and the write-ahead log settings of the local development
        setup (``max_wal_size`` 16 GB, ``checkpoint_timeout`` 30 min), keyed by the
        ``.env`` variable that ``compose/db.yml`` reads.
    """
    mib = memory_bytes // _MIB
    return {
        "PG_SHARED_BUFFERS": f"{mib // 4}MB",
        "PG_EFFECTIVE_CACHE_SIZE": f"{mib * 3 // 4}MB",
        "PG_MAINTENANCE_WORK_MEM": f"{min(2048, mib // 20)}MB",
        "PG_MAX_WAL_SIZE": "16GB",
        "PG_CHECKPOINT_TIMEOUT": "30min",
    }


def database_url(infra: InfraConfig, secrets: Secrets) -> str:
    """Build the Postgres URL used by the stream and the explorer (pure).

    The password is percent-encoded, so any character is safe in it.
    """
    password = quote(secrets.POSTGRES_PASSWORD.get_secret_value(), safe="")
    host = infra.roles.db.fixed_ip
    return f"postgres://{POSTGRES_USER}:{password}@{host}:{POSTGRES_PORT}/{POSTGRES_DB}"


def role_env(
    role: Role, deployment: Deployment, tag: str, memory_bytes: int | None = None
) -> dict[str, str]:
    """List the ``.env`` variables of a role: only what its services need (pure).

    Parameters
    ----------
    role : Role
        The machine.
    deployment : Deployment
        Validated configuration, secrets included.
    tag : str
        Image tag to run (compute and explorer).
    memory_bytes : int | None
        Memory of the db machine; required for the db.

    Returns
    -------
    dict[str, str]
        db: Postgres credentials, private bind address and settings; compute: database URL
        and Kafka password; explorer: database URL and pgAdmin credentials. The compute and
        the explorer also get the image registry and tag.

    Raises
    ------
    ValueError
        ``memory_bytes`` is missing for the db.
    """
    infra, secrets = deployment.infra, deployment.secrets
    if role == Role.DB:
        if memory_bytes is None:
            raise ValueError("the db needs its memory size to size Postgres")
        return {
            "POSTGRES_USER": POSTGRES_USER,
            "POSTGRES_PASSWORD": secrets.POSTGRES_PASSWORD.get_secret_value(),
            "POSTGRES_DB": POSTGRES_DB,
            "DB_BIND_IP": str(infra.roles.db.fixed_ip),
            **postgres_settings(memory_bytes),
        }
    images_env = {"FFAT_REGISTRY": infra.images.registry, "FFAT_TAG": tag}
    if role == Role.COMPUTE:
        kafka = secrets.FINK_FAT_KAFKA_PASSWORD
        return {
            **images_env,
            "FINK_FAT_DATABASE_URL": database_url(infra, secrets),
            "FINK_FAT_KAFKA_PASSWORD": kafka.get_secret_value() if kafka else "",
        }
    return {
        **images_env,
        "DATABASE_URL": database_url(infra, secrets),
        "PGADMIN_EMAIL": PGADMIN_EMAIL,
        "PGADMIN_PASSWORD": secrets.PGADMIN_PASSWORD.get_secret_value(),
    }


def render_env(values: Mapping[str, str]) -> str:
    """Write ``.env`` text for Docker Compose (pure).

    Values are single-quoted: Compose then takes them literally (no ``$`` interpolation).

    Raises
    ------
    ValueError
        A value holds a single quote or a line break, which cannot be written literally
        (the variable is named, never its value).
    """
    lines = ["# Generated by ff-ops deploy. Do not edit: redeploy instead."]
    for key, value in values.items():
        if "'" in value or "\n" in value or "\r" in value:
            raise ValueError(f"{key} holds a quote or a line break, which .env cannot hold")
        lines.append(f"{key}='{value}'")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- bundle


@dataclass(frozen=True)
class Bundle:
    """What is sent to a machine.

    Attributes
    ----------
    role : Role
        The machine.
    tag : str
        Image tag deployed.
    files : dict[str, bytes]
        Versioned and generated files, by path under :data:`REMOTE_DIR` (no secret).
    env : str
        ``.env`` text (secrets): sent with mode ``600``, never fingerprinted nor shown.
    """

    role: Role
    tag: str
    files: dict[str, bytes]
    env: str

    @property
    def fingerprint(self) -> str:
        """Fingerprint of :attr:`files` (see :func:`config_fingerprint`)."""
        return config_fingerprint(self.files)


def build_bundle(
    role: Role,
    deployment: Deployment,
    tag: str,
    versioned: Mapping[str, bytes],
    memory_bytes: int | None = None,
) -> Bundle:
    """Assemble the bundle of a role (pure).

    Parameters
    ----------
    role : Role
        The machine.
    deployment : Deployment
        Validated configuration.
    tag : str
        Image tag to run.
    versioned : Mapping[str, bytes]
        Content of :func:`static_files`, read by :func:`read_files`.
    memory_bytes : int | None
        Memory of the db machine (db only).
    """
    files = {**versioned, **generated_files(role, deployment)}
    env = render_env(role_env(role, deployment, tag, memory_bytes))
    return Bundle(role=role, tag=tag, files=files, env=env)


def pack(bundle: Bundle) -> str:
    """Pack a bundle into a base64-encoded ``.tar.gz`` (pure and reproducible).

    Files get mode ``644``, the ``.env`` mode ``600``; dates and owners are zeroed, so the
    same bundle always gives the same archive.
    """
    # (path in /opt/finkfat, content, mode). Sorted, so the archive does not depend on the
    # order the files were collected in.
    entries = [(name, content, 0o644) for name, content in sorted(bundle.files.items())]
    entries.append((ENV_FILE, bundle.env.encode(), 0o600))  # secrets: owner only
    # The archive is built in memory: nothing is written on the workstation, so the
    # secrets of the .env never land in a local file.
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, content, mode in entries:
            info = tarfile.TarInfo(name)  # "config/stream.yaml" creates config/ on unpacking
            info.size, info.mode, info.mtime = len(content), mode, 0
            archive.addfile(info, io.BytesIO(content))
    # base64: the archive is binary, but a Command's stdin is text. The machine decodes it
    # with `base64 -d` before `tar` (see upload_command).
    return base64.b64encode(buffer.getvalue()).decode()


def deploy_record(bundle: Bundle, deployed_at: datetime) -> str:
    """Write the ``deployed.json`` of a successful deployment (pure; no secret in it)."""
    record = {
        "role": str(bundle.role),
        "tag": bundle.tag,
        "config": bundle.fingerprint,
        "deployed_at": deployed_at.isoformat(timespec="seconds"),
    }
    return json.dumps(record, indent=2) + "\n"


@dataclass(frozen=True)
class DeployRecord:
    """Parsed ``deployed.json``: what runs on a machine.

    Attributes
    ----------
    tag : str
        Image tag.
    config : str
        Configuration fingerprint.
    deployed_at : str
        ISO date of the deployment.
    """

    tag: str
    config: str
    deployed_at: str


def parse_deploy_record(text: str) -> DeployRecord | None:
    """Read a ``deployed.json`` (pure); ``None`` when absent or unreadable."""
    try:
        data = json.loads(text)
        return DeployRecord(str(data["tag"]), str(data["config"]), str(data["deployed_at"]))
    except (ValueError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------- commands


def _remote(config: Path, role: Role, script: str, *, timeout_s: float | None = 120.0) -> Command:
    """Run a shell script on a machine (``ssh`` hands it to the remote shell).

    The script is ONE argument of the local ``ssh`` (no local shell interprets it); ``ssh``
    sends it as is to the remote user's shell, which interprets ``&&``, ``|``, ``$(…)``…
    """
    return ssh_command(config, role, (script,), timeout_s=timeout_s)


def compose_script(*args: str) -> str:
    """Return the remote shell line running ``docker compose <args>`` in :data:`REMOTE_DIR`.

    ``cd`` first: ``docker compose`` finds ``compose.yml`` and reads ``.env`` in the current
    directory. ``shlex.join`` quotes the arguments for the remote shell.
    """
    return f"cd {REMOTE_DIR} && docker compose {shlex.join(args)}"


def compose_command(
    config: Path, role: Role, *args: str, timeout_s: float | None = 120.0
) -> Command:
    """Build ``ssh … 'cd /opt/finkfat && docker compose <args>'`` (pure)."""
    return _remote(config, role, compose_script(*args), timeout_s=timeout_s)


def logs_command(
    config: Path, role: Role, service: str | None, *, follow: bool, tail: int
) -> Command:
    """Show the logs of a role's services (``docker compose logs``), streamed (pure).

    Parameters
    ----------
    config : Path
        Generated ``ssh_config``.
    role : Role
        The machine.
    service : str | None
        One service (e.g. ``caddy``); ``None`` for all of them.
    follow : bool
        Keep streaming new lines until Ctrl-C.
    tail : int
        Number of past lines per service.
    """
    args = [
        "logs",
        "--tail",
        str(tail),
        *(["--follow"] if follow else []),
        *([service] if service else []),
    ]
    command = compose_command(config, role, *args, timeout_s=None)
    return replace(command, interactive=True)


def memory_command(config: Path, role: Role) -> Command:
    """Read ``/proc/meminfo`` on a machine (pure)."""
    return _remote(config, role, "cat /proc/meminfo")


def prepare_command(config: Path, role: Role, infra: InfraConfig) -> Command:
    """Create :data:`REMOTE_DIR`, owned by the SSH user, and set the data owner (pure).

    On the compute, the data volume is given to the engine image's user
    (:data:`ENGINE_UID`). It is done here rather than at first boot: changing a VM's
    first-boot script makes Heat replace the VM.
    """
    user = shlex.quote(infra.ssh_user)
    # /opt belongs to root: sudo creates the directory and hands it to the SSH user, so
    # every later step (unpacking, docker compose) runs without sudo. `install -d` does
    # nothing if the directory already exists with these rights.
    steps = [f"sudo install -d -o {user} -g {user} -m 755 {REMOTE_DIR}"]
    volume = infra.roles.of(role).volume
    if role == Role.COMPUTE and volume is not None:
        # The stream runs as UID 10001 in its container and writes in /srv/finkfat
        # (mounted on /data): it must own the directory. Idempotent.
        steps.append(f"sudo chown {ENGINE_UID}:{ENGINE_UID} {shlex.quote(volume.mount)}")
    return _remote(config, role, " && ".join(steps))


def upload_command(config: Path, role: Role, archive: str) -> Command:
    """Unpack a :func:`pack` archive, received on stdin, into :data:`REMOTE_DIR` (pure).

    On the machine: ``base64 -d`` decodes stdin, ``tar -x`` unpacks it in place
    (``-z`` gunzip, ``-p`` keep the 600 of ``.env``, ``--no-same-owner`` files owned by the
    SSH user). Existing files are overwritten; nothing else in the directory is touched.
    """
    script = f"cd {REMOTE_DIR} && base64 -d | tar -xzpf - --no-same-owner"
    # stdin, not argv: the archive holds the .env, and a command line is visible to every
    # user of the machine (ps) and in --dry-run output.
    return replace(_remote(config, role, script), stdin=archive)


def record_command(config: Path, role: Role, record: str) -> Command:
    """Write ``deployed.json`` from stdin (pure)."""
    return replace(_remote(config, role, f"cat > {REMOTE_DIR}/{RECORD_FILE}"), stdin=record)


def login_command(config: Path, role: Role, registry: str, token: str) -> Command:
    """``docker login`` to the registry with a read-only token on stdin (pure)."""
    host = registry.split("/", 1)[0]  # ghcr.io/fusroman → ghcr.io
    user = images.registry_owner(registry)  # GHCR ignores the user name with a token
    script = f"docker login {shlex.quote(host)} -u {shlex.quote(user)} --password-stdin"
    return replace(_remote(config, role, script), stdin=token)


def health_checks(role: Role) -> list[RemoteCheck]:
    """List the remote checks deciding whether a role's deployment succeeded (pure).

    db: Postgres accepts connections (waited for up to two minutes). compute: the stream is
    still running, without restart, 30 seconds after the start. explorer: Caddy reaches the
    explorer (waited for up to two minutes, the first start loads the ephemeris).
    """
    # Each check is one remote shell script; it passes when it exits with status 0
    # (ssh.evaluate_check). Its fix, shown on failure, is the role's logs.
    logs = f"ff-ops logs {role}"
    if role == Role.DB:
        # pg_isready, run inside the postgres container (`exec -T`: no terminal), exits 0
        # once Postgres accepts connections. Retried every 2 s, 60 times: the first start
        # runs initdb on the empty volume, which takes a while.
        ready = f"pg_isready -q -U {POSTGRES_USER} -d {POSTGRES_DB}"
        wait = (
            f"for _ in $(seq 60); do {compose_script('exec', '-T', 'postgres')} {ready} "
            "&& exit 0; sleep 2; done; exit 1"
        )
        return [RemoteCheck("postgres accepts connections", (wait,), logs, timeout_s=180.0)]
    if role == Role.COMPUTE:
        # The stream has no health endpoint: a crash shows as a restart (restart:
        # unless-stopped). After 30 s, the container must be running AND never restarted.
        # `docker compose ps -q stream` gives the container ID, `docker inspect` prints
        # e.g. "true/0" (running / restart count), compared to the expected text.
        stable = (
            f'sleep 30 && id=$({compose_script("ps", "-q", "stream")}) && test -n "$id" && '
            'test "$(docker inspect -f \'{{.State.Running}}/{{.RestartCount}}\' "$id")" = true/0'
        )
        return [
            RemoteCheck("stream running, no restart after 30 s", (stable,), logs, timeout_s=90.0)
        ]
    # Explorer: the app is not published on the VM, only Caddy reaches it on the compose
    # network. So the request is sent from inside the caddy container (its image has
    # wget), exactly the path a real visitor's request takes. Retried every 3 s, 40 times.
    probe = f"{compose_script('exec', '-T', 'caddy')} wget -q -O /dev/null http://app:8080/"
    wait = f"for _ in $(seq 40); do {probe} && exit 0; sleep 3; done; exit 1"
    return [RemoteCheck("explorer answers behind Caddy", (wait,), logs, timeout_s=180.0)]


def public_check_command(domain: str) -> Command:
    """Fetch the explorer's home page over HTTPS from this machine (pure).

    It also checks the certificate. Retries leave Caddy time to obtain it.
    """
    return Command.of(
        "curl",
        "--silent",
        "--show-error",
        "--fail",
        "--output",
        "/dev/null",
        "--retry",
        "12",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        f"https://{domain}/",
        timeout_s=180.0,
    )


# --------------------------------------------------------------------------------- I/O


@dataclass(frozen=True)
class DeployContext:
    """What :func:`deploy_role` needs.

    Attributes
    ----------
    deployment : Deployment
        Validated configuration.
    paths : DeployPaths
        File locations.
    ssh_config : Path
        Generated ``ssh_config``.
    runner : Runner
        Runs the commands.
    tag : str
        Image tag to deploy.
    now : Callable[[], datetime]
        Clock of the deployment record.
    echo : Callable[[str], None]
        Progress messages.
    dry_run : bool
        Nothing is really run: the db memory is assumed (:data:`DRY_RUN_MEMORY_BYTES`).
    """

    deployment: Deployment
    paths: DeployPaths
    ssh_config: Path
    runner: Runner
    tag: str
    now: Callable[[], datetime]
    echo: Callable[[str], None] = print
    dry_run: bool = False


def _run_check(runner: Runner, command: Command) -> CommandResult:
    """Run a check, turning a timeout or an SSH failure into a failed result."""
    try:
        return runner.run(command, check=False)
    except CommandError as error:
        return CommandResult(returncode=1, stderr=str(error).splitlines()[0])


def _memory_bytes(context: DeployContext, role: Role) -> int | None:
    """Read the db machine's memory (``None`` for the other roles)."""
    if role != Role.DB:
        return None
    result = context.runner.run(memory_command(context.ssh_config, role))
    # In dry-run mode the command is only printed and its output is empty: an assumed
    # size keeps the rest of the sequence printable.
    return DRY_RUN_MEMORY_BYTES if context.dry_run else parse_meminfo(result.stdout)


def deploy_role(context: DeployContext, role: Role) -> list[Finding]:
    """Deploy one role: send its bundle, pull and start its services, check them.

    The deployment record is written only when every check passes, so ``ff-ops status``
    never reports a failed deployment as running.

    Parameters
    ----------
    context : DeployContext
        Configuration, runner and options.
    role : Role
        The machine.

    Returns
    -------
    list[Finding]
        One finding per health check (and the public HTTPS check of the explorer).

    Raises
    ------
    CommandError
        A deployment step (not a check) failed.
    ValueError
        A deployment file is missing, or the db memory cannot be read.
    """
    # Steps 1 to 7 use runner.run(...) with its default check=True: a failure raises
    # CommandError, which stops the whole `ff-ops deploy` (cli._fail_nicely prints it).
    # Steps 8 and 9 never raise: a failed check becomes a finding.
    deployment, config, runner = context.deployment, context.ssh_config, context.runner

    # 1. Read the versioned files first: a missing file fails before touching the machine.
    versioned = read_files(static_files(role, context.paths))

    # 2. /opt/finkfat exists and belongs to the SSH user; compute data owned by UID 10001.
    runner.run(prepare_command(config, role, deployment.infra))

    # 3. Postgres is sized from the db machine's memory (None for the other roles).
    memory = _memory_bytes(context, role)

    # 4. Everything the role needs, assembled in memory: files + generated Caddyfile + .env.
    bundle = build_bundle(role, deployment, context.tag, versioned, memory)
    context.echo(f"  bundle: {', '.join(sorted(bundle.files))}, .env (config {bundle.fingerprint})")

    # 5. One ssh call sends the whole bundle on stdin and unpacks it in /opt/finkfat.
    runner.run(upload_command(config, role, pack(bundle)))

    # 6. Private images only. The db runs the public postgres image of Docker Hub: no login.
    token = deployment.secrets.GHCR_READ_TOKEN
    if token is not None and role != Role.DB:
        runner.run(
            login_command(config, role, deployment.infra.images.registry, token.get_secret_value())
        )

    # 7. pull downloads the images named in compose.yml (tag from .env); up -d then
    #    (re)creates only the containers whose image or configuration changed, so a
    #    redeployment with nothing new restarts nothing. --remove-orphans drops containers
    #    of services removed from compose.yml.
    context.echo("  docker compose pull")
    runner.run(compose_command(config, role, "pull", "--quiet", timeout_s=PULL_TIMEOUT_S))
    context.echo("  docker compose up -d")
    runner.run(
        compose_command(config, role, "up", "-d", "--remove-orphans", timeout_s=UP_TIMEOUT_S)
    )

    # 8. Health checks, on the machine. check=False (inside _run_check): a failure is a
    #    result to evaluate, not an exception.
    findings = []
    for check in health_checks(role):
        command = ssh_command(config, role, check.command, timeout_s=check.timeout_s)
        findings.append(evaluate_check(check, _run_check(runner, command)))
    # Explorer only, and only if Caddy reaches the app: the site seen from outside
    # (DNS, ports 80/443, Let's Encrypt certificate), as a visitor would.
    domain = deployment.local.explorer_domain
    if role == Role.EXPLORER and all(f.level == Level.OK for f in findings):
        findings.append(_public_check(runner, domain))

    # 9. The record says "this tag and this configuration run here": written only if no
    #    check failed (warnings are accepted). ff-ops status compares it with deploy/.
    if all(f.level != Level.ERROR for f in findings):
        runner.run(record_command(config, role, deploy_record(bundle, context.now())))
    return findings


def _public_check(runner: Runner, domain: str | None) -> Finding:
    """Check the explorer from this machine over HTTPS (needs a domain)."""
    if not domain:
        return Finding(
            Level.WARN,
            "no explorer_domain: the explorer is served over plain HTTP, without a certificate",
            "set explorer_domain in infra.local.yaml, then ff-ops deploy explorer",
        )
    result = _run_check(runner, public_check_command(domain))
    if result.ok:
        return Finding(Level.OK, f"https://{domain}/ answers with a valid certificate")
    detail = (result.stderr or result.stdout).strip().splitlines()
    return Finding(
        Level.ERROR,
        f"https://{domain}/ unreachable ({detail[-1] if detail else result.returncode})",
        "check the DNS record of the domain → floating IP, then ff-ops logs explorer caddy",
    )


def ordered_roles(infra: InfraConfig, wanted: Sequence[Role] | None) -> list[Role]:
    """Sort the roles to deploy in :data:`DEPLOYMENT_ORDER`, keeping enabled ones (pure).

    Raises
    ------
    ValueError
        A requested role is disabled in ``infra.yaml``.
    """
    enabled = [role for role in DEPLOYMENT_ORDER if infra.roles.of(role).enabled]
    if not wanted:
        return enabled
    disabled = sorted({str(role) for role in wanted} - {str(role) for role in enabled})
    if disabled:
        raise ValueError(f"disabled in infra.yaml: {', '.join(disabled)}")
    return [role for role in enabled if role in wanted]
