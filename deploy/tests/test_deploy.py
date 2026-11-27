"""Tests of finkfat_ops.deploy (bundles, .env, commands, deployment sequence)."""

from __future__ import annotations

import base64
import io
import re
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from finkfat_ops.checks import Level
from finkfat_ops.config import DeployPaths, load_deployment
from finkfat_ops.deploy import (
    DeployContext,
    build_bundle,
    config_fingerprint,
    database_url,
    deploy_record,
    deploy_role,
    file_findings,
    health_checks,
    logs_command,
    missing_files,
    ordered_roles,
    pack,
    parse_deploy_record,
    parse_meminfo,
    postgres_settings,
    read_files,
    render_caddyfile,
    render_env,
    role_env,
    static_files,
)
from finkfat_ops.naming import Role
from finkfat_ops.runner import CommandResult, FakeRunner

SECRETS = ("pg-password-0123456789", "pgadmin-password-0123")
CONFIG = Path("/d/.state/ssh_config")
NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
MEMINFO = "MemTotal:       32763204 kB\nMemFree:        1000 kB\n"


def _deployment(deploy_dir: Path):
    return load_deployment(DeployPaths(deploy_dir))


def _unpack(archive: str) -> dict[str, tuple[bytes, int]]:
    data = io.BytesIO(base64.b64decode(archive))
    with tarfile.open(fileobj=data, mode="r:gz") as tar:
        return {m.name: (tar.extractfile(m).read(), m.mode) for m in tar.getmembers()}  # type: ignore[union-attr]


def _bundle(deploy_dir: Path, role: Role):
    deployment = _deployment(deploy_dir)
    versioned = read_files(static_files(role, DeployPaths(deploy_dir)))
    memory = 32 * 1024**3 if role == Role.DB else None
    return build_bundle(role, deployment, "v1.0.0", versioned, memory)


# ------------------------------------------------------------------------------- files


def test_static_files_per_role(deploy_dir: Path) -> None:
    paths = DeployPaths(deploy_dir)
    assert set(static_files(Role.DB, paths)) == {"compose.yml"}
    assert set(static_files(Role.COMPUTE, paths)) == {
        "compose.yml",
        "config/engine_config.yaml",
        "config/stream.yaml",
    }
    assert set(static_files(Role.EXPLORER, paths)) == {"compose.yml", "config/engine_config.yaml"}
    assert static_files(Role.DB, paths)["compose.yml"] == deploy_dir / "compose" / "db.yml"


def test_missing_file_is_named(deploy_dir: Path) -> None:
    (deploy_dir / "compose" / "db.yml").unlink()
    with pytest.raises(ValueError, match=r"db\.yml$"):
        read_files(static_files(Role.DB, DeployPaths(deploy_dir)))


def test_caddyfile_with_and_without_domain() -> None:
    assert "fink-fat.example.org {" in render_caddyfile("fink-fat.example.org")
    assert "reverse_proxy app:8080" in render_caddyfile("fink-fat.example.org")
    assert ":80 {" in render_caddyfile(None)


def test_fingerprint_depends_on_names_and_contents() -> None:
    base = config_fingerprint({"a": b"1", "b": b"2"})
    assert base == config_fingerprint({"b": b"2", "a": b"1"})
    assert base != config_fingerprint({"a": b"1", "b": b"3"})
    assert base != config_fingerprint({"a": b"1", "c": b"2"})
    assert len(base) == 12


# ------------------------------------------------------------------------- environment


def test_meminfo_and_postgres_settings() -> None:
    memory = parse_meminfo(MEMINFO)
    assert memory == 32763204 * 1024
    settings = postgres_settings(memory)
    assert settings["PG_SHARED_BUFFERS"] == "7998MB"
    assert settings["PG_EFFECTIVE_CACHE_SIZE"] == "23996MB"
    assert settings["PG_MAINTENANCE_WORK_MEM"] == "1599MB"
    assert postgres_settings(128 * 1024**3)["PG_MAINTENANCE_WORK_MEM"] == "2048MB"
    with pytest.raises(ValueError, match="MemTotal"):
        parse_meminfo("")


def test_every_compose_variable_is_provided(deploy_dir: Path) -> None:
    deployment = _deployment(deploy_dir)
    for role in Role:
        text = (deploy_dir / "compose" / f"{role}.yml").read_text(encoding="utf-8")
        compose = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
        used = set(re.findall(r"(?<!\$)\$\{([A-Z_]+)\}", compose))
        provided = set(role_env(role, deployment, "v1.0.0", 32 * 1024**3))
        assert used <= provided, f"{role}: {used - provided} missing from its .env"


def test_each_role_only_gets_its_secrets(deploy_dir: Path) -> None:
    deployment = _deployment(deploy_dir)
    db = role_env(Role.DB, deployment, "v1.0.0", 32 * 1024**3)
    compute = role_env(Role.COMPUTE, deployment, "v1.0.0")
    explorer = role_env(Role.EXPLORER, deployment, "v1.0.0")
    assert SECRETS[1] not in str(db) and SECRETS[1] not in str(compute)
    assert "PGADMIN_PASSWORD" in explorer
    assert "FINK_FAT_KAFKA_PASSWORD" in compute and "FINK_FAT_KAFKA_PASSWORD" not in explorer
    assert compute["FFAT_TAG"] == explorer["FFAT_TAG"] == "v1.0.0"
    assert db["DB_BIND_IP"] == str(deployment.infra.roles.db.fixed_ip)
    with pytest.raises(ValueError, match="memory"):
        role_env(Role.DB, deployment, "v1.0.0")


def test_database_url_encodes_the_password(deploy_dir: Path) -> None:
    deployment = _deployment(deploy_dir)
    secrets = deployment.secrets.model_copy(
        update={"POSTGRES_PASSWORD": type(deployment.secrets.POSTGRES_PASSWORD)("a/b@c:d%e+f=g")}
    )
    url = database_url(deployment.infra, secrets)
    assert url == "postgres://finkfat:a%2Fb%40c%3Ad%25e%2Bf%3Dg@10.42.0.20:5432/finkfat"


def test_env_values_are_literal() -> None:
    assert render_env({"A": "x$y"}).endswith("A='x$y'\n")
    with pytest.raises(ValueError, match="A holds") as error:
        render_env({"A": "it's"})
    assert "it's" not in str(error.value)


# ----------------------------------------------------------------------------- bundle


def test_archive_holds_the_files_and_a_private_env(deploy_dir: Path) -> None:
    bundle = _bundle(deploy_dir, Role.EXPLORER)
    files = _unpack(pack(bundle))
    assert set(files) == {"Caddyfile", "compose.yml", "config/engine_config.yaml", ".env"}
    assert files[".env"][1] == 0o600
    assert files["compose.yml"][1] == 0o644
    assert SECRETS[1].encode() in files[".env"][0]
    assert pack(bundle) == pack(bundle)  # reproducible


def test_no_secret_outside_the_env(deploy_dir: Path) -> None:
    for role in Role:
        bundle = _bundle(deploy_dir, role)
        record = deploy_record(bundle, NOW)
        for secret in SECRETS:
            assert all(secret.encode() not in content for content in bundle.files.values())
            assert secret not in record


def test_record_round_trip(deploy_dir: Path) -> None:
    bundle = _bundle(deploy_dir, Role.COMPUTE)
    record = parse_deploy_record(deploy_record(bundle, NOW))
    assert record is not None
    assert (record.tag, record.config) == ("v1.0.0", bundle.fingerprint)
    assert record.deployed_at == "2026-01-02T03:04:05+00:00"
    assert parse_deploy_record("") is None
    assert parse_deploy_record('{"tag": "x"}') is None


def test_production_engine_config_stores_on_the_data_volume() -> None:
    config = Path(__file__).resolve().parents[1] / "config/engine_config.yaml"
    engine = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert engine["storage_path"] == "/data/storage"  # /data = /srv/finkfat, the data volume


def test_stream_template_reads_the_production_engine_config() -> None:
    template = Path(__file__).resolve().parents[1] / "config/stream.example.yaml"
    stream = yaml.safe_load(template.read_text(encoding="utf-8"))
    assert stream["engine_config"] == "/config/engine_config.yaml"
    assert "url" not in (stream.get("database") or {})  # comes from $FINK_FAT_DATABASE_URL
    # Placeholders only: the real broker and consumer group live in the gitignored stream.yaml.
    assert stream["source"]["bootstrap_servers"].startswith("kafka.example.org")
    assert stream["source"]["group_id"] == "my-finkfat-group"


def test_missing_stream_config_says_how_to_create_it(deploy_dir: Path) -> None:
    (deploy_dir / "config" / "stream.yaml").unlink()
    with pytest.raises(ValueError, match=r"copy stream\.example\.yaml and fill it in"):
        read_files(static_files(Role.COMPUTE, DeployPaths(deploy_dir)))


# --------------------------------------------------------------------------- sequence


def test_roles_are_ordered_and_must_be_enabled(deploy_dir: Path) -> None:
    infra = _deployment(deploy_dir).infra
    assert ordered_roles(infra, None) == [Role.DB, Role.COMPUTE, Role.EXPLORER]
    assert ordered_roles(infra, [Role.EXPLORER, Role.DB]) == [Role.DB, Role.EXPLORER]
    disabled = infra.model_copy(
        update={
            "roles": infra.roles.model_copy(
                update={"compute": infra.roles.compute.model_copy(update={"enabled": False})}
            )
        }
    )
    assert ordered_roles(disabled, None) == [Role.DB, Role.EXPLORER]
    with pytest.raises(ValueError, match="compute"):
        ordered_roles(disabled, [Role.COMPUTE])


def _context(deploy_dir: Path, runner: FakeRunner) -> DeployContext:
    return DeployContext(
        deployment=_deployment(deploy_dir),
        paths=DeployPaths(deploy_dir),
        ssh_config=CONFIG,
        runner=runner,
        tag="v1.0.0",
        now=lambda: NOW,
        echo=lambda _: None,
    )


SSH = ["ssh", "-F", str(CONFIG), "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]


def test_db_deployment_sequence(deploy_dir: Path) -> None:
    runner = (
        FakeRunner()
        .on([*SSH, "ffat-db", "cat /proc/meminfo"], CommandResult(stdout=MEMINFO))
        .on([*SSH, "ffat-db"], CommandResult())
    )
    findings = deploy_role(_context(deploy_dir, runner), Role.DB)
    assert [f.level for f in findings] == [Level.OK]
    scripts = [call.argv[-1] for call in runner.calls]
    assert scripts[0].startswith("sudo install -d")
    assert "chown" not in scripts[0]
    assert scripts[1] == "cat /proc/meminfo"
    assert "tar -xzpf" in scripts[2]
    assert "docker login" not in " ".join(scripts)  # public images: no token
    assert scripts[3].endswith("docker compose pull --quiet")
    assert scripts[4].endswith("docker compose up -d --remove-orphans")
    assert "pg_isready" in scripts[5]
    assert scripts[6] == "cat > /opt/finkfat/deployed.json"
    upload = runner.calls[2]
    env = _unpack(upload.stdin or "")[".env"][0].decode()
    assert "PG_SHARED_BUFFERS='7998MB'" in env
    for call in runner.calls:  # secrets only ever travel on stdin
        assert all(secret not in " ".join(call.argv) for secret in SECRETS)


def test_compute_gets_its_data_volume_owner(deploy_dir: Path) -> None:
    runner = FakeRunner().on([*SSH, "ffat-compute"], CommandResult())
    deploy_role(_context(deploy_dir, runner), Role.COMPUTE)
    assert "sudo chown 10001:10001 /srv/finkfat" in runner.calls[0].argv[-1]


def test_failed_check_writes_no_record(deploy_dir: Path) -> None:
    (stable,) = health_checks(Role.COMPUTE)
    runner = (
        FakeRunner()
        .on([*SSH, "ffat-compute", *stable.command], CommandResult(returncode=1, stdout="restart"))
        .on([*SSH, "ffat-compute"], CommandResult())
    )
    findings = deploy_role(_context(deploy_dir, runner), Role.COMPUTE)
    assert findings[0].level == Level.ERROR
    assert findings[0].fix == "ff-ops logs compute"
    assert not any("deployed.json" in call.argv[-1] for call in runner.calls)


def test_explorer_is_checked_from_outside(deploy_dir: Path) -> None:
    runner = (
        FakeRunner()
        .on(["curl"], CommandResult(returncode=60, stderr="SSL certificate problem"))
        .on([*SSH, "ffat-explorer"], CommandResult())
    )
    findings = deploy_role(_context(deploy_dir, runner), Role.EXPLORER)
    assert [f.level for f in findings] == [Level.OK, Level.ERROR]
    assert "https://fink-fat.example.org/" in findings[1].message
    assert "SSL certificate problem" in findings[1].message


def test_private_images_log_in_with_the_token_on_stdin(deploy_dir: Path) -> None:
    with (deploy_dir / "secrets.env").open("a", encoding="utf-8") as secrets:
        secrets.write("GHCR_READ_TOKEN=ghp_readonly_token\n")
    runner = FakeRunner().on([*SSH, "ffat-compute"], CommandResult())
    deploy_role(_context(deploy_dir, runner), Role.COMPUTE)
    login = [c for c in runner.calls if "docker login" in c.argv[-1]]
    assert len(login) == 1
    assert login[0].argv[-1] == "docker login ghcr.io -u fusroman --password-stdin"
    assert login[0].stdin == "ghp_readonly_token"


def test_logs_command() -> None:
    command = logs_command(CONFIG, Role.EXPLORER, "caddy", follow=True, tail=50)
    assert command.interactive and command.timeout_s is None
    assert command.argv[-1] == "cd /opt/finkfat && docker compose logs --tail 50 --follow caddy"
    every = logs_command(CONFIG, Role.DB, None, follow=False, tail=10)
    assert every.argv[-1] == "cd /opt/finkfat && docker compose logs --tail 10"


def test_missing_files_of_several_roles(deploy_dir: Path) -> None:
    paths = DeployPaths(deploy_dir)
    assert missing_files(list(Role), paths) == []
    (deploy_dir / "config" / "stream.yaml").unlink()
    (deploy_dir / "config" / "engine_config.yaml").unlink()  # shared: reported once
    assert missing_files(list(Role), paths) == [
        deploy_dir / "config" / "engine_config.yaml",
        deploy_dir / "config" / "stream.yaml",
    ]
    assert missing_files([Role.DB], paths) == []


def test_file_findings_say_how_to_create_each_file(deploy_dir: Path) -> None:
    paths = DeployPaths(deploy_dir)
    assert [f.level for f in file_findings(list(Role), paths)] == [Level.OK]
    stream = deploy_dir / "config" / "stream.yaml"
    stream.unlink()
    (deploy_dir / "compose" / "db.yml").unlink()
    db, local = file_findings(list(Role), paths)
    assert db.message == "compose/db.yml is missing"
    assert db.fix == f"git checkout -- {deploy_dir / 'compose' / 'db.yml'}"
    assert local.message == "config/stream.yaml is missing"
    template = deploy_dir / "config" / "stream.example.yaml"
    assert local.fix == f"cp {template} {stream} && chmod 600 {stream}, then fill it in"
