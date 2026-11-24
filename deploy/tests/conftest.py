"""Shared fixtures: valid configuration data and a temporary deploy/ directory."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

DEPLOY_DIR = Path(__file__).resolve().parents[1]

VALID_LOCAL: dict[str, Any] = {
    "openstack": {"cloud": "test-cloud", "project": "test-project"},
    "admin_cidrs": ["192.0.2.0/24", "198.51.100.7/32"],
    "explorer_domain": "fink-fat.example.org",
}

VALID_SECRETS_TEXT = """\
# test secrets
POSTGRES_PASSWORD=pg-password-0123456789
PGADMIN_PASSWORD="pgadmin-password-0123"
FINK_FAT_KAFKA_PASSWORD=
"""


@pytest.fixture
def infra_data() -> dict[str, Any]:
    """The committed deploy/infra.yaml, parsed (so the real file is always tested)."""
    return yaml.safe_load((DEPLOY_DIR / "infra.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def local_data() -> dict[str, Any]:
    """A valid infra.local.yaml content."""
    return copy.deepcopy(VALID_LOCAL)


@pytest.fixture
def deploy_dir(tmp_path: Path, infra_data: dict[str, Any], local_data: dict[str, Any]) -> Path:
    """A temporary deploy/ directory holding three valid configuration files."""
    (tmp_path / "infra.yaml").write_text(yaml.safe_dump(infra_data), encoding="utf-8")
    (tmp_path / "infra.local.yaml").write_text(yaml.safe_dump(local_data), encoding="utf-8")
    secrets = tmp_path / "secrets.env"
    secrets.write_text(VALID_SECRETS_TEXT, encoding="utf-8")
    secrets.chmod(0o600)
    (tmp_path / "infra.local.yaml").chmod(0o600)
    return tmp_path
