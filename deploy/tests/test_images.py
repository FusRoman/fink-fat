"""Tests of finkfat_ops.images (names, published tags, local build commands)."""

from __future__ import annotations

from pathlib import Path

import pytest

from finkfat_ops.images import (
    ImageTarget,
    build_command,
    image_name,
    parse_versions,
    registry_owner,
    short_sha_tag,
    versions_command,
)


def test_names_follow_the_ci_convention() -> None:
    assert image_name("ghcr.io/fusroman", ImageTarget.ENGINE) == "ghcr.io/fusroman/fink-fat-engine"
    assert registry_owner("ghcr.io/FusRoman/") == "fusroman"
    assert short_sha_tag("abc1234def5678\n") == "sha-abc1234"


@pytest.mark.parametrize("registry", ["docker.io/fusroman", "ghcr.io", "ghcr.io/a/b"])
def test_registry_must_be_a_ghcr_owner(registry: str) -> None:
    with pytest.raises(ValueError, match=r"ghcr.io/<owner>"):
        registry_owner(registry)


def test_versions_command_targets_the_user_package() -> None:
    argv = versions_command("fusroman", ImageTarget.EXPLORER).argv
    assert argv[:2] == ("gh", "api")
    assert argv[2].startswith("/users/fusroman/packages/container/fink-fat-explorer/versions")


def test_versions_parsing_keeps_tagged_ones_newest_first() -> None:
    data = [
        {"name": "sha256:aa", "created_at": "2026-11-20T10:00:00Z",
         "metadata": {"container": {"tags": ["sha-1111111"]}}},
        {"name": "sha256:bb", "created_at": "2026-11-24T10:00:00Z",
         "metadata": {"container": {"tags": ["sha-2222222", "main"]}}},
        {"name": "sha256:cc", "created_at": "2026-11-24T10:00:01Z",
         "metadata": {"container": {"tags": []}}},
    ]  # fmt: skip
    versions = parse_versions(data)
    assert [v.tags for v in versions] == [("sha-2222222", "main"), ("sha-1111111",)]
    assert versions[0].digest == "sha256:bb"
    assert parse_versions(None) == []


def test_build_command_pushes_or_loads_one_stage() -> None:
    repo = Path("/repo")
    ref = "ghcr.io/fusroman/fink-fat-engine:sha-abc1234"
    command = build_command(repo, ImageTarget.ENGINE, [ref], push=True)
    assert command.argv == (
        "docker", "buildx", "build", "--file", "/repo/dockerfile", "--target", "engine",
        "--tag", ref, "--push", "/repo",
    )  # fmt: skip
    assert command.interactive and command.timeout_s is None
    assert "--load" in build_command(repo, ImageTarget.ENGINE, [ref], push=False).argv
