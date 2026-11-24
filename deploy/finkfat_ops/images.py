"""Docker images of fink-fat on GHCR: names, published tags, and local builds.

Images are normally built and published by CI (``.github/workflows/images.yml``) once the
CI is green. ff-ops lists what was published (through the ``gh`` CLI) and, for
troubleshooting only, builds and pushes from the operator's machine.

All functions are pure: they build :class:`~finkfat_ops.runner.Command` values or
interpret outputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from finkfat_ops.runner import Command, CommandResult

VERSIONS_PER_PAGE = 100
"""Most recent package versions fetched by :func:`versions_command`."""

SHORT_SHA_LENGTH = 7
"""Length of the commit hash in ``sha-<hash>`` tags (same as CI)."""


class ImageTarget(StrEnum):
    """Published images, named after their stage in ``dockerfile``."""

    ENGINE = "engine"
    EXPLORER = "explorer"


def package_name(target: ImageTarget) -> str:
    """Return the GHCR package name, e.g. ``fink-fat-engine``."""
    return f"fink-fat-{target}"


def image_name(registry: str, target: ImageTarget) -> str:
    """Return the full image name, e.g. ``ghcr.io/fusroman/fink-fat-engine``.

    Parameters
    ----------
    registry : str
        Registry and owner from ``infra.yaml`` (``images.registry``).
    target : ImageTarget
        Which image.
    """
    return f"{registry.rstrip('/')}/{package_name(target)}"


def registry_owner(registry: str) -> str:
    """Return the owner part of ``ghcr.io/<owner>`` (lowercase, as GHCR stores it).

    Raises
    ------
    ValueError
        ``registry`` is not of the form ``ghcr.io/<owner>``.
    """
    host, _, owner = registry.strip("/").partition("/")
    if host != "ghcr.io" or not owner or "/" in owner:
        raise ValueError(f"images.registry must be ghcr.io/<owner>, got '{registry}'")
    return owner.lower()


def short_sha_tag(sha: str) -> str:
    """Return the ``sha-<7 characters>`` tag of a commit, as CI tags its images."""
    return f"sha-{sha.strip()[:SHORT_SHA_LENGTH]}"


# ------------------------------------------------------------------------ published tags


@dataclass(frozen=True)
class ImageVersion:
    """One published version (manifest) of an image.

    Attributes
    ----------
    tags : tuple[str, ...]
        Its tags, e.g. ``("sha-abc1234", "main")``.
    created_at : str
        Publication time (ISO 8601), as GHCR reports it.
    digest : str
        Manifest digest, ``sha256:…``.
    """

    tags: tuple[str, ...]
    created_at: str
    digest: str


def versions_command(owner: str, target: ImageTarget) -> Command:
    """``gh api`` listing the most recent versions of a user-owned GHCR package.

    Requires ``gh`` logged in with the ``read:packages`` scope.
    """
    path = (
        f"/users/{owner}/packages/container/{package_name(target)}/versions"
        f"?per_page={VERSIONS_PER_PAGE}"
    )
    return Command.of("gh", "api", path)


def is_missing_package(result: CommandResult) -> bool:
    """Tell whether ``gh api`` answered that the package does not exist (HTTP 404).

    GitHub also answers 404 for a private package the token is not allowed to read.
    """
    return not result.ok and "HTTP 404" in f"{result.stderr}\n{result.stdout}"


def parse_versions(data: Any) -> list[ImageVersion]:
    """Build :class:`ImageVersion` values from the GitHub packages API.

    Parameters
    ----------
    data : Any
        Decoded JSON list of package versions; ``None`` (dry-run) gives an empty list.

    Returns
    -------
    list[ImageVersion]
        Tagged versions only (untagged ones are intermediate layers of multi-platform
        builds or overwritten tags), newest first.
    """
    versions = []
    for item in data or []:
        tags = tuple((item.get("metadata") or {}).get("container", {}).get("tags") or ())
        if tags:
            versions.append(ImageVersion(tags, item.get("created_at") or "", item.get("name", "")))
    return sorted(versions, key=lambda version: version.created_at, reverse=True)


# --------------------------------------------------------------------------- local build


def head_command(repo: Path) -> Command:
    """``git rev-parse HEAD`` in the repository."""
    return Command.of("git", "-C", str(repo), "rev-parse", "HEAD")


def dirty_command(repo: Path) -> Command:
    """``git status --porcelain``: empty output means a clean working tree."""
    return Command.of("git", "-C", str(repo), "status", "--porcelain")


def build_command(repo: Path, target: ImageTarget, refs: Sequence[str], *, push: bool) -> Command:
    """``docker buildx build`` of one stage of ``dockerfile``, tagged with ``refs``.

    Parameters
    ----------
    repo : Path
        Repository root (the build context, with its ``.dockerignore``).
    target : ImageTarget
        Stage to build.
    refs : Sequence[str]
        Full image references to tag, e.g. ``ghcr.io/fusroman/fink-fat-engine:sha-abc1234``.
    push : bool
        Push to the registry; otherwise load the image into the local Docker.

    Returns
    -------
    Command
        An interactive command (its build log is shown live), with no timeout.
    """
    tags = [arg for ref in refs for arg in ("--tag", ref)]
    argv = (
        "docker",
        "buildx",
        "build",
        "--file",
        str(repo / "dockerfile"),
        "--target",
        str(target),
        *tags,
        "--push" if push else "--load",
        str(repo),
    )
    return Command(argv=argv, timeout_s=None, interactive=True)
