# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validate resolver-selected binary artifacts before writing deployment locks."""

import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from dependency_profiles import DependencyProfile, architecture


def canonical_name(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(
        r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*", name
    ):
        raise ValueError("Invalid distribution name")
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class WheelArtifact:
    name: str
    version: str
    url: str
    sha256: str

    def record(self) -> dict:
        return asdict(self)


def wheel_artifact(record: dict, profile: DependencyProfile) -> WheelArtifact:
    if (
        not isinstance(record, dict)
        or set(record) != {"name", "version", "url", "sha256"}
        or not all(isinstance(value, str) for value in record.values())
    ):
        raise ValueError("Invalid artifact fields")
    name = canonical_name(record["name"])
    version, url, digest = record["version"], record["url"], record["sha256"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+!]*", version) or not re.fullmatch(
        r"[0-9a-f]{64}", digest
    ):
        raise ValueError("Invalid artifact version or SHA-256")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or any(ord(char) <= 32 or ord(char) == 127 for char in url)
        or "\\" in url
    ):
        raise ValueError("Artifact URL must be an uncredentialed HTTPS wheel URL")
    path = unquote(parsed.path)
    if (
        ".." in PurePosixPath(path).parts
        or "\\" in path
        or any(ord(char) <= 32 or ord(char) == 127 for char in path)
    ):
        raise ValueError("Invalid artifact URL path")
    allowed = parsed.hostname == "files.pythonhosted.org" and path.startswith(
        "/packages/"
    )
    if name == "torch" and profile.torch_index:
        index = urlsplit(profile.torch_index)
        # The official Torch index links to its Cloudflare R2 distribution host.
        allowed = parsed.hostname in {
            index.hostname,
            "download-r2.pytorch.org",
        } and path.startswith(index.path + "/")
    if not allowed:
        raise ValueError(f"Unapproved artifact origin for {name}")
    filename = PurePosixPath(path).name
    parts = filename.split("-")
    if (
        len(parts) not in (5, 6)
        or not filename.endswith(".whl")
        or canonical_name(parts[0]) != name
        or parts[1] != version
    ):
        raise ValueError("Wheel identity differs from resolver metadata")
    if profile.backend == "bootstrap" and not filename.endswith("-py3-none-any.whl"):
        raise ValueError("Bootstrap artifacts must be universal wheels")
    return WheelArtifact(name, version, url, digest)


def check_report_environment(environment: dict, profile: DependencyProfile) -> None:
    if not isinstance(environment, dict) or (
        environment.get("python_version") != profile.python
        or environment.get("implementation_name") != "cpython"
        or environment.get("sys_platform") != profile.sys_platform
    ):
        raise ValueError(
            f"Resolver environment must be {profile.system} CPython {profile.python}"
        )
    machine = environment.get("platform_machine")
    if not isinstance(machine, str) or (
        profile.architecture != "any" and architecture(machine) != profile.architecture
    ):
        raise ValueError("Resolver architecture differs from profile")


def report_artifacts(
    report: dict, profile: DependencyProfile
) -> tuple[dict, list[WheelArtifact]]:
    if (
        not isinstance(report, dict)
        or report.get("version") != "1"
        or report.get("pip_version") != "26.2.1"
    ):
        raise ValueError("Resolve with reviewed pip 26.2.1 report schema 1")
    environment = report.get("environment", {})
    check_report_environment(environment, profile)
    records = report.get("install", [])
    if not isinstance(records, list) or not records or len(records) > 200:
        raise ValueError("Missing or excessive resolver artifacts")
    artifacts = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid resolver artifact")
        if record.get("is_yanked") or record.get("is_editable"):
            raise ValueError("Yanked/editable artifacts are not deployable")
        metadata, download = record.get("metadata"), record.get("download_info")
        if not isinstance(metadata, dict) or not isinstance(download, dict):
            raise ValueError("Missing resolver metadata/download")
        archive = download.get("archive_info", {})
        hashes = archive.get("hashes", {}) if isinstance(archive, dict) else {}
        if not isinstance(hashes, dict):
            raise ValueError("Missing resolver artifact hash")
        artifacts.append(
            wheel_artifact(
                {
                    "name": metadata.get("name"),
                    "version": metadata.get("version"),
                    "url": download.get("url"),
                    "sha256": hashes.get("sha256", ""),
                },
                profile,
            )
        )
    if len({artifact.name for artifact in artifacts}) != len(artifacts):
        raise ValueError("Duplicate resolver distributions")
    if profile.torch_version and not any(
        artifact.name == "torch" and artifact.version == profile.torch_version
        for artifact in artifacts
    ):
        raise ValueError("Resolver substituted the backend Torch profile")
    return environment, sorted(artifacts, key=lambda artifact: artifact.name)
