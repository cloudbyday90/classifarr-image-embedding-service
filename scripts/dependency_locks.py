# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read/write complete wheel locks and verify their input and inventory contracts."""

import hashlib
import json
import os
from importlib.metadata import distributions
from pathlib import Path
from tempfile import NamedTemporaryFile

from dependency_artifacts import (
    WheelArtifact,
    canonical_name,
    check_report_environment,
    wheel_artifact,
)
from dependency_profiles import DependencyProfile


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def input_contract(root: Path, profile: DependencyProfile) -> dict[str, str]:
    # Stable across Windows/Linux Git checkout line-ending policies.
    return {
        name: sha256((root / name).read_bytes().replace(b"\r\n", b"\n"))
        for name in sorted(set((*profile.inputs, "requirements-bootstrap.txt")))
    }


def render_lock(profile: DependencyProfile, artifacts: list[WheelArtifact]) -> str:
    header = f"# Generated for {profile.name}; regenerate with scripts/lock_dependencies.py.\n"
    return header + "".join(
        f"{artifact.name} @ {artifact.url} --hash=sha256:{artifact.sha256}\n"
        for artifact in sorted(artifacts, key=lambda artifact: artifact.name)
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".lock-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_lock(
    root: Path,
    profile: DependencyProfile,
    environment: dict,
    artifacts: list[WheelArtifact],
) -> None:
    check_report_environment(environment, profile)
    artifacts = [wheel_artifact(artifact.record(), profile) for artifact in artifacts]
    if not artifacts or len({artifact.name for artifact in artifacts}) != len(
        artifacts
    ):
        raise ValueError("Locks require unique artifacts")
    content = render_lock(profile, artifacts).encode()
    manifest = {
        "schema_version": 1,
        "profile": profile.name,
        "python": "3.12",
        "architecture": profile.architecture,
        "backend": profile.backend,
        "resolver": "pip==26.2.1",
        "environment": environment,
        "inputs": input_contract(root, profile),
        "requirements_sha256": sha256(content),
        "artifacts": [artifact.record() for artifact in artifacts],
    }
    directory = root / "requirements/locks"
    _atomic_write(directory / f"{profile.name}.txt", content)
    # Publish the attestation last; interruption produces a rejected mismatched pair.
    _atomic_write(
        directory / f"{profile.name}.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )


def load_lock(
    root: Path, profile: DependencyProfile
) -> tuple[Path, list[WheelArtifact]]:
    directory = root / "requirements/locks"
    manifest_file, lock_file = (
        directory / f"{profile.name}.json",
        directory / f"{profile.name}.txt",
    )
    if (
        manifest_file.stat().st_size > 1024 * 1024
        or lock_file.stat().st_size > 1024 * 1024
    ):
        raise ValueError("Dependency lock is oversized")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "profile": profile.name,
        "python": "3.12",
        "architecture": profile.architecture,
        "backend": profile.backend,
        "resolver": "pip==26.2.1",
    }
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or any(manifest.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("Dependency lock profile contract differs")
    check_report_environment(manifest.get("environment", {}), profile)
    if manifest.get("inputs") != input_contract(root, profile):
        raise ValueError("Dependency inputs changed; regenerate affected locks")
    records = manifest.get("artifacts", [])
    if not isinstance(records, list) or not records or len(records) > 200:
        raise ValueError("Missing or excessive lock artifacts")
    artifacts = [wheel_artifact(record, profile) for record in records]
    if len({artifact.name for artifact in artifacts}) != len(artifacts):
        raise ValueError("Duplicate lock distributions")
    if not any(
        artifact.name == "pip" and artifact.version == "26.2.1"
        for artifact in artifacts
    ):
        raise ValueError("Lock must include reviewed pip 26.2.1")
    if profile.backend == "bootstrap" and len(artifacts) != 1:
        raise ValueError("Bootstrap must contain only pip")
    if profile.torch_version and not any(
        artifact.name == "torch" and artifact.version == profile.torch_version
        for artifact in artifacts
    ):
        raise ValueError("Lock substituted the backend Torch profile")
    content = lock_file.read_bytes().replace(b"\r\n", b"\n")
    if sha256(content) != manifest.get(
        "requirements_sha256"
    ) or content.decode() != render_lock(profile, artifacts):
        raise ValueError("Dependency lock text/attestation differs")
    return lock_file, artifacts


def verify_inventory(
    artifacts: list[WheelArtifact], site_packages: Path | None = None
) -> None:
    actual = {}
    installed = (
        distributions(path=[str(site_packages)]) if site_packages else distributions()
    )
    for distribution in installed:
        name = canonical_name(distribution.metadata["Name"])
        if name in actual:
            raise ValueError(f"Duplicate installed distribution: {name}")
        actual[name] = distribution.version
    expected = {artifact.name: artifact.version for artifact in artifacts}
    if actual != expected:
        missing, extra = (
            sorted(expected.keys() - actual.keys()),
            sorted(actual.keys() - expected.keys()),
        )
        changed = sorted(
            name
            for name in actual.keys() & expected.keys()
            if actual[name] != expected[name]
        )
        raise ValueError(
            f"Installed dependency contract differs: missing={missing}, extra={extra}, versions={changed}"
        )
