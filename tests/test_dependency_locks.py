# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stale intent, partial publication and runtime inventory drift fail closed."""

import json
from pathlib import Path
from types import SimpleNamespace

import dependency_locks
import dependency_profiles
import install_dependencies
import pytest
from dependency_artifacts import WheelArtifact
from dependency_locks import load_lock, verify_inventory, write_lock
from dependency_profiles import check_environment, selected_profile
from install_dependencies import pip_command
from test_dependency_artifacts import ENVIRONMENT, PIP


@pytest.fixture
def bootstrap_lock(tmp_path):
    (tmp_path / "requirements-bootstrap.txt").write_bytes(b"pip==26.2.1\n")
    profile = selected_profile("bootstrap", "amd64")
    artifact = WheelArtifact(**PIP)
    write_lock(tmp_path, profile, ENVIRONMENT, [artifact])
    return tmp_path, profile, artifact


def test_lock_is_portable_across_checkout_line_endings(bootstrap_lock):
    root, profile, artifact = bootstrap_lock
    assert load_lock(root, profile)[1] == [artifact]
    (root / "requirements-bootstrap.txt").write_bytes(b"pip==26.2.1\r\n")
    lock = root / "requirements/locks/bootstrap.txt"
    lock.write_bytes(lock.read_bytes().replace(b"\n", b"\r\n"))
    assert load_lock(root, profile)[1] == [artifact]
    (root / "requirements-bootstrap.txt").write_bytes(b"pip==26.3\n")
    with pytest.raises(ValueError, match="inputs changed"):
        load_lock(root, profile)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(schema_version=True),
        lambda m: m.update(profile="cpu"),
        lambda m: m.update(resolver="pip==26.3"),
        lambda m: m.update(architecture="arm64"),
        lambda m: m["environment"].update(sys_platform="darwin"),
        lambda m: m.update(artifacts=[]),
        lambda m: m.update(artifacts=[None]),
        lambda m: m["artifacts"].append(m["artifacts"][0].copy()),
        lambda m: m["artifacts"][0].update(sha256="b" * 64),
        lambda m: m.update(requirements_sha256="b" * 64),
    ],
)
def test_tampered_manifests_are_rejected(bootstrap_lock, mutation):
    root, profile, _ = bootstrap_lock
    file = root / "requirements/locks/bootstrap.json"
    manifest = json.loads(file.read_text())
    mutation(manifest)
    file.write_bytes(json.dumps(manifest).encode())
    with pytest.raises(ValueError):
        load_lock(root, profile)


def test_partial_publication_rejects_even_matching_unexpected_text(bootstrap_lock):
    root, profile, _ = bootstrap_lock
    file = root / "requirements/locks/bootstrap.txt"
    file.write_bytes(file.read_bytes() + b"--extra-index-url https://evil.test\n")
    manifest_file = file.with_suffix(".json")
    manifest = json.loads(manifest_file.read_text())
    manifest["requirements_sha256"] = dependency_locks.sha256(file.read_bytes())
    manifest_file.write_bytes(json.dumps(manifest).encode())
    with pytest.raises(ValueError, match="text/attestation"):
        load_lock(root, profile)


@pytest.mark.parametrize(
    "installed",
    [
        [],
        [("pip", "26.1")],
        [("pip", "26.2.1"), ("unreviewed", "1.0")],
        [("pip", "26.2.1"), ("Pip", "26.2.1")],
    ],
)
def test_inventory_rejects_missing_changed_extra_and_duplicate_packages(
    monkeypatch, installed
):
    monkeypatch.setattr(
        dependency_locks,
        "distributions",
        lambda **kw: [
            SimpleNamespace(metadata={"Name": name}, version=version)
            for name, version in installed
        ],
    )
    with pytest.raises(ValueError, match="distribution|contract"):
        verify_inventory([WheelArtifact(**PIP)])


def test_isolated_inventory_checks_only_requested_target(monkeypatch):
    seen = []

    def installed(**kwargs):
        seen.append(kwargs)
        return [SimpleNamespace(metadata={"Name": "Pip"}, version="26.2.1")]

    monkeypatch.setattr(dependency_locks, "distributions", installed)
    verify_inventory([WheelArtifact(**PIP)], Path("/audit"))
    assert seen == [{"path": ["/audit"]}]


@pytest.mark.parametrize(
    "backend,arch",
    [("cuda", "arm64"), ("openvino", "arm64"), ("qa", "arm64"), ("unknown", "amd64")],
)
def test_unsupported_profiles_never_fall_back(backend, arch):
    with pytest.raises(ValueError, match="Unsupported"):
        selected_profile(backend, arch)


def test_install_cannot_cross_architecture_or_use_another_python(monkeypatch):
    monkeypatch.setattr(dependency_profiles.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dependency_profiles.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(dependency_profiles.sys, "version_info", (3, 12, 3))
    assert selected_profile("cpu").architecture == "arm64"
    with pytest.raises(ValueError, match="does not match"):
        check_environment(selected_profile("cpu", "amd64"))
    monkeypatch.setattr(dependency_profiles.sys, "version_info", (3, 13, 0))
    with pytest.raises(ValueError, match="CPython 3.12"):
        selected_profile("cpu")


def test_installer_forces_complete_hash_binary_and_no_index_mode():
    command = pip_command("-r", "/reviewed.txt", target=Path("/isolated"))
    for flag in ("--isolated", "--no-index", "--require-hashes", "--only-binary=:all:"):
        assert flag in command
    assert command[-4:] == ["--target", "/isolated", "-r", "/reviewed.txt"]


@pytest.mark.parametrize("installed_version", ["26.1", "26.2.1"])
def test_bootstrap_verify_does_not_accept_an_unreviewed_installer(
    bootstrap_lock, monkeypatch, installed_version
):
    root, profile, _ = bootstrap_lock
    monkeypatch.setattr(install_dependencies, "selected_profile", lambda _: profile)
    monkeypatch.setattr(install_dependencies, "check_environment", lambda _: None)
    monkeypatch.setattr(install_dependencies, "version", lambda _: installed_version)
    monkeypatch.setattr(
        install_dependencies.sys,
        "argv",
        [
            "install_dependencies.py",
            "--backend",
            "bootstrap",
            "--root",
            str(root),
            "--verify-only",
        ],
    )
    if installed_version != "26.2.1":
        with pytest.raises(ValueError, match="Bootstrap installer version"):
            install_dependencies.main()
    else:
        install_dependencies.main()
