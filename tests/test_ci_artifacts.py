# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import hashlib
import io
import json
import tarfile
from unittest.mock import MagicMock

import ci_downloads
import install_ci_tool
import pytest
from ci_tool_contracts import ROOT, tool_contract


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/a/b/releases/download/v1/tool",
        "https://github.com.evil.test/tool",
        "https://user:secret@github.com/tool",
        "https://github.com:8443/tool",
        "https://github.com/tool#fragment",
        "https://github.com/tool\\unsafe",
        "https://github.com/tool\nunsafe",
    ],
)
def test_download_refuses_unapproved_authorities(url):
    with pytest.raises(ValueError):
        ci_downloads.validate_asset_url(url)


def response(content):
    result = MagicMock()
    result.status = 200
    result.url = "https://release-assets.githubusercontent.com/test?signature=fixture"
    result.read1.side_effect = [content, b""]
    result.__enter__.return_value = result
    return result


@pytest.mark.parametrize(
    "change", ["hash", "empty", "size", "origin", "status", "deadline"]
)
def test_download_rejects_tampering_and_incomplete_responses(
    tmp_path, monkeypatch, change
):
    content = b"reviewed artifact"
    reply = response(b"" if change == "empty" else content)
    opener = MagicMock()
    opener.open.return_value = reply
    monkeypatch.setattr(ci_downloads, "build_opener", lambda *args: opener)
    if change == "origin":
        reply.url = "https://evil.test/stolen"
    if change == "status":
        reply.status = 206
    if change == "deadline":
        clock = iter([0, 601])
        monkeypatch.setattr(ci_downloads.time, "monotonic", lambda: next(clock))
    with pytest.raises((ValueError, TimeoutError)):
        ci_downloads.verified_download(
            "https://github.com/fixture/tool/releases/download/v1/tool",
            tmp_path / "staged",
            "0" * 64 if change == "hash" else hashlib.sha256(content).hexdigest(),
            1 if change == "size" else 100,
        )


def test_reviewed_download_writes_exact_bytes(tmp_path, monkeypatch):
    content = b"reviewed artifact"
    opener = MagicMock()
    opener.open.return_value = response(content)
    monkeypatch.setattr(ci_downloads, "build_opener", lambda *args: opener)
    target = tmp_path / "staged"
    ci_downloads.verified_download(
        "https://github.com/fixture/tool/releases/download/v1/tool",
        target,
        hashlib.sha256(content).hexdigest(),
        100,
    )
    assert target.read_bytes() == content
    assert opener.open.call_args.kwargs["timeout"] == 30


def archive(path, members):
    with tarfile.open(path, "w:gz") as tar:
        for name, kind in members:
            entry = tarfile.TarInfo(name)
            entry.type = kind
            entry.size = 4 if kind == tarfile.REGTYPE else 0
            entry.linkname = "../../outside"
            tar.addfile(entry, io.BytesIO(b"tool") if kind == tarfile.REGTYPE else None)


@pytest.mark.parametrize(
    "members",
    [
        [],
        [("../tool", tarfile.REGTYPE)],
        [("tool", tarfile.SYMTYPE)],
        [("tool", tarfile.LNKTYPE)],
        [("tool", tarfile.CHRTYPE)],
        [("tool", tarfile.REGTYPE), ("tool", tarfile.REGTYPE)],
    ],
)
def test_archive_requires_one_regular_named_executable(tmp_path, members):
    source, target = tmp_path / "archive", tmp_path / "tool"
    archive(source, members)
    with pytest.raises(ValueError):
        install_ci_tool.extract_executable(source, "tool", target)
    assert not target.exists()


def test_extracts_only_named_executable_without_other_archive_paths(tmp_path):
    source, target = tmp_path / "archive", tmp_path / "tool"
    archive(source, [("tool", tarfile.REGTYPE), ("../outside", tarfile.REGTYPE)])
    install_ci_tool.extract_executable(source, "tool", target)
    assert target.read_bytes() == b"tool"
    assert not (tmp_path.parent / "outside").exists()


def test_rejected_download_keeps_previous_binary_and_removes_staging(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(install_ci_tool.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install_ci_tool.platform, "machine", lambda: "x86_64")
    target = tmp_path / "gitleaks"
    target.write_bytes(b"previous reviewed binary")

    def fail(*args):
        args[1].write_bytes(b"tampered")
        raise ValueError("digest mismatch")

    monkeypatch.setattr(install_ci_tool, "verified_download", fail)
    with pytest.raises(ValueError):
        install_ci_tool.install("gitleaks", tmp_path)
    assert target.read_bytes() == b"previous reviewed binary"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["gitleaks"]


@pytest.mark.parametrize("system,machine", [("Windows", "AMD64"), ("Linux", "aarch64")])
def test_installer_rejects_unreviewed_platforms(tmp_path, monkeypatch, system, machine):
    monkeypatch.setattr(install_ci_tool.platform, "system", lambda: system)
    monkeypatch.setattr(install_ci_tool.platform, "machine", lambda: machine)
    with pytest.raises(ValueError):
        install_ci_tool.install("gitleaks", tmp_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("sha256", ""),
        ("sha256", "f" * 63),
        ("executable", "../tool"),
        ("kind", "shell"),
        ("max_bytes", True),
        ("max_bytes", 0),
        ("url", "https://evil.test/tool"),
        ("url", "https://release-assets.githubusercontent.com/tool"),
    ],
)
def test_tool_manifest_rejects_invalid_contracts(tmp_path, key, value):
    data = json.loads((ROOT / ".github/ci-tools.json").read_text())
    data["tools"]["gitleaks"][key] = value
    manifest = tmp_path / "contract.json"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        tool_contract("gitleaks", manifest)


@pytest.mark.parametrize("name", ["gitleaks", "trivy", "buildx", "codeql"])
def test_all_reviewed_contracts_load(name):
    assert tool_contract(name)["sha256"]
