# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import re
from pathlib import Path

import pytest
import yaml
from check_osv_reports import check_report

ROOT = Path(__file__).resolve().parents[1]


def workflows():
    return [
        (path, yaml.safe_load(path.read_text()))
        for path in (ROOT / ".github/workflows").glob("*.yml")
    ]


def test_workflow_execution_refs_are_immutable_and_local_callees_exist():
    for path, workflow in workflows():
        assert workflow["permissions"] == {}, path
        for job in workflow["jobs"].values():
            nodes = [job, *job.get("steps", [])]
            for node in nodes:
                if "uses" not in node:
                    continue
                ref = node["uses"]
                if ref.startswith("./"):
                    assert (ROOT / ref).is_file()
                elif ref.startswith("docker://"):
                    assert re.fullmatch(r"docker://[^@]+@sha256:[a-f0-9]{64}", ref), (
                        path,
                        ref,
                    )
                else:
                    assert re.fullmatch(r"[^@]+@[a-f0-9]{40}", ref), (path, ref)
                if ref.startswith("actions/checkout@"):
                    assert node["with"]["persist-credentials"] is False, path
                    assert job["permissions"]["contents"] == "read", path


def test_only_tag_release_jobs_have_package_write_and_scanner_has_no_pr_write():
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    for _, workflow in workflows():
        for job in workflow["jobs"].values():
            permissions = job.get("permissions", {})
            assert "pull-requests" not in permissions
            assert "contents" not in permissions or permissions["contents"] == "read"
            if permissions.get("packages") == "write":
                assert job in ci["jobs"].values()
                assert "github.event_name == 'push'" in job["if"]
                assert "refs/tags/v" in job["if"]
                assert job["needs"]


def test_scanners_and_build_setup_use_reviewed_native_contracts():
    osv = yaml.safe_load((ROOT / ".github/workflows/osv-scanner.yml").read_text())
    scanner = osv["jobs"]["pr-scan-dependabot"]["steps"][-1]
    assert scanner["with"]["entrypoint"] == "/root/osv-scanner"
    assert "--allow-no-lockfiles=false" in scanner["with"]["args"]
    trivy = yaml.safe_load((ROOT / ".github/workflows/trivy.yml").read_text())
    for job in trivy["jobs"].values():
        for step in job["steps"]:
            if step.get("uses", "").startswith("aquasecurity/trivy-action@"):
                assert step["with"]["skip-setup-trivy"] is True
                assert step["with"]["cache"] is False
    codeql = yaml.safe_load((ROOT / ".github/workflows/codeql.yml").read_text())
    init = next(
        s for s in codeql["jobs"]["analyze"]["steps"] if "/init@" in s.get("uses", "")
    )
    assert init["with"]["tools"].endswith(
        "/verified-tools/codeql-bundle-linux64.tar.gz"
    )
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    steps = ci["jobs"]["docker-release"]["steps"]
    qemu = next(
        s for s in steps if s.get("uses", "").startswith("docker/setup-qemu-action@")
    )
    buildx = next(
        s for s in steps if s.get("uses", "").startswith("docker/setup-buildx-action@")
    )
    assert "@sha256:" in qemu["with"]["image"]
    assert qemu["with"]["platforms"] == "arm64"
    assert "@sha256:" in buildx["with"]["driver-opts"]
    assert buildx["with"]["buildkitd-flags"] == "--debug=false"
    assert any("install_ci_tool.py buildx" in step.get("run", "") for step in steps)


@pytest.mark.parametrize(
    "content", ["", "{}", "[]", '{"results":null}', '{"results":{}}', "broken JSON"]
)
def test_osv_report_rejects_missing_or_invalid_results(tmp_path, content):
    path = tmp_path / "report.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        check_report(path, "failure")


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_osv_completed_clean_or_vulnerability_reports_can_be_compared(
    tmp_path, outcome
):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"results": []}))
    check_report(path, outcome)


def test_osv_skipped_and_absent_reports_fail_closed(tmp_path):
    with pytest.raises(ValueError):
        check_report(tmp_path / "missing", "skipped")
    with pytest.raises(FileNotFoundError):
        check_report(tmp_path / "missing", "failure")
