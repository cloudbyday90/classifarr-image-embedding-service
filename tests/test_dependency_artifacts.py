# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resolver evidence must not widen the trusted artifact or interpreter boundary."""

from copy import deepcopy

import pytest
from dependency_artifacts import report_artifacts, wheel_artifact
from dependency_profiles import selected_profile

ENVIRONMENT = {
    "python_version": "3.12",
    "implementation_name": "cpython",
    "sys_platform": "linux",
    "platform_machine": "x86_64",
}
PIP = {
    "name": "pip",
    "version": "26.2.1",
    "url": "https://files.pythonhosted.org/packages/reviewed/pip-26.2.1-py3-none-any.whl",
    "sha256": "a" * 64,
}


def pip_report():
    return {
        "version": "1",
        "pip_version": "26.2.1",
        "environment": dict(ENVIRONMENT),
        "install": [
            {
                "metadata": {"name": "pip", "version": "26.2.1"},
                "download_info": {
                    "url": PIP["url"],
                    "archive_info": {"hashes": {"sha256": PIP["sha256"]}},
                },
            }
        ],
    }


@pytest.mark.parametrize(
    "url",
    [
        "http://files.pythonhosted.org/packages/pip-26.2.1-py3-none-any.whl",
        "https://files.pythonhosted.org.evil.test/packages/pip-26.2.1-py3-none-any.whl",
        "https://user:secret@files.pythonhosted.org/packages/pip-26.2.1-py3-none-any.whl",
        "https://files.pythonhosted.org:443/packages/pip-26.2.1-py3-none-any.whl",
        "https://files.pythonhosted.org/packages/pip-26.2.1-py3-none-any.whl?token=secret",
        "https://files.pythonhosted.org/packages/pip-26.2.1-py3-none-any.whl#sha256=bad",
        "https://files.pythonhosted.org/packages/%2e%2e/pip-26.2.1-py3-none-any.whl",
        "https://files.pythonhosted.org/packages/%0apip-26.2.1-py3-none-any.whl",
        "https://files.pythonhosted.org/packages/%5cpip-26.2.1-py3-none-any.whl",
        "https://files.pythonhosted.org\n/packages/pip-26.2.1-py3-none-any.whl",
        "https://files.pythonhosted.org/packages/pip-26.2.1.tar.gz",
        "https://files.pythonhosted.org/packages/other-26.2.1-py3-none-any.whl",
        "https://files.pythonhosted.org/packages/pip-26.1-py3-none-any.whl",
        "https://files.pythonhosted.org/packages/pip-26.2.1-cp312-cp312-linux_x86_64.whl",
    ],
)
def test_rejects_untrusted_or_substituted_bootstrap_artifacts(url):
    with pytest.raises(ValueError):
        wheel_artifact(dict(PIP, url=url), selected_profile("bootstrap", "amd64"))


@pytest.mark.parametrize(
    "record",
    [
        None,
        [],
        {**PIP, "extra": "unexpected"},
        {**PIP, "sha256": "a" * 63},
        {**PIP, "sha256": "G" * 64},
        {**PIP, "name": "-pip"},
        {**PIP, "version": "1;injected"},
    ],
)
def test_rejects_invalid_artifact_identity_or_hash(record):
    with pytest.raises(ValueError):
        wheel_artifact(record, selected_profile("bootstrap", "amd64"))


@pytest.mark.parametrize("host", ["download.pytorch.org", "download-r2.pytorch.org"])
def test_torch_uses_exact_official_host_and_backend_path(host):
    profile = selected_profile("cpu", "amd64")
    record = dict(
        PIP,
        name="torch",
        version="2.14.1+cpu",
        url=(
            f"https://{host}/whl/cpu/torch-2.14.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl"
        ),
    )
    assert wheel_artifact(record, profile).version == profile.torch_version
    for url in (
        record["url"].replace("/cpu/", "/cu130/"),
        record["url"]
        .replace(host, "files.pythonhosted.org")
        .replace("/whl/cpu/", "/packages/"),
        record["url"].replace(host, "download-r2.pytorch.org.evil.test"),
    ):
        with pytest.raises(ValueError, match="origin"):
            wheel_artifact(dict(record, url=url), profile)
    with pytest.raises(ValueError, match="origin"):
        wheel_artifact(
            dict(PIP, url=f"https://{host}/whl/cpu/pip-26.2.1-py3-none-any.whl"),
            profile,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("python_version", "3.13"),
        ("implementation_name", "pypy"),
        ("sys_platform", "win32"),
        ("platform_machine", "aarch64"),
        ("platform_machine", None),
    ],
)
def test_rejects_wrong_resolver_environment(field, value):
    report = pip_report()
    report["environment"][field] = value
    with pytest.raises(ValueError):
        report_artifacts(report, selected_profile("audit", "amd64"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(version="2"),
        lambda r: r.update(pip_version="26.3"),
        lambda r: r.update(install=[]),
        lambda r: r.update(install={"bad": "shape"}),
        lambda r: r["install"].append(deepcopy(r["install"][0])),
        lambda r: r["install"][0].update(is_yanked=True),
        lambda r: r["install"][0].update(is_editable=True),
        lambda r: r["install"][0].pop("metadata"),
        lambda r: r["install"][0]["download_info"].update(archive_info={}),
        lambda r: r["install"][0]["download_info"].update(archive_info={"hashes": []}),
    ],
)
def test_rejects_incomplete_or_unsafe_resolver_evidence(mutation):
    report = pip_report()
    mutation(report)
    with pytest.raises(ValueError):
        report_artifacts(report, selected_profile("bootstrap", "amd64"))


def test_reports_cannot_substitute_backend_torch():
    with pytest.raises(ValueError, match="substituted"):
        report_artifacts(pip_report(), selected_profile("cpu", "amd64"))
    environment, artifacts = report_artifacts(
        pip_report(), selected_profile("bootstrap", "amd64")
    )
    assert environment == ENVIRONMENT
    assert artifacts[0].record() == PIP
