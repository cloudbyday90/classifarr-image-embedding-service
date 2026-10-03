# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native validation targets cannot substitute an OS, interpreter or graph."""

from collections import namedtuple
from copy import deepcopy

import dependency_profiles as profiles
import pytest
from dependency_artifacts import WheelArtifact, check_report_environment
from dependency_locks import load_lock, write_lock
from test_dependency_artifacts import ENVIRONMENT, PIP


@pytest.mark.parametrize("minor", [13, 14])
def test_windows_selection_and_complete_lock_target(monkeypatch, tmp_path, minor):
    version = namedtuple("Version", "major minor micro")(3, minor, 5)
    monkeypatch.setattr(profiles.platform, "system", lambda: "Windows")
    monkeypatch.setattr(profiles.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(profiles.sys, "version_info", version)
    monkeypatch.setattr(profiles.sysconfig, "get_config_var", lambda name: 0)
    profile = profiles.selected_profile("bootstrap")
    profiles.check_environment(profile)
    env = dict(
        ENVIRONMENT,
        python_version=f"3.{minor}",
        sys_platform="win32",
        platform_machine="AMD64",
    )
    (tmp_path / "requirements-bootstrap.txt").write_bytes(b"pip==26.2.1\n")
    write_lock(tmp_path, profile, env, [WheelArtifact(**PIP)])
    assert load_lock(tmp_path, profile)[1] == [WheelArtifact(**PIP)]
    other = profiles.PROFILES[f"bootstrap-py3{27 - minor}-windows-amd64"]
    with pytest.raises(ValueError, match="CPython"):
        profiles.check_environment(other)
    monkeypatch.setattr(profiles.sysconfig, "get_config_var", lambda name: 1)
    with pytest.raises(ValueError, match="CPython"):
        profiles.check_environment(profile)


@pytest.mark.parametrize(
    "mutation",
    [
        {"sys_platform": "linux"},
        {"python_version": "3.12"},
        {"platform_machine": "ARM64"},
    ],
)
def test_windows_report_cannot_reuse_linux_or_other_target(mutation):
    env = dict(
        deepcopy(ENVIRONMENT),
        python_version="3.14",
        sys_platform="win32",
        platform_machine="AMD64",
    )
    env.update(mutation)
    with pytest.raises(ValueError):
        check_report_environment(
            env, profiles.PROFILES["windows-contracts-py314-windows-amd64"]
        )
