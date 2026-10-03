# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Missing native execution or unexpected skips cannot produce a green gate."""

from types import SimpleNamespace

import pytest
from windows_contract_policy import REQUIRED, WindowsContractPolicy, allowed_skip


def completed_policy():
    policy = WindowsContractPolicy()
    for file, cases in REQUIRED.items():
        for name, count in cases.items():
            for index in range(count):
                policy.pytest_runtest_logreport(
                    SimpleNamespace(
                        nodeid=f"tests/{file}::{name}[{index}]",
                        when="call",
                        outcome="passed",
                        passed=True,
                        skipped=False,
                        failed=False,
                    )
                )
    return policy


def test_gate_requires_successful_execution_not_collection_or_setup():
    policy = completed_policy()
    session = SimpleNamespace(exitstatus=0)
    policy.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 0 and not policy.errors
    key = ("test_secret_setup.py", "test_windows_junction_root_is_refused")
    policy.passed[key] = 0
    policy.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 1 and "0/1 passed" in policy.errors[0]


@pytest.mark.parametrize(
    "case",
    [
        "test_windows_junction_root_is_refused",
        "test_windows_dacl_is_owner_only_at_creation_and_after_rotation",
    ],
)
def test_critical_skip_fails_even_with_a_plausible_platform_reason(case):
    policy = completed_policy()
    report = SimpleNamespace(
        nodeid=f"tests/test_secret_setup.py::{case}",
        when="setup",
        outcome="skipped",
        skipped=True,
        passed=False,
        failed=False,
        longrepr="secret-bearing arbitrary skip message",
        capstdout="private data",
    )
    policy.pytest_runtest_logreport(report)
    session = SimpleNamespace(exitstatus=0)
    policy.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 1
    assert policy.outcomes[-1]["classification"] == "Unexpected skip"
    assert "private data" not in str(policy.outcomes) and "secret-bearing" not in str(
        policy.outcomes
    )


def test_skip_exceptions_are_scoped_to_specific_setup_cases():
    assert allowed_skip(
        "tests/test_secret_setup.py::test_writable_shared_setup_directory_is_refused[777]"
    )
    assert allowed_skip(
        "tests/test_secret_setup.py::test_unsafe_targets_are_refused_without_following_or_mutating_them[symlink-False-.env]"
    )
    assert (
        allowed_skip(
            "tests/test_secret_setup.py::test_unsafe_targets_are_refused_without_following_or_mutating_them[hardlink]"
        )
        is None
    )
    assert (
        allowed_skip(
            "tests/test_other.py::test_writable_shared_setup_directory_is_refused"
        )
        is None
    )


def test_only_uvloop_fixture_cases_are_deselected():
    items = [
        SimpleNamespace(
            nodeid=str(value), callspec=SimpleNamespace(params={"anyio_backend": value})
        )
        for value in (False, True, "asyncio")
    ]
    calls = []
    config = SimpleNamespace(
        hook=SimpleNamespace(pytest_deselected=lambda **kw: calls.append(kw))
    )
    policy = WindowsContractPolicy()
    policy.pytest_collection_modifyitems(config, items)
    assert [item.nodeid for item in items] == ["False", "asyncio"]
    assert policy.deselected == ["True"] and len(calls[0]["items"]) == 1


def test_failure_exit_status_is_never_cleared():
    policy = completed_policy()
    session = SimpleNamespace(exitstatus=2)
    policy.pytest_sessionfinish(session, 2)
    assert session.exitstatus == 2
