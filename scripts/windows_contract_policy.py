# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Mandatory native outcomes and tightly scoped unsupported-platform skips."""

from collections import Counter

SUITES = (
    "tests/test_secret_setup.py",
    "tests/test_remote_process.py",
    "tests/test_remote_process_network.py",
    "tests/test_remote_native_transport.py",
    "tests/test_remote_batch_process.py",
    "tests/test_response_http_protocol.py",
)

# Minimum parametrized successes; collection alone never establishes execution.
REQUIRED = {
    "test_secret_setup.py": {
        "test_windows_dacl_is_owner_only_at_creation_and_after_rotation": 1,
        "test_windows_junction_root_is_refused": 1,
        "test_windows_acl_or_handle_failure_happens_before_data_and_cleans_file": 3,
        "test_exclusive_publication_has_one_winner_under_real_concurrent_writers": 1,
        "test_final_name_is_never_visible_with_partial_data": 2,
        "test_start_launcher_generates_once_and_keeps_key_out_of_console": 1,
    },
    "test_remote_process.py": {
        "test_production_worker_blocks_private_destination_and_hides_ambient_secrets": 1,
        "test_native_blocked_dns_is_terminated_and_reaped": 1,
        "test_detached_inference_keeps_permit_until_child_reaped": 1,
        "test_cleanup_failure_still_closes_pipes_and_reaps_before_return": 1,
    },
    "test_remote_process_network.py": {
        "test_native_trickle_redirect_and_tls_share_total_budget": 4,
        "test_native_worker_success_preserves_host_sni_and_strips_ambient_state": 2,
        "test_native_worker_rejects_bad_tls_without_url_details": 2,
    },
    "test_remote_native_transport.py": {
        "test_real_numeric_transport_preserves_host_tls_and_ignores_ambient_auth": 2,
        "test_real_tls_rejects_untrusted_ca_and_wrong_hostname": 2,
        "test_real_gzip_body_is_limited_by_decoded_bytes": 1,
    },
    "test_remote_batch_process.py": {
        "test_native_32_item_dns_batch_uses_one_shared_budget": 1,
        "test_detached_batch_keeps_permit_until_active_fetch_reaped": 1,
        "test_native_earlier_of_per_image_and_shared_deadline": 2,
        "test_native_success_then_trickle_spends_remaining_shared_budget": 1,
    },
    "test_response_http_protocol.py": {
        "test_slow_reader_is_aborted_before_ingress_release_and_capacity_recovers": 8,
        "test_stream_gaps_trickle_and_disconnect_finish_owned_work": 6,
        "test_successful_terminal_response_bytes_and_keepalive_are_preserved": 2,
        "test_real_starlette_stream_cancels_and_cleans_before_owner_release": 2,
    },
}

POSIX_SETUP = {
    "test_writable_shared_setup_directory_is_refused",
    "test_private_permissions_are_in_place_before_the_first_write",
    "test_posix_linked_setup_directory_is_refused",
}
OPTIONAL_SYMLINK_CASES = {
    f"[{kind}-{force}-{name}]"
    for kind in ("symlink", "dangling")
    for force in (False, True)
    for name in (".env", "config.toml")
}


def identity(nodeid: str) -> tuple[str, str]:
    file, _, case = nodeid.partition("::")
    return file.rsplit("/", 1)[-1], case.partition("[")[0]


def allowed_skip(nodeid: str) -> str | None:
    file, name = identity(nodeid)
    if file != "test_secret_setup.py":
        return None
    if name in POSIX_SETUP:
        return "POSIX-only setup contract"
    if (
        name == "test_unsafe_targets_are_refused_without_following_or_mutating_them"
        and "[" + nodeid.partition("[")[2] in OPTIONAL_SYMLINK_CASES
    ):
        return "Windows symlink privilege unavailable"
    return None


class WindowsContractPolicy:
    def __init__(self) -> None:
        self.passed: Counter = Counter()
        self.outcomes: list[dict] = []
        self.deselected: list[str] = []
        self.errors: list[str] = []

    def pytest_collection_modifyitems(self, config, items) -> None:
        unsupported = [
            item
            for item in items
            if getattr(getattr(item, "callspec", None), "params", {}).get(
                "anyio_backend"
            )
            is True
        ]
        for item in unsupported:
            items.remove(item)
            self.deselected.append(item.nodeid)
        if unsupported:
            config.hook.pytest_deselected(items=unsupported)

    def pytest_runtest_logreport(self, report) -> None:
        if report.passed and report.when == "call":
            self.passed[identity(report.nodeid)] += 1
        if report.when == "call" or report.skipped or report.failed:
            result = {
                "nodeid": report.nodeid,
                "phase": report.when,
                "outcome": report.outcome,
            }
            if report.skipped:
                reason = allowed_skip(report.nodeid)
                result["classification"] = reason or "Unexpected skip"
                if reason is None:
                    self.errors.append(f"Unexpected skip: {report.nodeid}")
            # Never serialize test stdout, traceback, arbitrary skip reasons or secrets.
            self.outcomes.append(result)

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        for file, cases in REQUIRED.items():
            for case, minimum in cases.items():
                count = self.passed[(file, case)]
                if count < minimum:
                    self.errors.append(
                        f"Required {file}::{case}: {count}/{minimum} passed"
                    )
        if self.errors:
            session.exitstatus = 1

    def pytest_terminal_summary(self, terminalreporter) -> None:
        for error in self.errors:
            terminalreporter.write_line(error, red=True)
