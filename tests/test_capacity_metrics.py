# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import threading

import pytest
from capacity_metrics import MemoryReader, MemorySampler, discover_cgroup


def controller(
    tmp_path, version=2, root="/docker/container", member="/docker/container"
):
    proc, mount = tmp_path / "proc", tmp_path / "controller"
    proc.mkdir()
    mount.mkdir()
    filesystem = "cgroup2 cgroup rw" if version == 2 else "cgroup cgroup rw,memory"
    (proc / "cgroup").write_text(
        f"0::{member}\n" if version == 2 else f"5:memory:{member}\n"
    )
    (proc / "mountinfo").write_text(f"1 2 0:3 {root} {mount} ro - {filesystem}\n")
    (proc / "status").write_text("Name:\tfixture\nVmRSS:\t12 kB\nVmHWM:\t19 kB\n")
    (
        mount / ("memory.current" if version == 2 else "memory.usage_in_bytes")
    ).write_text("4096")
    return proc, mount


def test_v2_reads_own_controller_and_preserves_missing_counters(tmp_path):
    proc, mount = controller(tmp_path)
    for name, value in {
        "memory.peak": "8192",
        "memory.max": "65536",
        "memory.swap.max": "0",
        "memory.events": "max 2\noom 1\noom_kill 1\n",
    }.items():
        (mount / name).write_text(value)
    result = MemoryReader(proc)()
    assert result["process_rss_bytes"] == 12 * 1024
    assert result["process_peak_rss_bytes"] == 19 * 1024
    assert result["cgroup"] == {
        "version": 2,
        "current_bytes": 4096,
        "peak_bytes": 8192,
        "limit_bytes": 65536,
        "swap_limit_bytes": 0,
        "events": {"max": 2, "oom": 1, "oom_kill": 1},
    }
    (mount / "memory.events").unlink()
    assert MemoryReader(proc)()["cgroup"]["events"] == {}


def test_v1_uses_relative_mount_and_combined_swap_limit(tmp_path):
    proc, mount = controller(tmp_path, 1)
    for name, value in {
        "memory.max_usage_in_bytes": "7000",
        "memory.limit_in_bytes": "65536",
        "memory.memsw.limit_in_bytes": "65536",
        "memory.failcnt": "3",
        "memory.oom_control": "oom_kill_disable 0\nunder_oom 0\noom_kill 2\n",
    }.items():
        (mount / name).write_text(value)
    result = MemoryReader(proc)()["cgroup"]
    assert result["version"] == 1 and result["peak_bytes"] == 7000
    assert result["swap_limit_bytes"] == 0
    assert result["events"] == {"memory.failcnt": 3, "oom_kill": 2}
    (mount / "memory.memsw.limit_in_bytes").write_text("98304")
    assert MemoryReader(proc)()["cgroup"]["swap_limit_bytes"] == 32768


@pytest.mark.parametrize(
    "version,value", [(1, str(2**63 - 4096)), (2, "max"), (2, "-1"), (2, "bad")]
)
def test_unbounded_or_invalid_limit_is_unknown(tmp_path, version, value):
    proc, mount = controller(tmp_path, version)
    (mount / ("memory.max" if version == 2 else "memory.limit_in_bytes")).write_text(
        value
    )
    assert MemoryReader(proc)()["cgroup"]["limit_bytes"] is None


def test_v2_nested_membership_and_cgroup_namespace(tmp_path):
    proc, mount = controller(tmp_path, root="/", member="/tenant")
    nested = mount / "tenant"
    nested.mkdir()
    (nested / "memory.current").write_text("1")
    assert discover_cgroup(proc) == (2, nested)
    (proc / "cgroup").write_text("0::/\n")
    (proc / "mountinfo").write_text(
        f"1 2 0:3 /docker/container {mount} ro - cgroup2 cgroup rw\n"
    )
    assert discover_cgroup(proc) == (2, mount)


@pytest.mark.parametrize(
    "member", ["/other/container", "/docker/container/../other", "relative"]
)
def test_unmatched_or_traversing_membership_never_reads_mount_root(tmp_path, member):
    proc, _ = controller(tmp_path, member=member)
    assert discover_cgroup(proc) is None
    assert MemoryReader(proc)()["cgroup"] is None


def test_missing_proc_information_is_explicitly_unknown(tmp_path):
    result = MemoryReader(tmp_path)()
    assert result == {
        "process_rss_bytes": None,
        "process_peak_rss_bytes": None,
        "cgroup": None,
    }


def test_escaped_controller_mount_is_resolved(tmp_path):
    proc, mount = controller(tmp_path)
    renamed = mount.rename(tmp_path / "controller space")
    escaped = str(renamed).replace(" ", r"\040")
    (proc / "mountinfo").write_text(
        f"1 2 0:3 /docker/container {escaped} ro - cgroup2 cgroup rw\n"
    )
    assert discover_cgroup(proc) == (2, renamed)


def test_background_reader_failure_cannot_report_success():
    failed = threading.Event()
    calls = 0

    def reader():
        nonlocal calls
        calls += 1
        if calls == 2:
            failed.set()
            raise OSError("fixture")
        return {"process_rss_bytes": None, "cgroup": None}

    with pytest.raises(RuntimeError, match="sampling failed"):
        with MemorySampler(reader, 0.01) as sampler:
            assert failed.wait(2)
    assert not sampler._thread.is_alive()


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf"), 11])
def test_sampler_refuses_invalid_intervals(interval):
    with pytest.raises(ValueError, match="finite"):
        MemorySampler(lambda: {}, interval)


def test_sampler_aggregates_stage_peaks_and_counter_changes():
    values = iter([10, 90, 20])
    observed_peak = threading.Event()

    def reader():
        value = next(values, 20)
        if value == 90:
            observed_peak.set()
        return {
            "process_rss_bytes": value,
            "cgroup": {"current_bytes": value * 2, "events": {"oom": int(value == 20)}},
        }

    with MemorySampler(reader, 0.01) as sampler:
        assert observed_peak.wait(2)
    assert sampler.report["sampled_peaks"] == {
        "process_rss_bytes": 90,
        "cgroup_current_bytes": 180,
    }
    assert sampler.report["cgroup_event_delta"] == {"oom": 1}
    assert sampler.report["sample_count"] >= 3
    assert not sampler._thread.is_alive()


def test_sampler_joins_after_workload_failure():
    with pytest.raises(RuntimeError, match="workload"):
        with MemorySampler(
            lambda: {"process_rss_bytes": None, "cgroup": None}
        ) as sampler:
            raise RuntimeError("workload")
    assert not sampler._thread.is_alive()
    assert sampler.report["sampled_peaks"] == {}
