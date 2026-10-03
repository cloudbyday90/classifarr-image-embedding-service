# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import tempfile
import threading
from types import SimpleNamespace

import pytest
from capacity_metrics import MemorySampler
from capacity_resources import ResourceReader
from test_capacity_metrics import controller


@pytest.mark.parametrize("version", [1, 2])
def test_tasks_resolve_independent_controller_and_unknown_peak(tmp_path, version):
    proc, memory = controller(tmp_path, version)
    pids = memory if version == 2 else tmp_path / "pids"
    pids.mkdir(exist_ok=True)
    if version == 1:
        (proc / "cgroup").write_text(
            "5:memory:/docker/container\n6:pids:/tenant/child\n"
        )
        with (proc / "mountinfo").open("a") as stream:
            stream.write(f"2 3 0:4 /tenant/child {pids} ro - cgroup cgroup rw,pids\n")
    for name, value in {
        "pids.current": "17",
        "pids.max": "256",
        "pids.events": "max 3\n",
    }.items():
        (pids / name).write_text(value)
    (proc / "status").write_text("Threads:\t12\n")
    result = ResourceReader(tmp_path, proc)()
    assert result["process_threads"] == 12
    assert result["tasks"] == {
        "version": version,
        "current": 17,
        "lifetime_peak": None,
        "limit": 256,
        "events": {"max": 3},
    }
    (pids / "pids.max").write_text("max")
    (pids / "pids.peak").write_text("23")
    assert ResourceReader(tmp_path, proc)()["tasks"]["limit"] is None
    assert ResourceReader(tmp_path, proc)()["tasks"]["lifetime_peak"] == 23


def test_filesystem_counts_reserved_space_and_inodes(monkeypatch, tmp_path):
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda _: SimpleNamespace(
            f_blocks=100, f_bfree=80, f_bavail=75, f_frsize=4096, f_favail=19
        ),
        raising=False,
    )
    result = ResourceReader(tmp_path, tmp_path)()
    assert result["tasks"] is None and result["process_threads"] is None
    assert result["temporary_filesystem"] == {
        "path": str(tmp_path),
        "capacity_bytes": 409600,
        "used_bytes": 81920,
        "available_bytes": 307200,
        "available_inodes": 19,
    }


def test_missing_filesystem_is_unknown(monkeypatch, tmp_path):
    def unavailable(_):
        raise OSError("unmounted")

    monkeypatch.setattr(os, "statvfs", unavailable, raising=False)
    assert ResourceReader(tmp_path, tmp_path)()["temporary_filesystem"] is None


@pytest.mark.skipif(not hasattr(os, "statvfs"), reason="Unix filesystem accounting")
def test_anonymous_output_occupies_space_until_close(tmp_path):
    reader = ResourceReader(tmp_path, tmp_path)
    before = reader()["temporary_filesystem"]["used_bytes"]
    with tempfile.TemporaryFile(dir=tmp_path) as output:
        output.write(b"x" * (2 * 1024 * 1024))
        output.flush()
        assert list(tmp_path.iterdir()) == []
        during = reader()["temporary_filesystem"]["used_bytes"]
        assert during >= before + 2 * 1024 * 1024
    assert reader()["temporary_filesystem"]["used_bytes"] <= during


def test_aggregate_resource_peaks_minima_and_limit_events():
    observed = threading.Event()
    counts = iter([1, 9, 2])

    def reader():
        count = next(counts, 2)
        if count == 9:
            observed.set()
        return {
            "process_rss_bytes": None,
            "cgroup": None,
            "resources": {
                "process_threads": count,
                "tasks": {"current": count + 1, "events": {"max": int(count == 2)}},
                "temporary_filesystem": {
                    "used_bytes": count,
                    "available_bytes": 100 - count,
                    "available_inodes": 20 - count,
                },
            },
            "cuda": {
                "allocated_bytes": count * 2,
                "reserved_bytes": count * 3,
                "device_free_bytes": 200 - count,
            },
        }

    with MemorySampler(reader, 0.01) as sampler:
        assert observed.wait(2)
    assert sampler.report["sampled_peaks"] == {
        "process_threads": 9,
        "cgroup_tasks": 10,
        "temporary_used_bytes": 9,
        "cuda_allocated_bytes": 18,
        "cuda_reserved_bytes": 27,
    }
    assert sampler.report["sampled_minima"] == {
        "temporary_available_bytes": 91,
        "temporary_available_inodes": 11,
        "cuda_device_free_bytes": 191,
    }
    assert sampler.report["task_event_delta"] == {"max": 1}
