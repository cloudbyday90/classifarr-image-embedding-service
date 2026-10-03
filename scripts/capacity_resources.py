# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read-only task and filesystem observations, including anonymous worker output."""

import os
from pathlib import Path

from capacity_metrics import _fields, _number, _text, discover_cgroup


class ResourceReader:
    def __init__(self, temporary: Path, proc: Path = Path("/proc/self")) -> None:
        self.temporary, self.proc = temporary, proc
        self.pids = discover_cgroup(proc, "pids")

    def __call__(self) -> dict:
        threads = _fields(_text(self.proc / "status").replace("Threads:", "Threads"))
        tasks = None
        if self.pids is not None:
            version, path = self.pids
            tasks = {
                "version": version,
                "current": _number(path / "pids.current"),
                "lifetime_peak": _number(path / "pids.peak"),
                "limit": _number(path / "pids.max", limit=True),
                "events": _fields(_text(path / "pids.events")),
            }
        filesystem = None
        try:
            info = os.statvfs(self.temporary)
            filesystem = {
                "path": str(self.temporary),
                "capacity_bytes": info.f_blocks * info.f_frsize,
                "used_bytes": (info.f_blocks - info.f_bfree) * info.f_frsize,
                "available_bytes": info.f_bavail * info.f_frsize,
                "available_inodes": info.f_favail,
            }
        except (OSError, AttributeError):
            pass
        return {
            "process_threads": threads.get("Threads"),
            "tasks": tasks,
            "temporary_filesystem": filesystem,
        }
