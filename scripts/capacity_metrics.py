# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read-only Linux memory observations for bounded capacity experiments."""

import math
import re
import threading
from collections.abc import Callable
from pathlib import Path, PurePosixPath


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _number(path: Path, *, limit: bool = False) -> int | None:
    try:
        value = int(_text(path).strip())
        return value if value >= 0 and not (limit and value >= 2**60) else None
    except ValueError:
        return None


def _fields(text: str) -> dict[str, int]:
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdecimal():
            values[parts[0]] = int(parts[1])
    return values


def _unescape_mount(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def discover_cgroup(proc: Path = Path("/proc/self")) -> tuple[int, Path] | None:
    """Resolve membership relative to the controller mount, including Docker roots."""
    memberships = []
    for line in _text(proc / "cgroup").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3:
            memberships.append((parts[1].split(","), PurePosixPath(parts[2])))
    for line in _text(proc / "mountinfo").splitlines():
        before, separator, after = line.partition(" - ")
        fields, filesystem = before.split(), after.split()
        if not separator or len(fields) < 5 or len(filesystem) < 3:
            continue
        if filesystem[0] not in ("cgroup", "cgroup2"):
            continue
        version = 2 if filesystem[0] == "cgroup2" else 1
        if version == 1 and "memory" not in filesystem[2].split(","):
            continue
        mount_root = PurePosixPath(_unescape_mount(fields[3]))
        mount_point = Path(_unescape_mount(fields[4]))
        for controllers, member in memberships:
            if (version == 1 and "memory" not in controllers) or (
                version == 2 and controllers != [""]
            ):
                continue
            if not member.is_absolute() or ".." in member.parts:
                continue
            try:
                relative = member.relative_to(mount_root)
            except ValueError:
                # A cgroup namespace can expose its own membership as '/'.
                if member != PurePosixPath("/"):
                    continue
                relative = PurePosixPath(".")
            candidate = mount_point / str(relative)
            filename = "memory.current" if version == 2 else "memory.usage_in_bytes"
            if (candidate / filename).is_file():
                return version, candidate
    return None


class MemoryReader:
    def __init__(self, proc: Path = Path("/proc/self")) -> None:
        self.proc = proc
        self.cgroup = discover_cgroup(proc)

    def __call__(self) -> dict:
        status = {}
        for line in _text(self.proc / "status").splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[2] == "kB" and parts[1].isdecimal():
                status[parts[0].rstrip(":")] = int(parts[1]) * 1024
        result: dict = {
            "process_rss_bytes": status.get("VmRSS"),
            "process_peak_rss_bytes": status.get("VmHWM"),
            "cgroup": None,
        }
        if self.cgroup is None:
            return result
        version, path = self.cgroup
        if version == 2:
            current, peak, limit, swap = (
                _number(path / name, limit=name.endswith("max"))
                for name in (
                    "memory.current",
                    "memory.peak",
                    "memory.max",
                    "memory.swap.max",
                )
            )
            events = _fields(_text(path / "memory.events"))
        else:
            current = _number(path / "memory.usage_in_bytes")
            peak = _number(path / "memory.max_usage_in_bytes")
            limit = _number(path / "memory.limit_in_bytes", limit=True)
            combined = _number(path / "memory.memsw.limit_in_bytes", limit=True)
            swap = (
                combined - limit
                if combined is not None and limit is not None and combined >= limit
                else None
            )
            events = {}
            for name in ("memory.failcnt", "memory.memsw.failcnt"):
                value = _number(path / name)
                if value is not None:
                    events[name] = value
            oom = _fields(_text(path / "memory.oom_control"))
            if "oom_kill" in oom:
                events["oom_kill"] = oom["oom_kill"]
        result["cgroup"] = {
            "version": version,
            "current_bytes": current,
            "peak_bytes": peak,
            "limit_bytes": limit,
            "swap_limit_bytes": swap,
            "events": events,
        }
        return result


class MemorySampler:
    """Keep aggregate sampled peaks; kernel lifetime peaks are reported separately."""

    def __init__(self, reader: Callable[[], dict], interval: float = 0.1) -> None:
        if not math.isfinite(interval) or not 0.01 <= interval <= 10:
            raise ValueError(
                "Sample interval must be finite and between 0.01 and 10 seconds"
            )
        self.reader, self.interval = reader, interval
        self.peaks: dict[str, int] = {}
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._error: Exception | None = None

    def _sample(self) -> dict:
        snapshot = self.reader()
        values = {
            "process_rss_bytes": snapshot["process_rss_bytes"],
            "cgroup_current_bytes": (snapshot["cgroup"] or {}).get("current_bytes"),
        }
        for key, value in values.items():
            if value is not None:
                self.peaks[key] = max(self.peaks.get(key, 0), value)
        self.samples += 1
        return snapshot

    def _loop(self) -> None:
        try:
            while not self._stop.wait(self.interval):
                self._sample()
        except Exception as error:
            self._error = error

    def __enter__(self):
        self.before = self._sample()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        self._thread.join()
        after = self._sample()
        start = (self.before["cgroup"] or {}).get("events", {})
        end = (after["cgroup"] or {}).get("events", {})
        self.report = {
            "before": self.before,
            "after": after,
            "sample_count": self.samples,
            "sampled_peaks": self.peaks,
            "cgroup_event_delta": {
                key: end[key] - start[key] for key in start.keys() & end.keys()
            },
        }
        if self._error is not None and exc_type is None:
            raise RuntimeError("Memory sampling failed") from self._error
