# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bounded probe-only request/owner timelines, including detached native work."""

import time
from contextvars import ContextVar
from threading import Lock

from image_embedder.execution_observation import ExecutionPhase

MAX_TIMELINES = 384  # 2 models * 4 fixtures * 3 sizes * 8 callers * 2 repeats.
PHASES = frozenset(
    {
        "request_started",
        "submitted",
        "admission_started",
        "admitted",
        "read_lock_acquired",
        "dispatched",
        "native_started",
        "native_failed",
        "native_finished",
        "waiter_detached",
        "owner_failed",
        "owner_cancelled",
        "owner_completed",
        "response_started",
        "request_completed",
    }
)


class ExecutionTimeline:
    def __init__(self, label, origin, clock):
        self.label, self.origin, self.clock = label, origin, clock
        self.events = {}
        self.invalid = False
        self.native_result_verified: bool | None = None
        self.lock = Lock()

    def note(self, phase: str) -> None:
        with self.lock:
            now = self.clock() - self.origin
            if (
                phase not in PHASES
                or phase in self.events
                or now < max(self.events.values(), default=0)
            ):
                self.invalid = True
                raise ValueError("Invalid capacity timeline")
            self.events[phase] = now

    def mark(self, phase: ExecutionPhase) -> None:
        self.note(phase)

    def verify_native_result(self, passed: bool) -> None:
        with self.lock:
            if (
                self.native_result_verified is not None
                or "native_started" not in self.events
                or "native_finished" in self.events
            ):
                self.invalid = True
                raise ValueError("Invalid native result observation")
            self.native_result_verified = passed

    def snapshot(self) -> dict:
        with self.lock:
            events, invalid = dict(self.events), self.invalid
            native_result_verified = self.native_result_verified
        if invalid or "request_completed" not in events:
            raise AssertionError("Capacity request timeline did not settle")
        if "submitted" in events and "owner_completed" not in events:
            raise AssertionError("Capacity inference timeline did not settle")
        if "native_started" in events and "native_finished" not in events:
            raise AssertionError("Capacity native timeline did not settle")
        if "native_finished" in events and native_result_verified is None:
            raise AssertionError("Capacity native result was not verified")

        def interval(start, finish):
            if start not in events or finish not in events:
                return None
            elapsed = events[finish] - events[start]
            if elapsed < 0:
                raise AssertionError("Capacity timeline interval went backwards")
            return elapsed / 1_000_000_000

        durations = {
            name: interval(start, finish)
            for name, start, finish in (
                ("request_to_submit", "request_started", "submitted"),
                ("owner_scheduling", "submitted", "admission_started"),
                ("admission_wait", "admission_started", "admitted"),
                ("read_lock_wait", "admitted", "read_lock_acquired"),
                ("pre_dispatch", "read_lock_acquired", "dispatched"),
                ("worker_dispatch_wait", "dispatched", "native_started"),
                ("native_call", "native_started", "native_finished"),
                ("post_native_cleanup", "native_finished", "owner_completed"),
                ("owner_total", "submitted", "owner_completed"),
                ("response_wait", "request_started", "response_started"),
            )
        }
        native_after_response = None
        if all(
            name in events
            for name in ("response_started", "native_started", "native_finished")
        ):
            native_after_response = (
                max(
                    0,
                    events["native_finished"]
                    - max(events["response_started"], events["native_started"]),
                )
                / 1_000_000_000
            )
        return {
            "label": self.label,
            "events_ns": events,
            "durations_seconds": durations,
            "waiter_detached": "waiter_detached" in events,
            "native_failed": "native_failed" in events,
            "native_result_verified": native_result_verified,
            "owner_failed": "owner_failed" in events,
            "owner_cancelled": "owner_cancelled" in events,
            "native_after_response_seconds": native_after_response,
        }


class ExecutionTimings:
    def __init__(self, labels: set[str], *, clock=time.perf_counter_ns):
        if (
            not labels
            or len(labels) > MAX_TIMELINES
            or any(
                not isinstance(label, str) or not 1 <= len(label) <= 128
                for label in labels
            )
        ):
            raise ValueError("Capacity timeline labels exceed experiment bounds")
        self.labels = frozenset(labels)
        self.clock, self.origin = clock, clock()
        self.records: dict[str, ExecutionTimeline] = {}
        self.current: ContextVar[ExecutionTimeline | None] = ContextVar(
            "capacity_execution_timeline", default=None
        )

    def begin(self) -> ExecutionTimeline | None:
        record = self.current.get()
        if record is not None:
            record.note("submitted")
        return record

    def wrap(self, app):
        async def observed(scope, receive, send):
            label = next(
                (
                    value.decode("ascii", errors="replace")
                    for key, value in scope.get("headers", [])
                    if key == b"x-capacity-id" and len(value) <= 128
                ),
                None,
            )
            if scope["type"] != "http" or label not in self.labels:
                await app(scope, receive, send)
                return
            if label in self.records:
                raise ValueError("Duplicate capacity timeline label")
            record = self.records[label] = ExecutionTimeline(
                label, self.origin, self.clock
            )
            record.note("request_started")
            token = self.current.set(record)

            async def observed_send(message):
                if message["type"] == "http.response.start":
                    record.note("response_started")
                await send(message)

            try:
                await app(scope, receive, observed_send)
            finally:
                record.note("request_completed")
                self.current.reset(token)

        return observed

    def report(self, client_records: list[dict]) -> dict:
        rows = {label: record.snapshot() for label, record in self.records.items()}
        for case in client_records:
            row = rows.get(case["label"])
            if case["passed"] and (
                row is None
                or row["native_result_verified"] is not True
                or row["owner_failed"]
                or row["owner_cancelled"]
                or row["native_failed"]
                or row["waiter_detached"]
                or any(
                    row["durations_seconds"][name] is None
                    for name in (
                        "owner_scheduling",
                        "admission_wait",
                        "read_lock_wait",
                        "pre_dispatch",
                        "worker_dispatch_wait",
                        "native_call",
                        "post_native_cleanup",
                    )
                )
            ):
                raise AssertionError(
                    "Successful capacity request lacks completed native timing"
                )
        return {
            "clock": "perf_counter_ns",
            "scope": "server process wall durations; native callable includes preparation/inference/probe result verification; not kernel or thread CPU time",
            "records": list(rows.values()),
        }
