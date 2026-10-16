# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import asyncio

import pytest
from capacity_execution_timing import MAX_TIMELINES, ExecutionTimeline, ExecutionTimings


def test_intervals_use_one_wall_clock_and_retain_detached_native_tail():
    ticks = iter(range(20))
    row = ExecutionTimeline("fixed", 0, lambda: next(ticks) * 1_000_000_000)
    for phase in (
        "request_started",
        "submitted",
        "admission_started",
        "admitted",
        "read_lock_acquired",
        "dispatched",
        "native_started",
        "waiter_detached",
        "response_started",
        "request_completed",
        "native_finished",
        "owner_completed",
    ):
        if phase == "native_finished":
            row.verify_native_result(True)
        row.note(phase)
    result = row.snapshot()
    assert result["durations_seconds"]["admission_wait"] == 1
    assert result["durations_seconds"]["native_call"] == 4
    assert result["durations_seconds"]["owner_total"] == 10
    assert result["native_after_response_seconds"] == 2
    assert result["waiter_detached"] and not result["native_failed"]
    assert result["native_result_verified"] is True
    assert (
        sum(
            result["durations_seconds"][name]
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
        == 10
    )


@pytest.mark.parametrize(
    "phases",
    [
        ["request_started"],
        ["request_started", "submitted", "request_completed"],
        [
            "request_started",
            "submitted",
            "native_started",
            "request_completed",
            "owner_completed",
        ],
    ],
)
def test_missing_completion_is_not_a_zero_duration(phases):
    row = ExecutionTimeline("fixed", 0, lambda: 1)
    for phase in phases:
        row.note(phase)
    with pytest.raises(AssertionError, match="did not settle"):
        row.snapshot()


@pytest.mark.parametrize(
    "ticks,phases",
    [([2, 1], ["submitted", "native_started"]), ([1, 2], ["submitted", "submitted"])],
)
def test_duplicate_or_backwards_events_fail_closed(ticks, phases):
    counter = iter(ticks)
    row = ExecutionTimeline("fixed", 0, lambda: next(counter))
    row.note(phases[0])
    with pytest.raises(ValueError):
        row.note(phases[1])
    with pytest.raises(AssertionError):
        row.snapshot()


def test_labels_are_bounded_and_unknown_requests_have_no_context():
    with pytest.raises(ValueError):
        ExecutionTimings(set())
    with pytest.raises(ValueError):
        ExecutionTimings({"x" * 129})
    with pytest.raises(ValueError):
        ExecutionTimings({str(i) for i in range(MAX_TIMELINES + 1)})
    timings = ExecutionTimings({"known"})

    async def app(scope, receive, send):
        assert timings.begin() is None

    asyncio.run(
        timings.wrap(app)(
            {"type": "http", "headers": [(b"x-capacity-id", b"unknown")]}, None, None
        )
    )
    assert timings.records == {} and timings.current.get() is None


def test_success_without_native_evidence_is_rejected_and_context_is_reset():
    timings = ExecutionTimings({"known"})

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})

    async def send(_):
        pass

    asyncio.run(
        timings.wrap(app)(
            {"type": "http", "headers": [(b"x-capacity-id", b"known")]}, None, send
        )
    )
    assert timings.current.get() is None
    with pytest.raises(AssertionError, match="lacks completed native timing"):
        timings.report([{"label": "known", "passed": True}])
    row = timings.report([{"label": "known", "passed": False}])["records"][0]
    assert row["durations_seconds"]["native_call"] is None


def test_native_completion_without_outcome_validation_is_not_success_evidence():
    row = ExecutionTimeline("fixed", 0, lambda: 1)
    for phase in [
        "request_started",
        "submitted",
        "native_started",
        "native_finished",
        "owner_completed",
        "request_completed",
    ]:
        row.note(phase)
    with pytest.raises(AssertionError, match="native result was not verified"):
        row.snapshot()


def test_native_outcome_cannot_be_recorded_outside_its_call():
    row = ExecutionTimeline("fixed", 0, lambda: 1)
    with pytest.raises(ValueError, match="Invalid native result"):
        row.verify_native_result(True)
