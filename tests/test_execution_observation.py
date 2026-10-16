# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Observers preserve detached ownership, reservation and failure contracts."""

import asyncio
from contextvars import ContextVar
from unittest.mock import Mock

import pytest
from worker_fakes import ThreadGate, wait_until

from image_embedder.execution import InferenceExecutor
from image_embedder.queue import EmbedQueue


@pytest.fixture
def anyio_backend():
    return "asyncio"


class Observer:
    def __init__(self):
        self.context = ContextVar("test_execution_observer", default=None)
        self.records = {}

    def begin(self):
        label = self.context.get()
        self.records[label] = []
        return self

    def mark(self, phase):
        self.records[self.context.get()].append(phase)


@pytest.mark.anyio
async def test_queued_calls_capture_context_and_separate_native_dispatch():
    queue = EmbedQueue(1, 1, 2)
    observer = Observer()
    executor = InferenceExecutor(queue, observer=observer)
    gate = ThreadGate()
    observer.context.set("first")
    first = asyncio.create_task(executor.run(gate.run, lambda: 41))
    try:
        await gate.wait_started()
        observer.context.set("second")
        second = asyncio.create_task(executor.run(lambda: 42))
        await wait_until(lambda: queue.stats().waiting == 1)
        assert observer.records["second"] == ["admission_started"]
        gate.release.set()
        assert await first == 41 and await second == 42
        assert await executor.close(2)
        expected = [
            "admission_started",
            "admitted",
            "read_lock_acquired",
            "dispatched",
            "native_started",
            "native_finished",
            "owner_completed",
        ]
        assert observer.records == {"first": expected, "second": expected}
        assert queue.stats().in_flight == queue.stats().rw_readers == 0
    finally:
        gate.release.set()
        await executor.close(2)


@pytest.mark.anyio
async def test_detached_native_failure_is_recorded_after_waiter_leaves():
    queue, observer, gate = EmbedQueue(1, 0, 2), Observer(), ThreadGate()
    executor = InferenceExecutor(queue, observer=observer)

    def fail():
        raise ValueError("private native arguments")

    task = asyncio.create_task(executor.run(gate.run, fail))
    try:
        await gate.wait_started()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert observer.records[None][-1] == "waiter_detached"
        assert queue.stats().in_flight == queue.stats().rw_readers == 1
        gate.release.set()
        assert await executor.close(2)
        assert observer.records[None][-4:] == [
            "native_failed",
            "native_finished",
            "owner_failed",
            "owner_completed",
        ]
        assert "private" not in str(observer.records)
        assert queue.stats().in_flight == queue.stats().rw_readers == 0
    finally:
        gate.release.set()
        await executor.close(2)


@pytest.mark.anyio
async def test_admission_cancellation_records_no_native_call():
    queue, observer = EmbedQueue(1, 1, 2), Observer()
    executor = InferenceExecutor(queue, observer=observer)
    await queue.acquire()
    task = asyncio.create_task(
        executor.run(lambda: pytest.fail("Canceled admission ran"))
    )
    try:
        await wait_until(lambda: queue.stats().waiting == 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await executor.close(1)
        assert observer.records[None] == [
            "admission_started",
            "waiter_detached",
            "owner_cancelled",
            "owner_completed",
        ]
        assert queue.stats().waiting == 0
    finally:
        await queue.release()


@pytest.mark.anyio
@pytest.mark.parametrize("fault", ["begin", "mark"])
async def test_observer_failure_cannot_change_result_error_or_capacity(
    fault, monkeypatch
):
    warning = Mock()
    monkeypatch.setattr("image_embedder.execution_observation.logger.warning", warning)
    observer = Observer()
    monkeypatch.setattr(
        observer, fault, Mock(side_effect=RuntimeError("private observer data"))
    )
    queue = EmbedQueue(1, 0, 2)
    executor = InferenceExecutor(queue, observer=observer)
    assert await executor.run(lambda: 42) == 42

    def fail():
        raise ValueError("original failure")

    with pytest.raises(ValueError, match="original failure"):
        await executor.run(fail)
    assert await executor.close(1)
    assert queue.stats().in_flight == queue.stats().rw_readers == 0
    assert warning.called and "private" not in str(warning.call_args_list)


@pytest.mark.anyio
async def test_reserved_ticket_observation_and_preparation_stay_with_owner():
    queue, observer = EmbedQueue(1, 0, 2), Observer()
    executor = InferenceExecutor(queue, observer=observer)
    ticket = queue.reserve()
    assert await executor.run_admitted(ticket, lambda: lambda: 42) == 42
    assert await executor.close(1)
    assert observer.records[None][-1] == "owner_completed"
    assert queue.stats().in_flight == queue.stats().rw_readers == 0


@pytest.mark.anyio
async def test_read_lock_wait_is_distinct_from_admission_and_native_start():
    queue, observer = EmbedQueue(1, 0, 2), Observer()
    executor = InferenceExecutor(queue, observer=observer)
    await queue.acquire_exclusive()
    task = asyncio.create_task(executor.run(lambda: 42))
    try:
        await wait_until(lambda: queue.stats().in_flight == 1)
        assert observer.records[None] == ["admission_started", "admitted"]
    finally:
        await queue.release_exclusive()
    assert await task == 42
    assert await executor.close(1)
    assert observer.records[None].index("read_lock_acquired") < observer.records[
        None
    ].index("native_started")


@pytest.mark.anyio
async def test_worker_limiter_wait_and_detachment_do_not_release_owner():
    from anyio.to_thread import current_default_thread_limiter

    limiter = current_default_thread_limiter()
    original_tokens = limiter.total_tokens
    borrower = object()
    limiter.total_tokens = 1
    await limiter.acquire_on_behalf_of(borrower)
    queue, observer = EmbedQueue(1, 0, 2), Observer()
    executor = InferenceExecutor(queue, observer=observer)
    task = asyncio.create_task(executor.run(lambda: 42))
    try:
        await wait_until(lambda: "dispatched" in observer.records.get(None, []))
        assert "native_started" not in observer.records[None]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert queue.stats().in_flight == queue.stats().rw_readers == 1
    finally:
        limiter.release_on_behalf_of(borrower)
        limiter.total_tokens = original_tokens
        assert await executor.close(2)
    assert "native_started" in observer.records[None]
    assert observer.records[None][-1] == "owner_completed"
    assert queue.stats().in_flight == queue.stats().rw_readers == 0
