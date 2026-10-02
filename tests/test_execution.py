# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Execution accounting under cancellation, worker errors, and shutdown."""

import asyncio
from unittest.mock import Mock

import pytest
from worker_fakes import ThreadGate, wait_until

from image_embedder.execution import ExecutionClosedError, InferenceExecutor
from image_embedder.queue import EmbedQueue, QueueFullError


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_executor(max_queue=0):
    queue = EmbedQueue(concurrency=1, max_queue=max_queue, max_wait_seconds=2)
    return queue, InferenceExecutor(queue)


@pytest.mark.anyio
async def test_execution_returns_values_and_releases_permits():
    queue, executor = make_executor()
    assert (
        await executor.run(
            lambda value, *, increment: value + increment, 40, increment=2
        )
        == 42
    )
    assert queue.stats().in_flight == queue.stats().rw_readers == 0
    assert await executor.close(1)


@pytest.mark.anyio
async def test_execution_propagates_worker_error_and_releases_capacity():
    queue, executor = make_executor()

    def fail():
        raise ValueError("worker failure")

    with pytest.raises(ValueError, match="worker failure"):
        await executor.run(fail)
    assert queue.stats().in_flight == queue.stats().rw_readers == 0
    assert await executor.run(lambda: 42) == 42
    assert await executor.close(1)


@pytest.mark.anyio
@pytest.mark.parametrize("cancellation", ["deadline", "caller"])
async def test_dispatched_worker_keeps_capacity_after_request_cancellation(
    cancellation,
):
    queue, executor = make_executor()
    gate = ThreadGate()
    task = asyncio.create_task(executor.run(gate.run, lambda: 42))
    try:
        await gate.wait_started()
        if cancellation == "deadline":
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(task, timeout=0.02)
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert gate.active == queue.stats().in_flight == queue.stats().rw_readers == 1
        with pytest.raises(QueueFullError):
            await executor.run(gate.run, lambda: 43)
        assert gate.calls == gate.peak == 1

        gate.release.set()
        await wait_until(lambda: queue.stats().in_flight == 0)
        assert queue.stats().rw_readers == gate.active == 0
        assert await executor.run(lambda: 44) == 44
    finally:
        gate.release.set()
        assert await executor.close(2)


@pytest.mark.anyio
@pytest.mark.parametrize("cancellation", ["deadline", "caller"])
async def test_canceled_admission_never_runs_and_leaves_no_waiter(cancellation):
    queue, executor = make_executor(max_queue=1)
    gate = ThreadGate()
    first = asyncio.create_task(executor.run(gate.run, lambda: 42))
    later_calls = []
    try:
        await gate.wait_started()
        waiter = asyncio.create_task(executor.run(later_calls.append, "expired"))
        await wait_until(lambda: queue.stats().waiting == 1)
        if cancellation == "deadline":
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(waiter, timeout=0.02)
        else:
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        assert queue.stats().waiting == 0
        assert queue.stats().in_flight == 1
        gate.release.set()
        assert await first == 42
        assert later_calls == []
        assert await executor.run(lambda: 43) == 43
    finally:
        gate.release.set()
        assert await executor.close(2)


@pytest.mark.anyio
async def test_cancellation_while_waiting_for_shared_lock_releases_partial_acquisition():
    queue, executor = make_executor()
    await queue.acquire_exclusive()
    calls = []
    task = asyncio.create_task(executor.run(calls.append, "unexpected"))
    try:
        await wait_until(lambda: queue.stats().in_flight == 1)
        assert queue.stats().rw_readers == 0
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert queue.stats().in_flight == 0
        assert calls == []
    finally:
        await queue.release_exclusive()
        assert await executor.close(2)


@pytest.mark.anyio
async def test_detached_failure_is_observed_without_logging_sensitive_exception_text(
    monkeypatch,
):
    queue, executor = make_executor()
    gate = ThreadGate()
    warning = Mock()
    monkeypatch.setattr("image_embedder.execution.logger.warning", warning)
    loop_errors = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

    def fail():
        raise RuntimeError("sensitive remote payload")

    task = asyncio.create_task(executor.run(gate.run, fail))
    try:
        await gate.wait_started()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate.release.set()
        assert await executor.close(2)
        assert queue.stats().in_flight == 0
        warning.assert_called_once_with(
            "Detached inference failed (%s)", "RuntimeError"
        )
        assert loop_errors == []
    finally:
        gate.release.set()
        await executor.close(2)
        loop.set_exception_handler(previous_handler)


@pytest.mark.anyio
async def test_close_cancels_admission_drains_dispatched_work_and_rejects_new_work():
    queue, executor = make_executor(max_queue=1)
    gate = ThreadGate()
    first = asyncio.create_task(executor.run(gate.run, lambda: 42))
    calls = []
    try:
        await gate.wait_started()
        waiter = asyncio.create_task(executor.run(calls.append, "unexpected"))
        await wait_until(lambda: queue.stats().waiting == 1)
        closing = asyncio.create_task(executor.close(2))
        with pytest.raises(ExecutionClosedError):
            await waiter
        assert queue.stats().waiting == 0
        assert not closing.done()
        with pytest.raises(ExecutionClosedError):
            await executor.run(lambda: 0)
        gate.release.set()
        assert await first == 42
        assert await closing
        assert calls == []
        assert queue.stats().in_flight == queue.stats().rw_readers == 0
        assert await executor.close(0)
    finally:
        gate.release.set()
        await executor.close(2)


@pytest.mark.anyio
async def test_drain_timeout_keeps_running_worker_and_permits_owned():
    queue, executor = make_executor()
    gate = ThreadGate()
    task = asyncio.create_task(executor.run(gate.run, lambda: 42))
    try:
        await gate.wait_started()
        assert not await executor.close(0.01)
        assert not task.done()
        assert gate.active == queue.stats().in_flight == queue.stats().rw_readers == 1
        with pytest.raises(ExecutionClosedError):
            await executor.run(lambda: 43)
        gate.release.set()
        assert await task == 42
        assert await executor.close(2)
        assert queue.stats().in_flight == 0
    finally:
        gate.release.set()
        await executor.close(2)


@pytest.mark.anyio
async def test_caller_cancellation_and_shutdown_do_not_interrupt_admission_cleanup():
    queue, executor = make_executor()
    await queue.acquire_exclusive()
    calls = []
    requester = asyncio.create_task(executor.run(calls.append, "unexpected"))
    closing = None
    try:
        await wait_until(lambda: queue.stats().in_flight == 1)
        owner = next(iter(executor._tasks))
        async with queue._cond:
            requester.cancel()
            await wait_until(lambda: owner.cancelling() == 1)
            closing = asyncio.create_task(executor.close(2))
            # Let close inspect the owner while its release is contended.
            await asyncio.sleep(0)
            assert owner.cancelling() == 1
            assert not owner.done()
        with pytest.raises(asyncio.CancelledError):
            await requester
        assert await closing
        assert calls == []
        assert queue.stats().in_flight == queue.stats().waiting == 0
    finally:
        await queue.release_exclusive()
        await executor.close(2)
        if closing is not None:
            await closing


@pytest.mark.anyio
async def test_immediate_close_before_owner_starts_returns_service_closed_error():
    queue, executor = make_executor()
    calls = []
    requester = asyncio.create_task(executor.run(calls.append, "unexpected"))
    closing = asyncio.create_task(executor.close(1))
    with pytest.raises(ExecutionClosedError):
        await requester
    assert await closing
    assert calls == []
    assert queue.stats().in_flight == queue.stats().waiting == 0
