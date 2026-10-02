# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared queue budgets, FIFO scheduling, deadlines, and ticket ownership."""

import asyncio

import pytest

from image_embedder.execution import ExecutionClosedError, InferenceExecutor
from image_embedder.queue import EmbedQueue, QueueFullError, QueueWaitTimeoutError


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_waiting_members_share_global_limit_and_ready_members_share_one_slot():
    queue = EmbedQueue(1, 2, 2)
    first = queue.reserve()
    first.add_member()
    assert queue.stats().in_flight == 1
    assert queue.stats().waiting == 0
    second = queue.reserve()
    second.add_member()
    assert queue.stats().waiting == 2
    with pytest.raises(QueueFullError):
        second.add_member()
    with pytest.raises(QueueFullError):
        queue.reserve()
    second.remove_member()
    assert queue.stats().waiting == 1
    first.discard()
    await second.wait()
    assert queue.stats().in_flight == 1
    assert queue.stats().waiting == 0
    second.discard()
    assert queue.stats().in_flight == 0


@pytest.mark.anyio
async def test_fifo_promotion_prevents_new_submissions_bypassing_existing_waiters():
    queue = EmbedQueue(1, 3, 2)
    first, second, third = [queue.reserve() for _ in range(3)]
    first.discard()
    newcomer = queue.reserve()
    assert second.admitted
    assert not third.admitted
    assert not newcomer.admitted
    second.discard()
    assert third.admitted
    third.discard()
    assert newcomer.admitted
    newcomer.discard()
    assert queue.stats().waiting == queue.stats().in_flight == 0


@pytest.mark.anyio
async def test_canceled_waiter_is_removed_without_leaking_a_promoted_slot():
    queue = EmbedQueue(1, 1, 2)
    first, second = queue.reserve(), queue.reserve()
    waiter = asyncio.create_task(second.wait())
    await asyncio.sleep(0)
    waiter.cancel()
    # Release before the canceled waiter's cleanup gets a turn.
    first.discard()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert second.finished
    assert queue.stats().waiting == queue.stats().in_flight == 0


@pytest.mark.anyio
async def test_departed_ready_members_cannot_release_a_transferred_worker_permit():
    queue = EmbedQueue(1, 0, 2)
    ticket = queue.reserve()
    ticket.claim()
    ticket.remove_member()
    assert ticket.members == 0
    assert queue.stats().in_flight == 1
    ticket.discard()
    assert queue.stats().in_flight == 1
    await queue.release_admission(ticket)
    assert queue.stats().in_flight == 0
    with pytest.raises(RuntimeError):
        ticket.remove_member()
    with pytest.raises(RuntimeError):
        ticket.claim()
    with pytest.raises(RuntimeError):
        ticket.add_member()


@pytest.mark.anyio
async def test_expired_waiter_releases_its_weight_and_can_never_be_promoted():
    queue = EmbedQueue(1, 2, 0.01)
    first, second = queue.reserve(), queue.reserve()
    second.add_member()
    with pytest.raises(QueueWaitTimeoutError):
        await second.wait()
    assert queue.stats().waiting == 0
    assert second.finished
    first.discard()
    assert not second.admitted
    assert queue.stats().in_flight == 0


@pytest.mark.anyio
async def test_promotion_checks_absolute_deadline_even_before_timer_callback_runs():
    queue = EmbedQueue(1, 1, 2)
    first, second = queue.reserve(), queue.reserve()
    second._deadline = asyncio.get_running_loop().time() - 1
    first.discard()
    with pytest.raises(QueueWaitTimeoutError):
        await second.wait()
    assert queue.stats().in_flight == queue.stats().waiting == 0


@pytest.mark.anyio
async def test_close_rejects_joins_and_reservations_with_fresh_errors_and_settles_waiters():
    queue = EmbedQueue(1, 2, 2)
    first, second = queue.reserve(), queue.reserve()
    queue.close_admission(ExecutionClosedError("service is shutting down"))
    with pytest.raises(ExecutionClosedError):
        await second.wait()
    errors = []
    for _ in range(2):
        with pytest.raises(ExecutionClosedError) as error:
            queue.reserve()
        errors.append(error.value)
    assert errors[0] is not errors[1]
    with pytest.raises(ExecutionClosedError):
        first.add_member()
    assert queue.stats().waiting == 0
    assert queue.stats().in_flight == 1
    first.discard()
    assert queue.stats().in_flight == 0
    queue.close_admission(ExecutionClosedError("again"))


@pytest.mark.anyio
async def test_no_queue_and_zero_wait_are_explicit_fail_fast_contracts():
    no_queue = EmbedQueue(1, 0, 2)
    first = no_queue.reserve()
    with pytest.raises(QueueFullError):
        no_queue.reserve()
    first.discard()
    zero_wait = EmbedQueue(1, 1, 0)
    first = zero_wait.reserve()
    with pytest.raises(QueueWaitTimeoutError):
        zero_wait.reserve()
    first.discard()


@pytest.mark.anyio
async def test_foreign_tickets_are_rejected_before_transfer():
    first, second = EmbedQueue(1, 0, 2), EmbedQueue(1, 0, 2)
    ticket = first.reserve()
    executor = InferenceExecutor(second)
    with pytest.raises(ValueError):
        await executor.run_admitted(ticket, lambda: lambda: 1)
    with pytest.raises(RuntimeError):
        await second.release_admission(ticket)
    assert first.stats().in_flight == 1
    assert not ticket.claimed
    ticket.discard()
    assert await executor.close(0)


@pytest.mark.anyio
async def test_last_queued_member_departure_frees_waiting_room_immediately():
    queue = EmbedQueue(1, 1, 2)
    first, second = queue.reserve(), queue.reserve()
    second.remove_member()
    assert queue.stats().waiting == 0
    newcomer = queue.reserve()
    first.discard()
    await newcomer.wait()
    newcomer.discard()
    assert queue.stats().in_flight == queue.stats().waiting == 0
