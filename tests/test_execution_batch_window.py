# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Batch dispatcher cancellation must not abandon its worker's ownership."""

import asyncio

import httpx2
import pytest
from asgi_lifespan import LifespanManager
from fakes import FakeEmbedder, _no_auth_settings
from worker_fakes import GatedEmbedder, ThreadGate, wait_until

from image_embedder.batch import BatchWindow, EmbedJob
from image_embedder.execution import ExecutionClosedError, InferenceExecutor
from image_embedder.main import create_app
from image_embedder.queue import EmbedQueue


@pytest.fixture
def anyio_backend():
    return "asyncio"


def job():
    return EmbedJob("https://example.com/image.png", None, None, True, None)


@pytest.mark.anyio
async def test_stop_settles_jobs_already_removed_for_collection():
    queue = EmbedQueue(1, 0, 2)
    executor = InferenceExecutor(queue)
    batch = BatchWindow(FakeEmbedder(), queue, 1000, 8, executor=executor)
    await batch.start()
    pending = asyncio.create_task(batch.submit(job()))
    try:
        await wait_until(lambda: len(batch._active) == 1)
        await batch.stop()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 1)
        with pytest.raises(ExecutionClosedError):
            await batch.submit(job())
        assert queue.stats().in_flight == 0
    finally:
        await batch.stop()
        await executor.close(2)


@pytest.mark.anyio
async def test_stopping_dispatcher_keeps_running_worker_permits():
    queue = EmbedQueue(1, 0, 2)
    executor = InferenceExecutor(queue)
    gate = ThreadGate()
    batch = BatchWindow(GatedEmbedder(gate), queue, 0, 8, executor=executor)
    await batch.start()
    pending = asyncio.create_task(batch.submit(job()))
    try:
        await gate.wait_started()
        await batch.stop()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 1)
        assert gate.active == queue.stats().in_flight == queue.stats().rw_readers == 1
        assert not await executor.close(0.01)
        gate.release.set()
        assert await executor.close(2)
        assert queue.stats().in_flight == queue.stats().rw_readers == 0
    finally:
        gate.release.set()
        await batch.stop()
        await executor.close(2)


@pytest.mark.anyio
async def test_canceled_collected_job_is_skipped_before_dispatch():
    queue = EmbedQueue(1, 0, 2)
    gate = ThreadGate()
    batch = BatchWindow(GatedEmbedder(gate), queue, 1000, 2)
    await batch.start()
    first = asyncio.create_task(batch.submit(job()))
    try:
        await wait_until(lambda: len(batch._active) == 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(batch.submit(job()))
        await gate.wait_started()
        gate.release.set()
        assert (await second)[1] == 768
        assert gate.calls == 1
    finally:
        gate.release.set()
        await batch.stop()


@pytest.mark.anyio
async def test_coalesced_http_timeouts_keep_capacity_shared_with_explicit_batch_endpoint():
    gate = ThreadGate()
    app = create_app(
        embedder=GatedEmbedder(gate),
        settings=_no_auth_settings(
            embed_batch_window_ms=10,
            embed_batch_max_size=2,
            embed_concurrency=1,
            embed_max_queue=0,
            embedding_timeout_seconds=0.1,
            cleanup_on_shutdown=False,
        ),
    )
    body = {"image_url": "https://example.com/image.png"}
    try:
        async with LifespanManager(app):
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app), base_url="http://test"
            ) as client:
                requests = [
                    asyncio.create_task(client.post("/embed-image", json=body))
                    for _ in range(2)
                ]
                await gate.wait_started()
                responses = await asyncio.gather(*requests)
                assert [response.status_code for response in responses] == [504, 504]
                assert gate.active == app.state.queue.stats().in_flight == 1
                retry = await client.post("/embed-batch", json={"items": [body]})
                assert retry.status_code == 429
                assert gate.calls == gate.peak == 1
                gate.release.set()
    finally:
        gate.release.set()
        await app.state.executor.close(2)


@pytest.mark.anyio
@pytest.mark.parametrize("blocked_by", ["capacity", "shared_lock"])
async def test_all_clients_cancel_while_group_waits_for_admission(blocked_by):
    queue = EmbedQueue(1, 1, 2)
    executor = InferenceExecutor(queue)
    gate = ThreadGate()
    batch = BatchWindow(GatedEmbedder(gate), queue, 0, 8, executor=executor)
    await batch.start()
    blocker = None
    if blocked_by == "capacity":
        blocker = asyncio.create_task(executor.run(gate.run, lambda: 42))
        await gate.wait_started()
    else:
        await queue.acquire_exclusive()
    request = asyncio.create_task(batch.submit(job()))
    try:
        if blocked_by == "capacity":
            await wait_until(lambda: queue.stats().waiting == 1)
        else:
            await wait_until(lambda: queue.stats().in_flight == 1)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        await wait_until(
            lambda: (
                queue.stats().waiting == 0
                and queue.stats().in_flight == (1 if blocker else 0)
            )
        )
        assert gate.calls == (1 if blocker else 0)
        assert not batch._task.done()
    finally:
        gate.release.set()
        if blocker is not None:
            assert await blocker == 42
        else:
            await queue.release_exclusive()
        await batch.stop()
        assert await executor.close(2)


@pytest.mark.anyio
async def test_one_canceled_client_does_not_cancel_another_client_in_same_group():
    # Queued group members now count individually toward the shared limit.
    queue = EmbedQueue(1, 2, 2)
    executor = InferenceExecutor(queue)
    gate = ThreadGate()
    batch = BatchWindow(GatedEmbedder(gate), queue, 1000, 2, executor=executor)
    await batch.start()
    blocker = asyncio.create_task(executor.run(gate.run, lambda: 42))
    requests = []
    try:
        await gate.wait_started()
        requests = [asyncio.create_task(batch.submit(job())) for _ in range(2)]
        await wait_until(lambda: queue.stats().waiting == 2)
        requests[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await requests[0]
        gate.release.set()
        assert await blocker == 42
        assert (await requests[1])[1] == 768
        assert gate.calls == 2
        assert gate.peak == 1
    finally:
        gate.release.set()
        await batch.stop()
        assert await executor.close(2)


@pytest.mark.anyio
async def test_closing_executor_settles_live_batch_clients_waiting_for_admission():
    queue = EmbedQueue(1, 1, 2)
    executor = InferenceExecutor(queue)
    gate = ThreadGate()
    batch = BatchWindow(FakeEmbedder(), queue, 0, 8, executor=executor)
    await batch.start()
    blocker = asyncio.create_task(executor.run(gate.run, lambda: 42))
    try:
        await gate.wait_started()
        request = asyncio.create_task(batch.submit(job()))
        await wait_until(lambda: queue.stats().waiting == 1)
        assert not await executor.close(0.01)
        with pytest.raises(ExecutionClosedError):
            await asyncio.wait_for(request, 1)
        assert queue.stats().waiting == 0
        assert queue.stats().in_flight == 1
        gate.release.set()
        assert await blocker == 42
    finally:
        gate.release.set()
        await batch.stop()
        await executor.close(2)
