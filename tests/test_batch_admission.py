# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bounded coalescer retention, live payloads, and cross-endpoint fairness."""

import asyncio

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fakes import FakeEmbedder, _no_auth_settings
from worker_fakes import GatedEmbedder, ThreadGate, wait_until

from image_embedder.batch import BatchWindow, EmbedJob
from image_embedder.execution import InferenceExecutor
from image_embedder.main import create_app
from image_embedder.queue import EmbedQueue, QueueFullError, QueueWaitTimeoutError


@pytest.fixture
def anyio_backend():
    return "asyncio"


def job(index=1, model=None, normalize=True):
    return EmbedJob(f"https://example.com/{index}", None, model, normalize, None)


class RecordingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = []

    @staticmethod
    def result(url, model, size):
        index = url.rsplit("/", 1)[1]
        if index == "error":
            raise ValueError("invalid test image")
        return [float(index)], 1, "local", model, size

    def embed(self, image_url, image_base64, model, normalize, image_size):
        self.calls.append((model or "ViT-L-14", [(image_url, normalize)]))
        return self.result(image_url, model or "ViT-L-14", image_size or 224)

    def embed_batch(self, spec, size, items):
        self.calls.append(
            (spec.name, [(item.image_url, item.normalize) for item in items])
        )
        results = []
        for item in items:
            try:
                results.append(self.result(item.image_url, spec.name, size))
            except ValueError as error:
                results.append(error)
        return results


@pytest.mark.anyio
async def test_no_queue_allows_compatible_collection_but_rejects_another_group():
    queue = EmbedQueue(1, 0, 2)
    embedder = RecordingEmbedder()
    batch = BatchWindow(embedder, queue, 1000, 2)
    await batch.start()
    first = asyncio.create_task(batch.submit(job(1)))
    try:
        await wait_until(lambda: len(batch._active) == 1)
        assert queue.stats().in_flight == 1
        assert queue.stats().waiting == 0
        with pytest.raises(QueueFullError):
            await batch.submit(job(3, "ViT-B-16"))
        second = asyncio.create_task(batch.submit(job(2, normalize=False)))
        assert (await first)[0] == [1.0]
        assert (await second)[0] == [2.0]
        assert embedder.calls == [
            ("ViT-L-14", [(job(1).image_url, True), (job(2).image_url, False)])
        ]
        assert queue.stats().in_flight == 0
    finally:
        await batch.stop()


@pytest.mark.anyio
async def test_finite_waiting_room_charges_members_and_cancellation_reopens_it():
    queue = EmbedQueue(1, 2, 2)
    executor = InferenceExecutor(queue)
    gate = ThreadGate()
    batch = BatchWindow(FakeEmbedder(), queue, 1000, 8, executor=executor)
    await batch.start()
    blocker = asyncio.create_task(executor.run(gate.run, lambda: 42))
    requests = []
    try:
        await gate.wait_started()
        requests = [asyncio.create_task(batch.submit(job(i))) for i in (1, 2)]
        await wait_until(lambda: queue.stats().waiting == 2)
        rejected = job(3)
        with pytest.raises(QueueFullError):
            await batch.submit(rejected)
        assert rejected._future is None
        assert len(batch._active) == 2
        requests[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await requests[0]
        assert queue.stats().waiting == 1
        replacement = asyncio.create_task(batch.submit(job(4)))
        requests.append(replacement)
        await wait_until(lambda: queue.stats().waiting == 2)
    finally:
        gate.release.set()
        await blocker
        await batch.stop()
        await asyncio.gather(*requests, return_exceptions=True)
        assert await executor.close(2)
    assert queue.stats().waiting == queue.stats().in_flight == 0
    assert not batch._groups


@pytest.mark.anyio
async def test_queue_deadline_expires_before_a_long_collection_window():
    queue = EmbedQueue(1, 2, 0.02)
    executor = InferenceExecutor(queue)
    gate = ThreadGate()
    batch = BatchWindow(FakeEmbedder(), queue, 1000, 8, executor=executor)
    await batch.start()
    blocker = asyncio.create_task(executor.run(gate.run, lambda: 42))
    try:
        await gate.wait_started()
        with pytest.raises(QueueWaitTimeoutError):
            await asyncio.wait_for(batch.submit(job()), 1)
        assert queue.stats().waiting == 0
        assert len(batch._active) == 0
        assert gate.calls == 1
    finally:
        gate.release.set()
        await blocker
        await batch.stop()
        await executor.close(2)


@pytest.mark.anyio
async def test_last_collection_client_departure_removes_payload_and_reserved_capacity():
    queue = EmbedQueue(1, 0, 2)
    batch = BatchWindow(FakeEmbedder(), queue, 1000, 8)
    await batch.start()
    request = asyncio.create_task(batch.submit(job()))
    await wait_until(lambda: len(batch._active) == 1)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert not batch._active
    assert not batch._open
    assert queue.stats().in_flight == 0
    await batch.stop()
    assert not batch._groups


@pytest.mark.anyio
async def test_immediate_stop_before_group_owner_starts_settles_admission_and_future():
    queue = EmbedQueue(1, 0, 2)
    batch = BatchWindow(FakeEmbedder(), queue, 1000, 8)
    await batch.start()
    request = asyncio.create_task(batch.submit(job()))
    stopping = asyncio.create_task(batch.stop())
    with pytest.raises(asyncio.CancelledError):
        await request
    await stopping
    assert not batch._groups
    assert queue.stats().in_flight == queue.stats().waiting == 0


@pytest.mark.anyio
async def test_expired_payload_removed_after_shared_lock_wait_with_result_order_preserved():
    queue = EmbedQueue(1, 2, 2)
    executor = InferenceExecutor(queue)
    embedder = RecordingEmbedder()
    batch = BatchWindow(embedder, queue, 1000, 3, executor=executor)
    await batch.start()
    await queue.acquire_exclusive()
    requests = [asyncio.create_task(batch.submit(job(i))) for i in (1, 2, 3)]
    try:
        await wait_until(lambda: len(executor._tasks) == 1)
        requests[1].cancel()
        with pytest.raises(asyncio.CancelledError):
            await requests[1]
        assert len(batch._active) == 2
        await queue.release_exclusive()
        assert (await requests[0])[0] == [1.0]
        assert (await requests[2])[0] == [3.0]
        assert embedder.calls == [
            ("ViT-L-14", [(job(1).image_url, True), (job(3).image_url, True)])
        ]
    finally:
        if queue.stats().rw_writer:
            await queue.release_exclusive()
        await batch.stop()
        await executor.close(2)


@pytest.mark.anyio
async def test_grouping_and_per_item_error_association_remain_stable():
    queue = EmbedQueue(2, 2, 2)
    embedder = RecordingEmbedder()
    batch = BatchWindow(embedder, queue, 10, 2)
    await batch.start()
    try:
        results = await asyncio.gather(
            batch.submit(job("error")),
            batch.submit(job(2)),
            batch.submit(job(3, "ViT-B-16")),
            batch.submit(job(4, "ViT-B-16")),
            return_exceptions=True,
        )
        assert isinstance(results[0], ValueError)
        assert results[1][0] == [2.0]
        assert results[2][0] == [3.0]
        assert results[3][0] == [4.0]
        assert {call[0] for call in embedder.calls} == {"ViT-L-14", "ViT-B-16"}
    finally:
        await batch.stop()


@pytest.mark.anyio
async def test_explicit_submission_ahead_of_window_group_keeps_fifo_priority():
    queue = EmbedQueue(1, 3, 2)
    executor = InferenceExecutor(queue)
    gate = ThreadGate()
    embedder = RecordingEmbedder()
    batch = BatchWindow(embedder, queue, 10, 2, executor=executor)
    await batch.start()
    blocker = asyncio.create_task(executor.run(gate.run, lambda: 42))
    try:
        await gate.wait_started()
        spec = embedder.resolve_model(None)
        from image_embedder.embedder import BatchItem

        explicit = asyncio.create_task(
            executor.run(
                embedder.embed_batch,
                spec,
                224,
                [BatchItem(job(9).image_url, None, True)],
            )
        )
        await wait_until(lambda: queue.stats().waiting == 1)
        requests = [asyncio.create_task(batch.submit(job(i))) for i in (1, 2)]
        await wait_until(lambda: queue.stats().waiting == 3)
        gate.release.set()
        await asyncio.gather(blocker, explicit, *requests)
        assert [call[1][0][0] for call in embedder.calls] == [
            job(9).image_url,
            job(1).image_url,
        ]
        assert queue.stats().waiting == queue.stats().in_flight == 0
    finally:
        gate.release.set()
        await batch.stop()
        await executor.close(2)


@pytest.mark.anyio
async def test_http_burst_is_bounded_and_pending_is_visible_in_headers_and_health():
    gate = ThreadGate()
    app = create_app(
        embedder=GatedEmbedder(gate),
        settings=_no_auth_settings(
            embed_batch_window_ms=100,
            embed_batch_max_size=2,
            embed_concurrency=1,
            embed_max_queue=3,
            max_http_requests=32,  # Exercise inference admission independently of ingress.
            embed_max_wait_seconds=2,
            rate_limit_embed="1000/minute",
            embedding_timeout_seconds=2,
            cleanup_on_shutdown=False,
        ),
    )
    requests = []
    try:
        async with LifespanManager(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                requests = [
                    asyncio.create_task(
                        client.post(
                            "/embed-image", json={"image_url": job(i).image_url}
                        )
                    )
                    for i in range(20)
                ]
                await gate.wait_started()
                await wait_until(lambda: sum(task.done() for task in requests) == 15)
                assert len(app.state.batch_window._active) == 5
                rejected = [task.result() for task in requests if task.done()]
                assert all(response.status_code == 429 for response in rejected)
                assert all(
                    response.headers["x-queue-waiting"] == "3" for response in rejected
                )
                assert all(
                    response.headers["retry-after"] == "1" for response in rejected
                )
                health = (await client.get("/health")).json()
                assert health["queue"]["waiting"] == 3
                assert health["queue"]["in_flight"] == 1
                explicit = await client.post(
                    "/embed-batch", json={"items": [{"image_url": job(30).image_url}]}
                )
                assert explicit.status_code == 429
                gate.release.set()
                responses = await asyncio.gather(*requests)
                assert sum(response.status_code == 200 for response in responses) == 5
                assert gate.peak == 1
                assert (
                    app.state.queue.stats().waiting
                    == app.state.queue.stats().in_flight
                    == 0
                )
    finally:
        gate.release.set()
        await asyncio.gather(*requests, return_exceptions=True)
        await app.state.executor.close(2)


@pytest.mark.anyio
async def test_http_queue_deadline_maps_504_without_dispatching_expired_payload():
    gate = ThreadGate()
    app = create_app(
        embedder=GatedEmbedder(gate),
        settings=_no_auth_settings(
            embed_batch_window_ms=1000,
            embed_batch_max_size=2,
            embed_concurrency=1,
            embed_max_queue=1,
            embed_max_wait_seconds=0.02,
            embedding_timeout_seconds=2,
            cleanup_on_shutdown=False,
        ),
    )
    blocker = None
    try:
        async with LifespanManager(app):
            blocker = asyncio.create_task(app.state.executor.run(gate.run, lambda: 42))
            await gate.wait_started()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/embed-image", json={"image_url": job().image_url}
                )
                assert response.status_code == 504
                assert "waiting for a slot" in response.json()["detail"]
                assert response.headers["x-queue-waiting"] == "0"
                assert not app.state.batch_window._active
                assert gate.calls == 1
                gate.release.set()
                await blocker
    finally:
        gate.release.set()
        if blocker is not None:
            await blocker
        await app.state.executor.close(2)
