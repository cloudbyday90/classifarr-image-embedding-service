# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""HTTP deadlines and lifespan teardown with event-controlled worker threads."""

import asyncio
import signal
from unittest.mock import Mock

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fakes import FakeEmbedder, _no_auth_settings
from worker_fakes import GatedEmbedder, ThreadGate, wait_until

from image_embedder.main import create_app

ENDPOINTS = [
    ("/embed-image", {"image_url": "https://example.com/image.png"}),
    ("/embed-batch", {"items": [{"image_url": "https://example.com/image.png"}]}),
]


@pytest.fixture
def anyio_backend():
    return "asyncio"


def gated_app(gate, **overrides):
    settings = _no_auth_settings(
        embedding_timeout_seconds=0.1,
        embed_concurrency=1,
        embed_max_queue=0,
        rate_limit_embed="1000/minute",
        cleanup_on_shutdown=False,
    )
    for name, value in overrides.items():
        setattr(settings, name, value)
    return create_app(embedder=GatedEmbedder(gate), settings=settings)


@pytest.mark.anyio
@pytest.mark.parametrize("path, body", ENDPOINTS)
async def test_http_timeout_retains_real_concurrency_and_retry_rejects_until_worker_finishes(
    path, body
):
    gate = ThreadGate()
    app = gated_app(gate)
    try:
        async with LifespanManager(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                task = asyncio.create_task(client.post(path, json=body))
                await gate.wait_started()
                response = await task
                assert response.status_code == 504
                assert response.headers["x-queue-in-flight"] == "1"
                assert gate.active == app.state.queue.stats().rw_readers == 1
                health = await client.get("/health")
                assert health.json()["queue"]["in_flight"] == 1
                retry = await client.post(path, json=body)
                assert retry.status_code == 429
                assert retry.headers["retry-after"] == "1"
                assert gate.calls == gate.peak == 1
                gate.release.set()
                await wait_until(lambda: app.state.queue.stats().in_flight == 0)
                assert (await client.post(path, json=body)).status_code == 200
                assert gate.peak == 1
    finally:
        gate.release.set()
        await app.state.executor.close(2)


@pytest.mark.anyio
@pytest.mark.parametrize("path, body", ENDPOINTS)
async def test_http_caller_cancellation_does_not_release_running_work(path, body):
    gate = ThreadGate()
    app = gated_app(gate, embedding_timeout_seconds=5)
    try:
        async with LifespanManager(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                task = asyncio.create_task(client.post(path, json=body))
                await gate.wait_started()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert gate.active == app.state.queue.stats().in_flight == 1
                assert (await client.post(path, json=body)).status_code == 429
                assert gate.peak == 1
                gate.release.set()
    finally:
        gate.release.set()
        await app.state.executor.close(2)


@pytest.mark.anyio
@pytest.mark.parametrize("path, body", ENDPOINTS)
async def test_http_expired_queue_admission_is_removed(path, body):
    gate = ThreadGate()
    app = gated_app(gate, embed_max_queue=1)
    try:
        async with LifespanManager(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                first = asyncio.create_task(client.post(path, json=body))
                await gate.wait_started()
                assert (await first).status_code == 504
                waiting = asyncio.create_task(client.post(path, json=body))
                await wait_until(lambda: app.state.queue.stats().waiting == 1)
                assert (await waiting).status_code == 504
                assert app.state.queue.stats().waiting == 0
                assert gate.calls == 1
                gate.release.set()
                await wait_until(lambda: app.state.queue.stats().in_flight == 0)
                assert gate.calls == 1
    finally:
        gate.release.set()
        await app.state.executor.close(2)


@pytest.mark.anyio
@pytest.mark.parametrize("path, body", ENDPOINTS)
async def test_closed_executor_returns_503(path, body):
    app = create_app(embedder=FakeEmbedder(), settings=_no_auth_settings())
    assert await app.state.executor.close(1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(path, json=body)
    assert response.status_code == 503
    assert response.json()["detail"] == "service is shutting down"


@pytest.mark.anyio
async def test_lifespan_drain_precedes_memory_cleanup(monkeypatch):
    gate = ThreadGate()
    app = gated_app(gate, cleanup_on_shutdown=True)
    drain_started = asyncio.Event()
    original_close = app.state.executor.close

    async def close(timeout):
        drain_started.set()
        return await original_close(timeout)

    def cleanup():
        assert gate.active == app.state.queue.stats().in_flight == 0
        return {}

    cleanup_spy = Mock(side_effect=cleanup)
    monkeypatch.setattr(app.state.executor, "close", close)
    monkeypatch.setattr("image_embedder.lifecycle.force_cleanup", cleanup_spy)

    async def release_during_drain():
        await drain_started.wait()
        gate.release.set()

    release_task = asyncio.create_task(release_during_drain())
    try:
        async with LifespanManager(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                assert (
                    await client.post(ENDPOINTS[0][0], json=ENDPOINTS[0][1])
                ).status_code == 504
                assert gate.active == 1
        await release_task
        cleanup_spy.assert_called_once()
    finally:
        gate.release.set()
        release_task.cancel()
        await original_close(2)


@pytest.mark.anyio
async def test_lifespan_skips_gpu_cleanup_when_drain_budget_expires(monkeypatch):
    gate = ThreadGate()
    app = gated_app(gate, cleanup_on_shutdown=True, shutdown_timeout_seconds=0.01)
    cleanup = Mock()
    monkeypatch.setattr("image_embedder.lifecycle.force_cleanup", cleanup)
    try:
        async with LifespanManager(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                assert (
                    await client.post(ENDPOINTS[0][0], json=ENDPOINTS[0][1])
                ).status_code == 504
        cleanup.assert_not_called()
        assert gate.active == app.state.queue.stats().in_flight == 1
        gate.release.set()
        assert await app.state.executor.close(2)
        assert app.state.queue.stats().in_flight == 0
    finally:
        gate.release.set()
        await app.state.executor.close(2)


@pytest.mark.anyio
async def test_lifespan_preserves_signal_handlers_and_tears_down_after_body_error(
    monkeypatch,
):
    app = create_app(
        embedder=FakeEmbedder(), settings=_no_auth_settings(cleanup_on_shutdown=False)
    )
    loop = asyncio.get_running_loop()
    registration = Mock()
    monkeypatch.setattr(loop, "add_signal_handler", registration)
    original = signal.getsignal(signal.SIGTERM)
    with pytest.raises(RuntimeError, match="body error"):
        async with app.router.lifespan_context(app):
            assert signal.getsignal(signal.SIGTERM) == original
            raise RuntimeError("body error")
    registration.assert_not_called()
    assert signal.getsignal(signal.SIGTERM) == original
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (
            await client.post(ENDPOINTS[0][0], json=ENDPOINTS[0][1])
        ).status_code == 503
