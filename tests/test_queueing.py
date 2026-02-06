# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import asyncio
import time
import threading

import httpx
import pytest

from image_embedder.main import create_app


class SlowEmbedder:
    def __init__(self, delay_seconds: float, started_event: threading.Event | None = None):
        self._delay_seconds = delay_seconds
        self._started_event = started_event

    def list_models(self):
        class Spec:
            def __init__(self, name, dims, image_size):
                self.name = name
                self.dims = dims
                self.image_size = image_size

        return [Spec("ViT-L-14", 768, 224)]

    def embed(self, image_url, image_base64, model, normalize, image_size):
        if self._started_event is not None:
            self._started_event.set()
        time.sleep(self._delay_seconds)
        return [0.1, 0.2], 2, "local", model or "ViT-L-14", image_size or 224


@pytest.mark.anyio
async def test_queue_serializes_when_waiting_allowed(monkeypatch):
    monkeypatch.setenv("IMAGE_EMBEDDER_CONCURRENCY", "1")
    monkeypatch.setenv("IMAGE_EMBEDDER_MAX_QUEUE", "10")
    monkeypatch.setenv("IMAGE_EMBEDDER_MAX_WAIT_SECONDS", "2")

    app = create_app(embedder=SlowEmbedder(delay_seconds=0.20))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async def call():
            return await client.post("/embed-image", json={"image_base64": "AA==", "model": "ViT-L-14"})

        t0 = time.perf_counter()
        r1, r2 = await asyncio.gather(call(), call())
        dt = time.perf_counter() - t0

    assert r1.status_code == 200
    assert r2.status_code == 200
    # With concurrency=1 and 2x 200ms embeds, wall time should be ~400ms or more.
    assert dt >= 0.35
    assert "X-Queue-Concurrency" in r1.headers


@pytest.mark.anyio
async def test_queue_fail_fast_when_no_waiting(monkeypatch):
    monkeypatch.setenv("IMAGE_EMBEDDER_CONCURRENCY", "1")
    monkeypatch.setenv("IMAGE_EMBEDDER_MAX_QUEUE", "0")
    monkeypatch.setenv("IMAGE_EMBEDDER_MAX_WAIT_SECONDS", "60")

    started = threading.Event()
    app = create_app(embedder=SlowEmbedder(delay_seconds=0.50, started_event=started))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        task1 = asyncio.create_task(
            client.post("/embed-image", json={"image_base64": "AA==", "model": "ViT-L-14"})
        )
        # Ensure the first request is actively embedding (it only starts after acquiring the queue slot).
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set(), "first request never started embedding"

        t0 = time.perf_counter()
        r2 = await client.post("/embed-image", json={"image_base64": "AA==", "model": "ViT-L-14"})
        dt = time.perf_counter() - t0

        r1 = await task1

    assert r1.status_code == 200
    assert r2.status_code == 429
    assert dt < 0.20
    assert r2.headers.get("Retry-After") is not None


@pytest.mark.anyio
async def test_queue_wait_timeout_returns_504(monkeypatch):
    monkeypatch.setenv("IMAGE_EMBEDDER_CONCURRENCY", "1")
    monkeypatch.setenv("IMAGE_EMBEDDER_MAX_QUEUE", "10")
    # 0s wait forces immediate timeout for the second request if slot isn't available.
    monkeypatch.setenv("IMAGE_EMBEDDER_MAX_WAIT_SECONDS", "0")

    started = threading.Event()
    app = create_app(embedder=SlowEmbedder(delay_seconds=0.50, started_event=started))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        task1 = asyncio.create_task(
            client.post("/embed-image", json={"image_base64": "AA==", "model": "ViT-L-14"})
        )
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set(), "first request never started embedding"

        r2 = await client.post("/embed-image", json={"image_base64": "AA==", "model": "ViT-L-14"})
        r1 = await task1

    assert r1.status_code == 200
    assert r2.status_code == 504
