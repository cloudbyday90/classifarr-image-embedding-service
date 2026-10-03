# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Body and inline rejection occurs before admission and worker dispatch."""

import asyncio
import base64
import json
import threading

import httpx2
import numpy as np
import pytest
from asgi_lifespan import LifespanManager
from fakes import FakeEmbedder, _no_auth_settings, _png_bytes

from image_embedder.embedder import ImageEmbedder
from image_embedder.main import create_app


class CountingEmbedder(FakeEmbedder):
    calls = 0

    def embed(self, *args):
        self.calls += 1
        return super().embed(*args)

    def embed_batch(self, *args):
        self.calls += 1
        return super().embed_batch(*args)


async def raw_request(app, chunks, *, headers=()):
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/embed-image",
        "raw_path": b"/embed-image",
        "root_path": "",
        "query_string": b"",
        "scheme": "http",
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
        "headers": [(b"content-type", b"application/json"), *headers],
    }
    messages = iter(
        {"type": "http.request", "body": chunk, "more_body": i < len(chunks) - 1}
        for i, chunk in enumerate(chunks)
    )
    sent = []
    receives = 0
    finished = asyncio.Event()

    async def receive():
        nonlocal receives
        receives += 1
        try:
            return next(messages)
        except StopIteration:
            await finished.wait()
            return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and not message.get(
            "more_body", False
        ):
            finished.set()

    await asyncio.wait_for(app(scope, receive, send), 5)
    status = next(
        message["status"]
        for message in sent
        if message["type"] == "http.response.start"
    )
    return status, receives


@pytest.mark.anyio
@pytest.mark.parametrize("header", [None, b"1", b"invalid", b"-1"])
async def test_streamed_body_limit_cannot_be_bypassed_by_headers(header):
    embedder = CountingEmbedder()
    app = create_app(embedder, _no_auth_settings(max_request_body_bytes=10))
    headers = [] if header is None else [(b"content-length", header)]
    status, _ = await raw_request(
        app, [b'{"image_', b'url":"too big"}'], headers=headers
    )
    assert status == 413
    assert embedder.calls == 0
    assert app.state.queue.stats().in_flight == app.state.queue.stats().waiting == 0


@pytest.mark.anyio
async def test_declared_oversize_never_reads_body():
    embedder = CountingEmbedder()
    app = create_app(embedder, _no_auth_settings(max_request_body_bytes=10))
    status, receives = await raw_request(
        app, [b"{}"], headers=[(b"content-length", b"11")]
    )
    assert status == 413
    assert receives == 0
    assert embedder.calls == 0


@pytest.mark.anyio
async def test_exact_body_limit_preserves_successful_response():
    body = json.dumps({"image_url": "https://example.com/poster.png"}).encode()
    embedder = CountingEmbedder()
    app = create_app(embedder, _no_auth_settings(max_request_body_bytes=len(body)))
    status, _ = await raw_request(app, [body[:4], b"", body[4:]])
    assert status == 200
    assert embedder.calls == 1


@pytest.mark.anyio
@pytest.mark.parametrize("batch_window", [0, 100])
async def test_inline_oversize_rejected_before_single_admission(batch_window):
    embedder = CountingEmbedder()
    app = create_app(
        embedder,
        _no_auth_settings(max_image_bytes=3, embed_batch_window_ms=batch_window),
    )
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post("/embed-image", json={"image_base64": "YWJjZA=="})
    assert response.status_code == 413
    assert embedder.calls == 0
    assert app.state.queue.stats().in_flight == app.state.queue.stats().waiting == 0


@pytest.mark.anyio
async def test_inline_batch_aggregate_rejected_before_worker():
    embedder = CountingEmbedder()
    app = create_app(embedder, _no_auth_settings(max_batch_image_bytes=5))
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/embed-batch", json={"items": [{"image_base64": "YWJj"}] * 2}
        )
    assert response.status_code == 413
    assert "aggregate" in response.json()["detail"]
    assert embedder.calls == 0
    assert app.state.queue.stats().in_flight == 0


@pytest.mark.anyio
async def test_real_pixel_refusal_does_not_load_model(monkeypatch):
    settings = _no_auth_settings(max_image_pixels=1)
    embedder = ImageEmbedder(settings)
    monkeypatch.setattr(
        embedder, "_load_model", lambda *_a: pytest.fail("rejected image loaded model")
    )
    app = create_app(embedder, settings)
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/embed-image",
            json={"image_base64": base64.b64encode(_png_bytes()).decode()},
        )
    assert response.status_code == 413
    assert app.state.queue.stats().in_flight == 0


@pytest.mark.anyio
async def test_coalesced_pixel_budget_preserves_single_response_mapping(monkeypatch):
    settings = _no_auth_settings(
        embed_batch_window_ms=1000, embed_cache_size=0, max_batch_image_pixels=50_177
    )
    embedder = ImageEmbedder(settings)
    monkeypatch.setattr(
        embedder,
        "_load_model",
        lambda *_a: (
            lambda inputs: [np.zeros((len(inputs["images"]), 768), np.float32)],
            lambda *, images, **_kwargs: {"images": images},
            "ov:CPU",
        ),
    )
    app = create_app(embedder, settings)
    async with LifespanManager(app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app), base_url="http://test"
        ) as client:
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/embed-image",
                        json={"image_base64": base64.b64encode(_png_bytes()).decode()},
                    )
                    for _ in range(2)
                )
            )
    assert sorted(response.status_code for response in responses) == [200, 413]
    assert app.state.queue.stats().in_flight == app.state.queue.stats().waiting == 0


@pytest.mark.anyio
async def test_timed_out_worker_keeps_image_until_native_work_finishes(monkeypatch):
    settings = _no_auth_settings(
        embedding_timeout_seconds=0.05, cleanup_on_shutdown=False, embed_cache_size=0
    )
    embedder = ImageEmbedder(settings)
    started, release = threading.Event(), threading.Event()
    seen = []

    def processor(*, images, **_kwargs):
        seen.append(images)
        started.set()
        if not release.wait(5):
            raise RuntimeError("test worker was not released")
        return {"image": images}

    monkeypatch.setattr(
        embedder,
        "_load_model",
        lambda *_a: (
            lambda _inputs: [np.zeros((1, 768), np.float32)],
            processor,
            "ov:CPU",
        ),
    )
    app = create_app(embedder, settings)
    async with LifespanManager(app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app), base_url="http://test"
        ) as client:
            try:
                request = asyncio.create_task(
                    client.post(
                        "/embed-image",
                        json={"image_base64": base64.b64encode(_png_bytes()).decode()},
                    )
                )
                assert await asyncio.to_thread(started.wait, 3)
                response = await request
                assert response.status_code == 504
                assert app.state.queue.stats().in_flight == 1
                assert seen[0].getpixel((0, 0)) == (255, 0, 0)
            finally:
                release.set()
            assert await app.state.executor.close(3)
    with pytest.raises(ValueError, match="closed image"):
        seen[0].getpixel((0, 0))
