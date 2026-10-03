# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Explicit and coalesced endpoints preserve shared-budget outcomes and capacity."""

import asyncio
import base64

import httpx2
import pytest
from asgi_lifespan import LifespanManager
from fakes import _png_bytes
from test_remote_batch_budget import clock, inline_item, prepared_embedder
from test_remote_process import harness
from worker_fakes import wait_until

from image_embedder import remote_budget
from image_embedder.batch import BatchWindow, EmbedJob
from image_embedder.main import create_app
from image_embedder.queue import EmbedQueue


@pytest.fixture
def anyio_backend():
    return "asyncio"


def timed_fetch(monkeypatch):
    current = clock(monkeypatch)
    deadlines = []

    def fetch(_url, _settings, *, total_deadline):
        deadlines.append(total_deadline)
        current[0] += 0.4 if len(deadlines) == 1 else 0.6
        return _png_bytes()

    monkeypatch.setattr(remote_budget, "fetch_remote_image", fetch)
    return deadlines


@pytest.mark.anyio
async def test_explicit_batch_returns_ordered_partial_results(monkeypatch):
    deadlines = timed_fetch(monkeypatch)
    embedder, spec, _calls = prepared_embedder(
        monkeypatch,
        require_api_key=False,
        warmup_on_startup=False,
        cleanup_on_shutdown=False,
    )
    app = create_app(embedder=embedder, settings=embedder.settings)
    remote = {"image_url": "https://images.example/a?token=test-private-detail"}
    inline = {"image_base64": inline_item().image_base64}
    async with LifespanManager(app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/embed-batch",
                json={"normalize": False, "items": [remote, remote, inline, remote]},
            )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4 and body["succeeded"] == body["failed"] == 2
    assert [item["index"] for item in body["results"]] == [0, 1, 2, 3]
    assert [item["status"] for item in body["results"]] == [
        "ok",
        "error",
        "ok",
        "error",
    ]
    for index in (1, 3):
        assert body["results"][index]["error"] == "Remote batch fetch timed out"
    for index in (0, 2):
        assert len(body["results"][index]["embedding"]) == spec.dims
    assert "test-private-detail" not in response.text
    assert deadlines == [101, 101]
    assert app.state.queue.stats().in_flight == app.state.ingress.stats().active == 0


@pytest.mark.anyio
async def test_automatic_group_uses_same_budget_for_remote_and_inline_jobs(monkeypatch):
    deadlines = timed_fetch(monkeypatch)
    embedder, spec, _calls = prepared_embedder(monkeypatch)
    queue = EmbedQueue(1, 0, 2)
    batch = BatchWindow(embedder, queue, 1000, 3)
    await batch.start()
    try:
        jobs = [
            EmbedJob("https://images.example/a", None, None, False, None),
            EmbedJob("https://images.example/b", None, None, False, None),
            EmbedJob(None, inline_item().image_base64, None, False, None),
        ]
        results = await asyncio.gather(
            *(batch.submit(job) for job in jobs), return_exceptions=True
        )
        assert len(results[0][0]) == len(results[2][0]) == spec.dims
        assert str(results[1]) == "Remote batch fetch timed out"
        assert deadlines == [101, 101]
    finally:
        await batch.stop()
    assert queue.stats().in_flight == queue.stats().rw_readers == 0


@pytest.mark.anyio
async def test_http_timeout_retains_native_batch_owner_then_recovers_capacity(
    monkeypatch, tmp_path
):
    children = harness(monkeypatch, tmp_path)
    embedder, _spec, calls = prepared_embedder(
        monkeypatch,
        require_api_key=False,
        warmup_on_startup=False,
        cleanup_on_shutdown=False,
        embed_max_queue=0,
        embedding_timeout_seconds=0.15,
    )
    app = create_app(embedder=embedder, settings=embedder.settings)
    remote = {"image_url": "https://images.example/a"}
    inline = {"image_base64": base64.b64encode(_png_bytes()).decode()}
    async with LifespanManager(app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/embed-batch", json={"items": [remote] * 32})
            assert response.status_code == 504
            assert children[0].poll() is None
            assert (
                app.state.queue.stats().in_flight
                == app.state.queue.stats().rw_readers
                == 1
            )
            retry = await client.post("/embed-batch", json={"items": [inline]})
            assert retry.status_code == 429
            await wait_until(lambda: app.state.queue.stats().in_flight == 0)
            assert (
                len(children) == 1
                and children[0].poll() is not None
                and children[0].stdin.closed
            )
            assert calls == []
            recovered = await client.post("/embed-batch", json={"items": [inline]})
            assert recovered.status_code == 200
            assert app.state.queue.stats().rw_readers == 0
