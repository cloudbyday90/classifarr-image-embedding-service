# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The real shared fetch boundary remains enforced across all dispatch modes."""

import asyncio
import base64
import traceback

import httpx2
import numpy as np
import pytest
from asgi_lifespan import LifespanManager
from fakes import _no_auth_settings, _png_bytes
from remote_fakes import RemoteResponse, RemoteTransport, dns_answers
from urllib3.exceptions import ReadTimeoutError

from image_embedder.embedder import ImageEmbedder
from image_embedder.main import create_app
from image_embedder.remote_fetch import RemoteFetchError, fetch_remote_image


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["single", "coalesced", "batch"])
async def test_unsafe_redirect_refusal_preserves_api_and_per_item_order(
    monkeypatch, mode
):
    settings = _no_auth_settings(
        allow_remote_urls=True,
        embed_batch_window_ms=100 if mode == "coalesced" else 0,
        embed_cache_size=8,
        cleanup_on_shutdown=False,
    )
    embedder = ImageEmbedder(settings)
    dns_answers(monkeypatch, "8.8.8.8")
    responses = [
        RemoteResponse(302, {"location": "http://127.0.0.1/private"}) for _ in range(2)
    ]
    transport = RemoteTransport(monkeypatch, *responses)

    def processor(*, images, **_kwargs):
        return {"images": images}

    def model(inputs):
        return [np.zeros((len(inputs["images"]), 768), np.float32)]

    monkeypatch.setattr(
        embedder, "_load_model", lambda _spec: (model, processor, "ov:CPU")
    )
    app = create_app(embedder, settings)
    async with LifespanManager(app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app), base_url="http://test"
        ) as client:
            bad = {"image_url": "http://images.example/x"}
            if mode == "batch":
                result = await client.post(
                    "/embed-batch",
                    json={
                        "items": [
                            bad,
                            {"image_base64": base64.b64encode(_png_bytes()).decode()},
                        ]
                    },
                )
                assert result.status_code == 200
                body = result.json()
                assert body["succeeded"] == body["failed"] == 1
                assert [item["index"] for item in body["results"]] == [0, 1]
                assert [item["status"] for item in body["results"]] == ["error", "ok"]
                assert "private address" in body["results"][0]["error"]
                assert body["results"][1]["dims"] == 768
            else:
                count = 2 if mode == "coalesced" else 1
                results = await asyncio.gather(
                    *(client.post("/embed-image", json=bad) for _ in range(count))
                )
                assert all(result.status_code == 400 for result in results)
                assert all(
                    "private address" in result.json()["detail"] for result in results
                )
            assert await app.state.executor.close(3)
    assert app.state.queue.stats().in_flight == app.state.queue.stats().waiting == 0
    assert len(transport.calls) == (2 if mode == "coalesced" else 1)
    assert all(
        response.closed and not response.streamed
        for response in responses[: len(transport.calls)]
    )


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["single", "coalesced", "batch"])
@pytest.mark.parametrize("failure", ["http", "timeout"])
async def test_download_failure_retains_status_and_sanitizes_logs(
    monkeypatch, caplog, mode, failure
):
    settings = _no_auth_settings(
        allow_remote_urls=True,
        embed_batch_window_ms=100 if mode == "coalesced" else 0,
        cleanup_on_shutdown=False,
    )
    embedder = ImageEmbedder(settings)
    dns_answers(monkeypatch, "8.8.8.8")
    secret = "do-not-log-this-query"
    responses = [
        RemoteResponse(404)
        if failure == "http"
        else RemoteResponse(error=ReadTimeoutError(None, secret, "timeout"))
        for _ in range(2)
    ]
    transport = RemoteTransport(monkeypatch, *responses)
    app = create_app(embedder, settings)
    async with LifespanManager(app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app), base_url="http://test"
        ) as client:
            bad = {"image_url": f"http://images.example/x?token={secret}"}
            if mode == "batch":
                result = await client.post("/embed-batch", json={"items": [bad]})
                assert result.status_code == 200
                body = result.json()
                assert body["failed"] == 1 and body["succeeded"] == 0
                assert body["results"][0]["index"] == 0
                assert body["results"][0]["status"] == "error"
                assert secret not in result.text
            else:
                count = 2 if mode == "coalesced" else 1
                results = await asyncio.gather(
                    *(client.post("/embed-image", json=bad) for _ in range(count))
                )
                assert all(result.status_code == 500 for result in results)
                assert all(
                    result.json()["detail"] == "Internal server error"
                    for result in results
                )
            assert await app.state.executor.close(3)
    assert secret not in caplog.text
    assert len(transport.calls) == (2 if mode == "coalesced" else 1)
    assert all(response.closed for response in responses[: len(transport.calls)])
    assert app.state.queue.stats().in_flight == app.state.queue.stats().waiting == 0


def test_transport_exception_traceback_does_not_expose_query(monkeypatch):
    dns_answers(monkeypatch, "8.8.8.8")
    response = RemoteResponse(error=ReadTimeoutError(None, "secret-query", "timeout"))
    RemoteTransport(monkeypatch, response)
    with pytest.raises(RemoteFetchError) as error:
        fetch_remote_image(
            "http://images.example/x?secret-query",
            _no_auth_settings(allow_remote_urls=True),
        )
    rendered = "".join(traceback.format_exception(error.value))
    assert "secret-query" not in rendered
    assert response.closed
