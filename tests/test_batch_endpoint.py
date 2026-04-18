# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for POST /embed-batch endpoint."""

import base64

import httpx
import pytest
from asgi_lifespan import LifespanManager

from image_embedder.main import create_app
from fakes import FakeEmbedder, _no_auth_settings, _png_bytes

pytestmark = pytest.mark.anyio


def _b64() -> str:
    return base64.b64encode(_png_bytes()).decode()


@pytest.fixture
async def client():
    app = create_app(embedder=FakeEmbedder(), settings=_no_auth_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_embed_batch_all_succeed(client):
    resp = await client.post(
        "/embed-batch",
        json={"items": [{"image_base64": _b64()}, {"image_base64": _b64()}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["succeeded"] == 2
    assert data["failed"] == 0
    assert all(r["status"] == "ok" for r in data["results"])
    assert len(data["results"][0]["embedding"]) == 768
    assert data["model"] == "ViT-L-14"


async def test_embed_batch_queue_headers_present(client):
    resp = await client.post(
        "/embed-batch",
        json={"items": [{"image_base64": _b64()}]},
    )
    assert resp.status_code == 200
    assert "x-queue-concurrency" in resp.headers


async def test_embed_batch_result_indices_correct(client):
    resp = await client.post(
        "/embed-batch",
        json={
            "items": [
                {"image_base64": _b64()},
                {"image_base64": _b64()},
                {"image_base64": _b64()},
            ]
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"][0]["index"] == 0
    assert data["results"][1]["index"] == 1
    assert data["results"][2]["index"] == 2


async def test_embed_batch_explicit_model(client):
    resp = await client.post(
        "/embed-batch",
        json={"items": [{"image_base64": _b64()}], "model": "ViT-B-16"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "ViT-B-16"
    assert data["results"][0]["model"] == "ViT-B-16"


async def test_embed_batch_image_size_propagated(client):
    resp = await client.post(
        "/embed-batch",
        json={"items": [{"image_base64": _b64()}], "image_size": 224},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["image_size"] == 224
    assert data["results"][0]["image_size"] == 224


async def test_embed_batch_non_default_image_size_rejected(client):
    resp = await client.post(
        "/embed-batch",
        json={"items": [{"image_base64": _b64()}], "image_size": 336},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Payload validation errors (Pydantic / FastAPI)
# ---------------------------------------------------------------------------

async def test_embed_batch_missing_image_item_422(client):
    resp = await client.post(
        "/embed-batch",
        json={"items": [{"image_base64": _b64()}, {}]},
    )
    assert resp.status_code == 422


async def test_embed_batch_dual_image_source_item_422(client):
    resp = await client.post(
        "/embed-batch",
        json={
            "items": [
                {
                    "image_url": "https://example.com/poster.jpg",
                    "image_base64": _b64(),
                }
            ]
        },
    )
    assert resp.status_code == 422


async def test_embed_batch_all_missing_image_items_422(client):
    resp = await client.post("/embed-batch", json={"items": [{}, {}]})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Per-item embedder exception surfaced as error result (not HTTP 500)
# ---------------------------------------------------------------------------

class _PartialRaisingEmbedder(FakeEmbedder):
    def embed_batch(self, spec, target_size, items):
        results = []
        for i, _ in enumerate(items):
            if i == 0:
                results.append(ValueError("corrupt image data"))
            else:
                results.append(([0.1, 0.2, 0.3], 3, "local", spec.name, target_size))
        return results


async def test_embed_batch_per_item_exception_is_error_result():
    app = create_app(embedder=_PartialRaisingEmbedder(), settings=_no_auth_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/embed-batch",
                json={"items": [{"image_base64": _b64()}, {"image_base64": _b64()}]},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["failed"] == 1
    assert data["succeeded"] == 1
    assert data["results"][0]["status"] == "error"
    assert "corrupt image" in data["results"][0]["error"]
    assert data["results"][1]["status"] == "ok"


# ---------------------------------------------------------------------------
async def test_embed_batch_empty_items_422(client):
    resp = await client.post("/embed-batch", json={"items": []})
    assert resp.status_code == 422


async def test_embed_batch_invalid_model_pattern_422(client):
    resp = await client.post(
        "/embed-batch",
        json={"items": [{"image_base64": _b64()}], "model": "ViT L-14"},  # space is invalid
    )
    assert resp.status_code == 422


async def test_embed_batch_negative_image_size_422(client):
    resp = await client.post(
        "/embed-batch",
        json={"items": [{"image_base64": _b64()}], "image_size": 0},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Batch size limit (413)
# ---------------------------------------------------------------------------

async def test_embed_batch_exceeds_max_returns_413():
    settings = _no_auth_settings(embed_batch_api_max_items=2)
    app = create_app(embedder=FakeEmbedder(), settings=settings)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/embed-batch",
                json={
                    "items": [
                        {"image_base64": _b64()},
                        {"image_base64": _b64()},
                        {"image_base64": _b64()},
                    ]
                },
            )
    assert resp.status_code == 413
    assert "maximum" in resp.json()["detail"].lower()


async def test_embed_batch_exactly_at_max_succeeds():
    settings = _no_auth_settings(embed_batch_api_max_items=2)
    app = create_app(embedder=FakeEmbedder(), settings=settings)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/embed-batch",
                json={"items": [{"image_base64": _b64()}, {"image_base64": _b64()}]},
            )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

async def test_embed_batch_requires_auth():
    settings = _no_auth_settings()
    settings.require_api_key = True
    settings.service_api_key = "test-key-123"
    app = create_app(embedder=FakeEmbedder(), settings=settings)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/embed-batch",
                json={"items": [{"image_base64": _b64()}]},
            )
    assert resp.status_code in (401, 403)


async def test_embed_batch_valid_api_key_accepted():
    settings = _no_auth_settings()
    settings.require_api_key = True
    settings.service_api_key = "test-key-123"
    app = create_app(embedder=FakeEmbedder(), settings=settings)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/embed-batch",
                json={"items": [{"image_base64": _b64()}]},
                headers={"X-Api-Key": "test-key-123"},
            )
    assert resp.status_code == 200


class _ShortBatchResultEmbedder(FakeEmbedder):
    def embed_batch(self, spec, target_size, items):
        return [([0.1, 0.2, 0.3], 3, "local", spec.name, target_size)]


async def test_embed_batch_returns_500_when_embedder_result_count_is_wrong():
    app = create_app(embedder=_ShortBatchResultEmbedder(), settings=_no_auth_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/embed-batch",
                json={"items": [{"image_base64": _b64()}, {"image_base64": _b64()}]},
            )
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Internal server error"}
