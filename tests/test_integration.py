# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Integration tests — full ASGI lifespan (startup → request → shutdown).

These tests differ from unit/API-layer tests in that:
  - Every test uses LifespanManager so the app fully starts and stops.
  - Multiple components (auth, rate limiting, queue, embedder, routing) are
    exercised together without mocking individual layers.
  - Covers paths that only manifest when middleware and route handlers interact.
"""

import asyncio
import base64
import threading
import time

import httpx
import pytest
from asgi_lifespan import LifespanManager

from fakes import FakeEmbedder, _no_auth_settings, _png_bytes
from image_embedder.config import Settings
from image_embedder.main import create_app


# ── Shared helpers ────────────────────────────────────────────────────────────

_EMBED_BODY = {"image_url": "https://example.com/img.jpg", "model": "ViT-L-14"}
_API_KEY = "integration-test-key"


def _auth_settings(**kwargs) -> Settings:
    """Settings with auth enabled and a known API key."""
    s = Settings()
    s.require_api_key = True
    s.service_api_key = _API_KEY
    s.warmup_on_startup = False
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


class SlowEmbedder(FakeEmbedder):
    """FakeEmbedder that blocks for `delay` seconds, optionally signalling a threading.Event."""

    def __init__(self, delay: float, started: threading.Event | None = None):
        super().__init__()
        self._delay = delay
        self._started = started

    def embed(self, image_url, image_base64, model, normalize, image_size):
        if self._started is not None:
            self._started.set()
        time.sleep(self._delay)
        return super().embed(image_url, image_base64, model, normalize, image_size)


class RaisingEmbedder(FakeEmbedder):
    """FakeEmbedder whose embed() always raises an unexpected exception."""

    def embed(self, image_url, image_base64, model, normalize, image_size):
        raise RuntimeError("simulated embedder crash")


class WarmupTracker(FakeEmbedder):
    """FakeEmbedder that records warmup() calls."""

    def __init__(self):
        super().__init__()
        self.warmup_calls = 0

    def warmup(self, model_name=None):
        self.warmup_calls += 1


# ── 1. Happy path: full lifespan + authenticated embed ───────────────────────

@pytest.mark.anyio
async def test_embed_success_through_full_lifespan():
    """Startup → authenticated POST /embed-image → queue headers → shutdown."""
    app = create_app(embedder=FakeEmbedder(), settings=_auth_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/embed-image",
                json=_EMBED_BODY,
                headers={"X-Api-Key": _API_KEY},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dims"] == 768
    assert body["model"] == "ViT-L-14"
    # Queue telemetry headers must be present on every successful embed.
    for header in (
        "x-queue-concurrency",
        "x-queue-in-flight",
        "x-queue-waiting",
        "x-queue-max-queue",
        "x-queue-max-wait-seconds",
    ):
        assert header in resp.headers, f"Missing header: {header}"


# ── 2. Auth rejects missing key ───────────────────────────────────────────────

@pytest.mark.anyio
async def test_embed_rejects_unauthenticated_request():
    app = create_app(embedder=FakeEmbedder(), settings=_auth_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/embed-image", json=_EMBED_BODY)

    assert resp.status_code == 401


# ── 3. Auth rejects wrong key ─────────────────────────────────────────────────

@pytest.mark.anyio
async def test_embed_rejects_wrong_api_key():
    app = create_app(embedder=FakeEmbedder(), settings=_auth_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/embed-image",
                json=_EMBED_BODY,
                headers={"X-Api-Key": "wrong-key"},
            )

    assert resp.status_code == 401


# ── 4. Bearer token accepted through full stack ───────────────────────────────

@pytest.mark.anyio
async def test_bearer_token_accepted_through_full_stack():
    app = create_app(embedder=FakeEmbedder(), settings=_auth_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/models",
                headers={"Authorization": f"Bearer {_API_KEY}"},
            )

    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()]
    assert "ViT-L-14" in ids


# ── 5. /health and /ready are public even when auth is required ───────────────

@pytest.mark.anyio
async def test_health_and_ready_are_public_with_auth_enabled():
    app = create_app(embedder=FakeEmbedder(), settings=_auth_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health")
            ready = await client.get("/ready")

    assert health.status_code == 200
    assert ready.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.json()["ready"] is True


# ── 6. Pydantic rejects invalid model name pattern (422) ─────────────────────

@pytest.mark.anyio
async def test_invalid_model_name_returns_422():
    """The `pattern` constraint on EmbedImageRequest.model rejects path-traversal."""
    app = create_app(embedder=FakeEmbedder(), settings=_no_auth_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/embed-image",
                json={"image_url": "https://example.com/img.jpg", "model": "../../../etc/passwd"},
            )

    assert resp.status_code == 422


# ── 7. Invalid image-source combinations return 422 ─────────────────────────

@pytest.mark.anyio
async def test_missing_image_input_returns_422():
    """Neither image_url nor image_base64 provided → HTTP 422."""
    app = create_app(embedder=FakeEmbedder(), settings=_no_auth_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/embed-image", json={"model": "ViT-L-14"})

    assert resp.status_code == 422


@pytest.mark.anyio
async def test_dual_image_input_returns_422():
    """Both image_url and image_base64 provided → HTTP 422."""
    png_b64 = base64.b64encode(_png_bytes()).decode()
    app = create_app(embedder=FakeEmbedder(), settings=_no_auth_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/embed-image",
                json={
                    "image_url": "https://example.com/img.jpg",
                    "image_base64": png_b64,
                    "model": "ViT-L-14",
                },
            )

    assert resp.status_code == 422


# ── 8. base64 image path ──────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_embed_base64_image_path():
    png_b64 = base64.b64encode(_png_bytes()).decode()
    app = create_app(embedder=FakeEmbedder(), settings=_no_auth_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/embed-image",
                json={"image_base64": png_b64, "model": "ViT-B-16"},
            )

    assert resp.status_code == 200
    assert resp.json()["model"] == "ViT-B-16"


# ── 9. Request timeout returns 504 ───────────────────────────────────────────

@pytest.mark.anyio
async def test_request_timeout_returns_504():
    """asyncio.wait_for raises TimeoutError → HTTP 504 with queue headers."""
    # Embedder sleeps 0.5s; timeout is 0.1s — reliably triggers asyncio.TimeoutError.
    settings = _no_auth_settings()
    settings.request_timeout_seconds = 0.1  # type: ignore[assignment]
    app = create_app(embedder=SlowEmbedder(delay=0.5), settings=settings)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=5.0) as client:
            resp = await client.post("/embed-image", json=_EMBED_BODY)

    assert resp.status_code == 504
    assert "x-queue-concurrency" in resp.headers


# ── 10. Queue full returns 429 ────────────────────────────────────────────────

@pytest.mark.anyio
async def test_queue_full_returns_429():
    """concurrency=1, max_queue=0, slow embedder → second concurrent request gets 429."""
    started = threading.Event()
    settings = _no_auth_settings(
        embed_concurrency=1,
        embed_max_queue=0,
        embed_max_wait_seconds=0,
    )
    settings.request_timeout_seconds = 10  # type: ignore[assignment]
    app = create_app(embedder=SlowEmbedder(delay=1.0, started=started), settings=settings)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=15.0) as client:
            # First request occupies the single concurrency slot.
            task = asyncio.create_task(client.post("/embed-image", json=_EMBED_BODY))
            # Wait until the embedder has actually started (slot is taken).
            await asyncio.to_thread(started.wait, 5.0)
            # Second request must be rejected immediately.
            resp2 = await client.post("/embed-image", json=_EMBED_BODY)
            await task  # allow the first request to finish cleanly

    assert resp2.status_code == 429
    assert "x-queue-concurrency" in resp2.headers


# ── 11. Unhandled embedder exception → 500 ───────────────────────────────────

@pytest.mark.anyio
async def test_unhandled_embedder_exception_returns_500():
    """Global exception handler converts unexpected RuntimeError into HTTP 500."""
    app = create_app(embedder=RaisingEmbedder(), settings=_no_auth_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/embed-image", json=_EMBED_BODY)

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal server error"


# ── 12. /admin/cleanup always protected, even when require_api_key=false ──────

@pytest.mark.anyio
async def test_admin_cleanup_always_requires_auth():
    """Admin endpoint is always protected regardless of REQUIRE_API_KEY."""
    settings = _no_auth_settings()
    settings.require_api_key = False
    settings.service_api_key = _API_KEY
    app = create_app(embedder=FakeEmbedder(), settings=settings)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            no_key = await client.post("/admin/cleanup")
            with_key = await client.post("/admin/cleanup", headers={"X-Api-Key": _API_KEY})

    assert no_key.status_code == 401
    assert with_key.status_code == 200


# ── 13. Warmup is called exactly once on startup ──────────────────────────────

@pytest.mark.anyio
async def test_warmup_called_exactly_once_on_startup():
    tracker = WarmupTracker()
    settings = _no_auth_settings(warmup_on_startup=True)
    app = create_app(embedder=tracker, settings=settings)
    async with LifespanManager(app):
        pass  # lifespan startup/shutdown is all we need

    assert tracker.warmup_calls == 1


# ── 14. Rate limit returns 429 after quota exceeded ───────────────────────────

@pytest.mark.anyio
async def test_rate_limit_returns_429_after_quota_exceeded():
    """A very small rate limit (2/minute) is exhausted on the third request."""
    settings = _no_auth_settings(rate_limit_embed="2/minute")
    app = create_app(embedder=FakeEmbedder(), settings=settings)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r1 = await client.post("/embed-image", json=_EMBED_BODY)
            r2 = await client.post("/embed-image", json=_EMBED_BODY)
            r3 = await client.post("/embed-image", json=_EMBED_BODY)  # over quota

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429


# ── 15. Concurrent requests all succeed when queue has capacity ───────────────

@pytest.mark.anyio
async def test_concurrent_requests_all_succeed():
    settings = _no_auth_settings(
        embed_concurrency=3,
        embed_max_queue=10,
        embed_max_wait_seconds=5,
        rate_limit_embed="100/minute",
    )
    settings.request_timeout_seconds = 10  # type: ignore[assignment]
    app = create_app(embedder=FakeEmbedder(), settings=settings)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=15.0) as client:
            responses = await asyncio.gather(
                *[client.post("/embed-image", json=_EMBED_BODY) for _ in range(6)]
            )

    assert all(r.status_code == 200 for r in responses)
