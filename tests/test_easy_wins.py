# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import asyncio
import builtins
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from fakes import FakeEmbedder, _no_auth_settings
from image_embedder.batch import BatchWindow, EmbedJob
from image_embedder.config import Settings, _c, _get_bool, _int
from image_embedder.embedder import ImageEmbedder
from image_embedder.lifecycle import make_lifespan
from image_embedder.main import create_app
from image_embedder.memory import check_memory_health, force_cleanup, get_memory_usage
from image_embedder.queue import EmbedQueue, RWLock
from image_embedder.security import make_auth_dependency


@pytest.mark.anyio
async def test_rwlock_release_shared_without_acquire_raises():
    lock = RWLock()
    with pytest.raises(RuntimeError, match="release_shared"):
        await lock.release_shared()


@pytest.mark.anyio
async def test_rwlock_release_exclusive_without_acquire_raises():
    lock = RWLock()
    with pytest.raises(RuntimeError, match="release_exclusive"):
        await lock.release_exclusive()


def test_embed_queue_constructor_rejects_invalid_values():
    with pytest.raises(ValueError, match="concurrency"):
        EmbedQueue(concurrency=0, max_queue=1, max_wait_seconds=1)
    with pytest.raises(ValueError, match="max_queue"):
        EmbedQueue(concurrency=1, max_queue=-1, max_wait_seconds=1)
    with pytest.raises(ValueError, match="max_wait_seconds"):
        EmbedQueue(concurrency=1, max_queue=1, max_wait_seconds=-1)


@pytest.mark.anyio
async def test_auth_dependency_returns_503_when_key_required_but_not_configured():
    settings = Settings(require_api_key=True, service_api_key=None)
    verify = make_auth_dependency(settings)

    app = FastAPI()
    req = Request({"type": "http", "headers": [], "path": "/models", "app": app, "method": "GET"})

    with pytest.raises(Exception) as excinfo:
        await verify(req, x_api_key="some-key", authorization=None)

    assert getattr(excinfo.value, "status_code", None) == 503


class _BatchWindowSpy:
    def __init__(self):
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self):
        self.start_calls += 1

    async def stop(self):
        self.stop_calls += 1


@pytest.mark.anyio
async def test_lifecycle_starts_and_stops_batch_window():
    logger = Mock()
    settings = Settings(warmup_on_startup=False, cleanup_on_shutdown=False)
    batch_window = _BatchWindowSpy()

    app = FastAPI(lifespan=make_lifespan(FakeEmbedder(), settings, logger, batch_window=batch_window))

    async with LifespanManager(app):
        pass

    assert batch_window.start_calls == 1
    assert batch_window.stop_calls == 1


@pytest.mark.anyio
async def test_lifecycle_logs_force_cleanup_failure(monkeypatch):
    logger = Mock()
    settings = Settings(warmup_on_startup=False, cleanup_on_shutdown=True)

    def _boom():
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr("image_embedder.lifecycle.force_cleanup", _boom)

    app = FastAPI(lifespan=make_lifespan(FakeEmbedder(), settings, logger))
    async with LifespanManager(app):
        pass

    assert logger.error.called
    assert any(
        "Error during shutdown cleanup" in str(call.args[0])
        for call in logger.error.call_args_list
    )


def test_memory_usage_handles_psutil_import_error(monkeypatch):
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("psutil unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    usage = get_memory_usage()
    assert usage["process_rss_mb"] is None


def test_check_memory_health_reports_threshold_breaches(monkeypatch):
    monkeypatch.setattr(
        "image_embedder.memory.get_memory_usage",
        lambda: {
            "process_rss_mb": 256.0,
            "gpu_allocated_mb": 128.0,
            "gpu_reserved_mb": 256.0,
        },
    )
    is_healthy, issues = check_memory_health(max_process_mb=64, max_gpu_mb=64)
    assert is_healthy is False
    assert len(issues) == 2


def test_force_cleanup_calls_generation2_gc(monkeypatch):
    calls = []

    def _collect(*args):
        calls.append(args)
        return 0

    monkeypatch.setattr("image_embedder.memory.cleanup_gpu_memory", lambda device=None: {"gc_collected": 0, "gpu_freed_mb": 0.0})
    monkeypatch.setattr("image_embedder.memory.gc.collect", _collect)

    force_cleanup()
    assert (2,) in calls


def test_config_bool_y_and_on_are_true():
    assert _get_bool("y", default=False) is True
    assert _get_bool("on", default=False) is True


def test_config_fallback_helpers(monkeypatch):
    monkeypatch.setattr("image_embedder.config._TOML", {})
    monkeypatch.delenv("NO_SUCH_INT", raising=False)

    assert _c("missing", "key") is None
    assert _int("NO_SUCH_INT", "missing", "key", 42) == 42


def test_embedder_get_device_info_returns_unknown_on_device_failure(monkeypatch):
    embedder = ImageEmbedder(settings=Settings())
    monkeypatch.setattr(embedder, "_resolve_device", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert embedder.get_device_info() == {"type": "unknown"}


def test_embed_route_returns_queue_headers_on_success():
    app = create_app(embedder=FakeEmbedder(), settings=_no_auth_settings())
    client = TestClient(app)

    response = client.post(
        "/embed-image",
        json={
            "image_base64": "AA==",
            "model": "ViT-L-14",
        },
    )

    assert response.status_code == 200
    assert response.headers.get("X-Queue-Concurrency") is not None
    assert response.headers.get("X-Queue-In-Flight") is not None
    assert response.headers.get("X-Queue-Waiting") is not None
    assert response.headers.get("X-Queue-Max-Queue") is not None
    assert response.headers.get("X-Queue-Max-Wait-Seconds") is not None


class _QueueNoop:
    def __init__(self):
        self.acquire_calls = 0
        self.release_calls = 0
        self.acquire_shared_calls = 0
        self.release_shared_calls = 0

    async def acquire(self):
        self.acquire_calls += 1

    async def release(self):
        self.release_calls += 1

    async def acquire_shared(self):
        self.acquire_shared_calls += 1

    async def release_shared(self):
        self.release_shared_calls += 1


class _EmbedderThatFailsBatch:
    def resolve_model(self, model):
        return SimpleNamespace(name=model or "ViT-L-14", image_size=224)

    def embed(self, image_url, image_base64, model, normalize, image_size):
        return [0.1], 1, "local", model or "ViT-L-14", image_size or 224

    def embed_batch(self, spec, target_size, items):
        raise RuntimeError("batch failed")


@pytest.mark.anyio
async def test_batch_dispatch_sets_exception_on_each_future_when_embed_batch_fails():
    queue = _QueueNoop()
    embedder = _EmbedderThatFailsBatch()
    batch = BatchWindow(embedder, queue, batch_window_ms=10, batch_max_size=8)

    loop = asyncio.get_running_loop()
    j1 = EmbedJob(None, "AA==", "ViT-L-14", True, 224)
    j2 = EmbedJob(None, "AA==", "ViT-L-14", False, 224)
    f1 = j1.bind(loop)
    f2 = j2.bind(loop)

    await batch._dispatch([j1, j2])

    assert f1.done() and f2.done()
    with pytest.raises(RuntimeError, match="batch failed"):
        f1.result()
    with pytest.raises(RuntimeError, match="batch failed"):
        f2.result()

    assert queue.acquire_calls == 1
    assert queue.acquire_shared_calls == 1
    assert queue.release_shared_calls == 1
    assert queue.release_calls == 1


class _EmbedderThatReturnsShortBatch:
    def resolve_model(self, model):
        return SimpleNamespace(name=model or "ViT-L-14", image_size=224)

    def embed(self, image_url, image_base64, model, normalize, image_size):
        return [0.1], 1, "local", model or "ViT-L-14", image_size or 224

    def embed_batch(self, spec, target_size, items):
        return [([0.1], 1, "local", spec.name, target_size)]


@pytest.mark.anyio
async def test_batch_dispatch_sets_exception_on_each_future_when_batch_result_count_is_wrong():
    queue = _QueueNoop()
    embedder = _EmbedderThatReturnsShortBatch()
    batch = BatchWindow(embedder, queue, batch_window_ms=10, batch_max_size=8)

    loop = asyncio.get_running_loop()
    j1 = EmbedJob(None, "AA==", "ViT-L-14", True, 224)
    j2 = EmbedJob(None, "AA==", "ViT-L-14", False, 224)
    f1 = j1.bind(loop)
    f2 = j2.bind(loop)

    await batch._dispatch([j1, j2])

    assert f1.done() and f2.done()
    with pytest.raises(RuntimeError, match="returned 1 results for 2 jobs"):
        f1.result()
    with pytest.raises(RuntimeError, match="returned 1 results for 2 jobs"):
        f2.result()

    assert queue.acquire_calls == 1
    assert queue.acquire_shared_calls == 1
    assert queue.release_shared_calls == 1
    assert queue.release_calls == 1
