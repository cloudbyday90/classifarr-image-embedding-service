# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import base64
import io
import socket
import sys
import types

import pytest
from PIL import Image

from image_embedder.config import _get_csv_list, Settings
from image_embedder.embedder import ImageEmbedder, MODEL_CATALOG


def _png_bytes() -> bytes:
    img = Image.new("RGB", (1, 1), (0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_get_csv_list_splits_and_strips():
    assert _get_csv_list("a, b, ,c") == ["a", "b", "c"]


def test_list_models_returns_catalog_entries():
    embedder = ImageEmbedder(settings=Settings())
    models = embedder.list_models()
    assert {m.name for m in models} == {m.name for m in MODEL_CATALOG.values()}


def test_resolve_device_auto_uses_cpu_when_cuda_unavailable(monkeypatch):
    fake_torch = types.SimpleNamespace()
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.device = lambda name: f"dev:{name}"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    embedder = ImageEmbedder(settings=Settings(device="auto"))
    assert embedder._resolve_device() == "dev:cpu"


def test_load_model_returns_cached_inside_lock(monkeypatch):
    embedder = ImageEmbedder(settings=Settings(device="cpu"))
    spec = next(iter(MODEL_CATALOG.values()))

    sentinel = ("model", "processor", "cpu")

    class FakeLock:
        def __enter__(self):
            # Simulate another worker populating the cache right before we check inside the lock.
            embedder._models[spec.name] = sentinel
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    embedder._model_locks[spec.name] = FakeLock()

    assert embedder._load_model(spec) == sentinel


def test_validate_remote_url_invalid_missing_host():
    embedder = ImageEmbedder(settings=Settings(allow_remote_urls=True))
    with pytest.raises(ValueError, match="Invalid image URL"):
        embedder._validate_remote_url("http:///no-host")


def test_validate_remote_url_dns_failure(monkeypatch):
    embedder = ImageEmbedder(settings=Settings(allow_remote_urls=True))

    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a, **_k: (_ for _ in ()).throw(socket.gaierror()))

    with pytest.raises(ValueError, match="Unable to resolve remote image host"):
        embedder._validate_remote_url("https://example.com/x.png")


def test_fetch_image_bytes_skips_empty_chunks(monkeypatch):
    settings = Settings(allow_remote_urls=True, max_image_bytes=10)
    embedder = ImageEmbedder(settings=settings)
    monkeypatch.setattr(embedder, "_validate_remote_url", lambda *_a, **_k: None)

    class FakeResponse:
        headers = {"content-length": "2"}
        closed = False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield b""
            yield b"ab"

        def close(self):
            self.closed = True

    import requests

    fake = FakeResponse()
    monkeypatch.setattr(requests, "get", lambda *a, **k: fake)

    data = embedder._fetch_image_bytes("https://image.tmdb.org/t/p/w500/x.png")
    assert data == b"ab"
    assert fake.closed is True


def test_load_image_with_url_uses_fetch(monkeypatch):
    settings = Settings(allow_remote_urls=False)
    embedder = ImageEmbedder(settings=settings)

    monkeypatch.setattr(embedder, "_fetch_image_bytes", lambda _url: _png_bytes())
    img = embedder._load_image(image_url="https://example.com/x.png", image_base64=None)
    assert img.size == (1, 1)


def test_load_image_requires_payload():
    embedder = ImageEmbedder(settings=Settings())
    with pytest.raises(ValueError, match="image_url or image_base64 is required"):
        embedder._load_image(image_url=None, image_base64=None)

