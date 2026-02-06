# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import base64
import io

import pytest
from PIL import Image

from image_embedder.config import Settings, _get_bool
from image_embedder.embedder import ImageEmbedder, MODEL_CATALOG


def _png_bytes() -> bytes:
    img = Image.new("RGB", (1, 1), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_get_bool_parses_truthy_values():
    assert _get_bool("1", False) is True
    assert _get_bool("true", False) is True
    assert _get_bool("YES", False) is True
    assert _get_bool(" on ", False) is True
    assert _get_bool("0", True) is False
    assert _get_bool("no", True) is False


def test_get_bool_uses_default_for_none():
    assert _get_bool(None, True) is True
    assert _get_bool(None, False) is False


def test_resolve_model_falls_back_to_first_catalog_entry_when_default_invalid():
    settings = Settings(default_model="does-not-exist")
    embedder = ImageEmbedder(settings=settings)
    spec = embedder.resolve_model(None)

    first = next(iter(MODEL_CATALOG.values()))
    assert spec.name == first.name


def test_resolve_model_unknown_candidate_uses_default():
    settings = Settings(default_model="ViT-B-16")
    embedder = ImageEmbedder(settings=settings)
    spec = embedder.resolve_model("does-not-exist")
    assert spec.name == "ViT-B-16"


def test_decode_base64_rejects_invalid_payload():
    settings = Settings()
    embedder = ImageEmbedder(settings=settings)
    with pytest.raises(ValueError, match="Invalid base64 image payload"):
        embedder._decode_base64("not base64!!!")


def test_decode_base64_enforces_max_size():
    settings = Settings(max_image_bytes=3)
    embedder = ImageEmbedder(settings=settings)
    payload = base64.b64encode(b"abcd").decode("ascii")
    with pytest.raises(ValueError, match="Image payload exceeds maximum size"):
        embedder._decode_base64(payload)


def test_load_image_rejects_invalid_image_bytes():
    settings = Settings()
    embedder = ImageEmbedder(settings=settings)
    payload = base64.b64encode(b"not an image").decode("ascii")
    with pytest.raises(ValueError, match="Unable to decode image bytes"):
        embedder._load_image(image_url=None, image_base64=payload)


def test_load_image_accepts_valid_base64_png():
    settings = Settings()
    embedder = ImageEmbedder(settings=settings)
    payload = base64.b64encode(_png_bytes()).decode("ascii")
    img = embedder._load_image(image_url=None, image_base64=payload)
    assert img.mode == "RGB"
    assert img.size == (1, 1)


def test_load_image_prefers_base64_over_url(monkeypatch):
    settings = Settings()
    embedder = ImageEmbedder(settings=settings)
    payload = base64.b64encode(_png_bytes()).decode("ascii")

    def _boom(_url: str) -> bytes:
        raise AssertionError("_fetch_image_bytes should not be called when base64 is provided")

    monkeypatch.setattr(embedder, "_fetch_image_bytes", _boom)

    img = embedder._load_image(image_url="https://example.com/poster.jpg", image_base64=payload)
    assert img.size == (1, 1)


def test_embed_rejects_non_positive_image_size(monkeypatch):
    settings = Settings()
    embedder = ImageEmbedder(settings=settings)

    def _boom(*args, **kwargs):
        raise AssertionError("Model/image loading should not happen for invalid image_size")

    monkeypatch.setattr(embedder, "_load_model", _boom)
    monkeypatch.setattr(embedder, "_load_image", _boom)

    with pytest.raises(ValueError, match="image_size must be a positive integer"):
        embedder.embed(
            image_url="https://example.com/poster.jpg",
            image_base64=None,
            model="ViT-L-14",
            normalize=True,
            image_size=0,
        )


def test_fetch_image_bytes_rejects_remote_urls_when_disabled(monkeypatch):
    settings = Settings(allow_remote_urls=False)
    embedder = ImageEmbedder(settings=settings)

    # Ensure we don't accidentally hit network if behavior regresses.
    import requests

    def _boom(*args, **kwargs):
        raise AssertionError("requests.get should not be called when remote URLs are disabled")

    monkeypatch.setattr(requests, "get", _boom)

    with pytest.raises(ValueError, match="Remote image URLs are disabled"):
        embedder._fetch_image_bytes("https://example.com/poster.jpg")


def test_fetch_image_bytes_enforces_max_size_via_content_length(monkeypatch):
    settings = Settings(allow_remote_urls=True, max_image_bytes=3)
    embedder = ImageEmbedder(settings=settings)

    class FakeResponse:
        headers = {"content-length": "4"}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield b"abcd"

        def close(self):
            return None

    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse())
    monkeypatch.setattr(embedder, "_validate_remote_url", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="Image payload exceeds maximum size"):
        embedder._fetch_image_bytes("https://example.com/poster.jpg")


def test_fetch_image_bytes_enforces_max_size_via_actual_bytes(monkeypatch):
    settings = Settings(allow_remote_urls=True, max_image_bytes=3)
    embedder = ImageEmbedder(settings=settings)

    class FakeResponse:
        headers = {}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield b"ab"
            yield b"cd"

        def close(self):
            return None

    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse())
    monkeypatch.setattr(embedder, "_validate_remote_url", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="Image payload exceeds maximum size"):
        embedder._fetch_image_bytes("https://example.com/poster.jpg")
