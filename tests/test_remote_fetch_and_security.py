# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import types

import pytest

from image_embedder.config import Settings
from image_embedder.embedder import ImageEmbedder


def test_validate_remote_url_rejects_non_http_scheme():
    embedder = ImageEmbedder(settings=Settings(allow_remote_urls=True))
    with pytest.raises(ValueError, match="Only http\\(s\\) image URLs are supported"):
        embedder._validate_remote_url("file:///etc/passwd")


def test_validate_remote_url_rejects_localhost_hostname():
    embedder = ImageEmbedder(settings=Settings(allow_remote_urls=True))
    with pytest.raises(ValueError, match="private address"):
        embedder._validate_remote_url("http://localhost/poster.png")


def test_validate_remote_url_rejects_private_ip_literal():
    embedder = ImageEmbedder(settings=Settings(allow_remote_urls=True))
    with pytest.raises(ValueError, match="private address"):
        embedder._validate_remote_url("http://127.0.0.1/poster.png")

def test_validate_remote_url_accepts_public_ip_literal():
    embedder = ImageEmbedder(settings=Settings(allow_remote_urls=True))
    # 1.2.3.4 is a public IPv4 literal; no DNS should be attempted.
    embedder._validate_remote_url("https://1.2.3.4/poster.png")


def test_validate_remote_url_enforces_allowlist(monkeypatch):
    embedder = ImageEmbedder(
        settings=Settings(
            allow_remote_urls=True,
            allowed_remote_hosts=["image.tmdb.org"],
        )
    )

    # Ensure we don't do real DNS in tests.
    import socket

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(None, None, None, None, ("1.2.3.4", 0))],
    )

    # Allowed
    embedder._validate_remote_url("https://image.tmdb.org/t/p/w500/x.png")

    # Not allowed
    with pytest.raises(ValueError, match="allowlisted"):
        embedder._validate_remote_url("https://example.com/x.png")


def test_fetch_image_bytes_streaming_enforces_max_size(monkeypatch):
    settings = Settings(allow_remote_urls=True, max_image_bytes=3)
    embedder = ImageEmbedder(settings=settings)

    # Avoid DNS and allow validation to pass.
    monkeypatch.setattr(embedder, "_validate_remote_url", lambda *_a, **_k: None)

    class FakeResponse:
        headers = {}
        closed = False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield b"ab"
            yield b"cd"

        def close(self):
            self.closed = True

    import requests

    fake = FakeResponse()
    monkeypatch.setattr(requests, "get", lambda *a, **k: fake)

    with pytest.raises(ValueError, match="Image payload exceeds maximum size"):
        embedder._fetch_image_bytes("https://image.tmdb.org/t/p/w500/x.png")

    assert fake.closed is True


def test_fetch_image_bytes_returns_bytes_when_under_limit(monkeypatch):
    settings = Settings(allow_remote_urls=True, max_image_bytes=10)
    embedder = ImageEmbedder(settings=settings)

    monkeypatch.setattr(embedder, "_validate_remote_url", lambda *_a, **_k: None)

    class FakeResponse:
        headers = {"content-length": "4"}
        closed = False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield b"ab"
            yield b"cd"

        def close(self):
            self.closed = True

    import requests

    fake = FakeResponse()
    monkeypatch.setattr(requests, "get", lambda *a, **k: fake)

    data = embedder._fetch_image_bytes("https://image.tmdb.org/t/p/w500/x.png")
    assert data == b"abcd"
    assert fake.closed is True
