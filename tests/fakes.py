# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared fake/stub classes and helper functions used across the test suite."""

import io
import socket

from PIL import Image

from image_embedder.config import Settings


def install_remote_response(monkeypatch, response):
    """Exercise real destination validation while replacing outbound transport."""
    from image_embedder import remote_fetch

    # Transport unit fakes exercise the worker's direct policy. Fresh-child tests
    # independently verify production supervision; this seam is never automatic.
    monkeypatch.setattr("image_embedder.embedder.fetch_remote_image", remote_fetch.fetch_remote_image)

    monkeypatch.setattr(
        socket, "getaddrinfo", lambda host, port, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", port))
        ]
    )

    class Pool:
        def urlopen(self, *_args, **_kwargs):
            return response

        def close(self):
            pass

    monkeypatch.setattr(remote_fetch, "_pool", lambda *_args: Pool())


def _no_auth_settings(**kwargs) -> Settings:
    """Return a Settings instance with auth and warmup disabled."""
    s = Settings()
    s.require_api_key = False
    s.warmup_on_startup = False
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


def _png_bytes() -> bytes:
    """Return raw bytes of a minimal 1×1 RGB PNG image."""
    img = Image.new("RGB", (1, 1), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakeEmbedder:
    """Minimal in-process stub for ImageEmbedder, suitable for API-layer tests."""

    def __init__(self):
        self.models = [
            {"name": "ViT-L-14", "dims": 768, "image_size": 224},
            {"name": "ViT-B-16", "dims": 512, "image_size": 224},
        ]
        self._loaded_models = {"ViT-L-14"}

    def list_models(self):
        class Spec:
            def __init__(self, name, dims, image_size):
                self.name = name
                self.dims = dims
                self.image_size = image_size

        return [Spec(m["name"], m["dims"], m["image_size"]) for m in self.models]

    def embed(self, image_url, image_base64, model, normalize, image_size):
        name = model or "ViT-L-14"
        dims = 512 if name == "ViT-B-16" else 768
        embedding = [0.1] * dims
        return embedding, dims, "local", name, image_size or 224

    def resolve_model(self, model_name=None):
        class Spec:
            def __init__(self, name, dims, image_size):
                self.name = name
                self.dims = dims
                self.image_size = image_size

        name = model_name or "ViT-L-14"
        dims = 512 if name == "ViT-B-16" else 768
        return Spec(name, dims, 224)

    def embed_batch(self, spec, target_size, items):
        embedding = [0.1] * spec.dims
        return [(embedding, spec.dims, "local", spec.name, target_size) for _ in items]

    def warmup(self, model_name=None):
        pass

    def get_device_info(self):
        return {"type": "cpu"}

    def get_model_status(self):
        return [
            {"name": "ViT-L-14", "loaded": True},
            {"name": "ViT-B-16", "loaded": False},
        ]

    def get_memory_info(self):
        return None

    def is_default_model_loaded(self):
        return True

    def get_cache_info(self):
        return None
