# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import base64
import io
import sys
import types

from PIL import Image

from image_embedder.config import Settings
from image_embedder.embedder import ImageEmbedder


def _png_base64() -> str:
    img = Image.new("RGB", (1, 1), (0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class FakeInput:
    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    def __call__(self, images, return_tensors, size):
        return {"pixel_values": FakeInput()}


class FakeNumpy:
    def __init__(self, arr):
        self._arr = arr

    def astype(self, _dtype):
        return self

    def tolist(self):
        return self._arr


class FakeTensor:
    def __init__(self, arr):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return FakeNumpy(self._arr)


class _FakeOutput:
    def __init__(self):
        self.image_embeds = [FakeTensor([1.0, 2.0, 3.0])]


class FakeModel:
    def __call__(self, **kwargs):
        return _FakeOutput()


def _install_fake_torch(monkeypatch):
    calls = {"normalize": 0}

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _normalize(features, p, dim):
        calls["normalize"] += 1
        return features

    fake_torch = types.SimpleNamespace(
        no_grad=lambda: _NoGrad(),
        nn=types.SimpleNamespace(functional=types.SimpleNamespace(normalize=_normalize)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return calls


def test_embed_flow_offline_with_mocked_torch_and_transformers(monkeypatch):
    calls = _install_fake_torch(monkeypatch)

    embedder = ImageEmbedder(settings=Settings(allow_remote_urls=False))

    monkeypatch.setattr(embedder, "_load_model", lambda _spec: (FakeModel(), FakeProcessor(), "cpu"))

    embedding, dims, provider, model_name, image_size = embedder.embed(
        image_url=None,
        image_base64=_png_base64(),
        model="ViT-L-14",
        normalize=True,
        image_size=224,
    )

    assert provider == "local"
    assert model_name == "ViT-L-14"
    assert image_size == 224
    assert dims == 3
    assert embedding == [1.0, 2.0, 3.0]
    assert calls["normalize"] == 1


def test_embed_cleanup_runs_on_configured_cadence(monkeypatch):
    _install_fake_torch(monkeypatch)

    cleanup_calls = []

    def _cleanup(device):
        cleanup_calls.append(device)
        return {"gc_collected": 0, "gpu_freed_mb": 0.0}

    embedder = ImageEmbedder(
        settings=Settings(allow_remote_urls=False, embed_cleanup_every_n=2, embed_cache_size=0)
    )

    monkeypatch.setattr(embedder, "_load_model", lambda _spec: (FakeModel(), FakeProcessor(), "cpu"))
    monkeypatch.setattr("image_embedder.memory.cleanup_gpu_memory", _cleanup)

    for _ in range(3):
        embedder.embed(
            image_url=None,
            image_base64=_png_base64(),
            model="ViT-L-14",
            normalize=False,
            image_size=224,
        )

    assert cleanup_calls == ["cpu"]


def test_embed_cleanup_disabled_by_default(monkeypatch):
    _install_fake_torch(monkeypatch)

    cleanup_calls = []

    def _cleanup(device):
        cleanup_calls.append(device)
        return {"gc_collected": 0, "gpu_freed_mb": 0.0}

    embedder = ImageEmbedder(settings=Settings(allow_remote_urls=False))

    monkeypatch.setattr(embedder, "_load_model", lambda _spec: (FakeModel(), FakeProcessor(), "cpu"))
    monkeypatch.setattr("image_embedder.memory.cleanup_gpu_memory", _cleanup)

    embedder.embed(
        image_url=None,
        image_base64=_png_base64(),
        model="ViT-L-14",
        normalize=False,
        image_size=224,
    )

    assert cleanup_calls == []

