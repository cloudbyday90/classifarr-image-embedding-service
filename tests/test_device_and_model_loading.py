# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import types

import pytest

from image_embedder.config import Settings
from image_embedder.embedder import ImageEmbedder, MODEL_CATALOG


def _install_fake_torch(monkeypatch, cuda_available: bool):
    fake_torch = types.SimpleNamespace()
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    fake_torch.device = lambda name: f"dev:{name}"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


def test_resolve_device_cpu(monkeypatch):
    _install_fake_torch(monkeypatch, cuda_available=True)
    embedder = ImageEmbedder(settings=Settings(device="cpu"))
    assert embedder._resolve_device() == "dev:cpu"


def test_resolve_device_cuda_unavailable_raises(monkeypatch):
    _install_fake_torch(monkeypatch, cuda_available=False)
    embedder = ImageEmbedder(settings=Settings(device="cuda"))
    with pytest.raises(ValueError, match="CUDA requested but not available"):
        embedder._resolve_device()

def test_resolve_device_cuda_when_available(monkeypatch):
    _install_fake_torch(monkeypatch, cuda_available=True)
    embedder = ImageEmbedder(settings=Settings(device="cuda"))
    assert embedder._resolve_device() == "dev:cuda"


def test_resolve_device_auto_uses_cuda_when_available(monkeypatch):
    _install_fake_torch(monkeypatch, cuda_available=True)
    embedder = ImageEmbedder(settings=Settings(device="auto"))
    assert embedder._resolve_device() == "dev:cuda"


def test_resolve_device_unknown_raises(monkeypatch):
    _install_fake_torch(monkeypatch, cuda_available=True)
    embedder = ImageEmbedder(settings=Settings(device="bogus"))
    with pytest.raises(ValueError, match="Unsupported DEVICE value"):
        embedder._resolve_device()


def test_load_model_is_cached(monkeypatch):
    embedder = ImageEmbedder(settings=Settings(device="cpu"))
    monkeypatch.setattr(embedder, "_resolve_device", lambda: "cpu")

    calls = {"model": 0, "proc": 0}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _hf_id):
            calls["model"] += 1
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, _hf_id):
            calls["proc"] += 1
            return cls()

    fake_transformers = types.SimpleNamespace(CLIPModel=FakeModel, CLIPProcessor=FakeProcessor)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    spec = next(iter(MODEL_CATALOG.values()))
    first = embedder._load_model(spec)
    second = embedder._load_model(spec)

    assert first is second
    assert calls["model"] == 1
    assert calls["proc"] == 1
