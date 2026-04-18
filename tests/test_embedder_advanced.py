# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from image_embedder.config import Settings
from image_embedder.embedder import BatchItem, EmbeddingLRUCache, ImageEmbedder, MODEL_CATALOG


def _install_fake_torch_device_module(monkeypatch, *, cuda_available: bool, hip_version=None):
    fake_torch = types.SimpleNamespace()
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_name=lambda device: f"name:{device}",
        memory_allocated=lambda device=None: 8 * 1024 * 1024,
        memory_reserved=lambda device=None: 16 * 1024 * 1024,
    )
    fake_torch.device = lambda name: f"dev:{name}"
    fake_torch.version = types.SimpleNamespace(hip=hip_version)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


def _install_fake_openvino(monkeypatch, *, available_devices=None):
    state = {
        "available_devices": list(available_devices or ["CPU"]),
        "convert_calls": [],
        "save_calls": [],
        "read_calls": [],
        "compile_calls": [],
    }

    class FakeCore:
        def __init__(self):
            self.available_devices = list(state["available_devices"])

        def read_model(self, path):
            state["read_calls"].append(path)
            return f"read:{path}"

        def compile_model(self, model, device_name):
            state["compile_calls"].append((model, device_name))
            return {"compiled_model": model, "device": device_name}

    def convert_model(model, example_input):
        state["convert_calls"].append((model, example_input))
        return "converted-model"

    def save_model(model, path):
        state["save_calls"].append((model, path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("<xml/>", encoding="utf-8")

    fake_openvino = types.SimpleNamespace(
        Core=FakeCore,
        convert_model=convert_model,
        save_model=save_model,
    )
    monkeypatch.setitem(sys.modules, "openvino", fake_openvino)
    return state


def _install_fake_openvino_transformers(monkeypatch):
    calls = {"torch_model": 0, "processor": 0}

    class FakeTorchModel:
        @classmethod
        def from_pretrained(cls, _hf_id):
            calls["torch_model"] += 1
            return cls()

        def eval(self):
            return self

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, _hf_id):
            calls["processor"] += 1
            return cls()

    fake_transformers = types.SimpleNamespace(
        CLIPVisionModelWithProjection=FakeTorchModel,
        CLIPProcessor=FakeProcessor,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return calls


class _FakeOvProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, images, return_tensors, size):
        self.calls.append(
            {
                "images": images,
                "return_tensors": return_tensors,
                "size": size,
            }
        )
        return {"pixel_values": np.array([[1.0]], dtype=np.float32)}


def test_resolve_device_openvino_defaults_to_auto():
    embedder = ImageEmbedder(settings=Settings(device="openvino"))
    assert embedder._resolve_device() == "ov:AUTO"


def test_resolve_device_openvino_preserves_explicit_device():
    embedder = ImageEmbedder(settings=Settings(device="openvino:gpu.0"))
    assert embedder._resolve_device() == "ov:GPU.0"


def test_get_device_info_openvino_lists_available_devices(monkeypatch):
    embedder = ImageEmbedder(settings=Settings())
    monkeypatch.setattr(embedder, "_resolve_device", lambda: "ov:GPU.0")
    _install_fake_openvino(monkeypatch, available_devices=["CPU", "GPU.0"])

    info = embedder.get_device_info()

    assert info == {
        "type": "openvino",
        "ov_device": "GPU.0",
        "available_devices": ["CPU", "GPU.0"],
    }


def test_get_device_info_cuda_reports_rocm_when_hip_present(monkeypatch):
    embedder = ImageEmbedder(settings=Settings())
    monkeypatch.setattr(embedder, "_resolve_device", lambda: "cuda:0")
    _install_fake_torch_device_module(monkeypatch, cuda_available=True, hip_version="6.4.0")

    info = embedder.get_device_info()

    assert info == {
        "type": "rocm",
        "name": "name:cuda:0",
        "hip_version": "6.4.0",
    }


def test_get_memory_info_returns_cuda_metrics(monkeypatch):
    embedder = ImageEmbedder(settings=Settings())
    monkeypatch.setattr(embedder, "_resolve_device", lambda: "cuda:0")
    _install_fake_torch_device_module(monkeypatch, cuda_available=True)

    info = embedder.get_memory_info()

    assert info == {
        "allocated_mb": 8.0,
        "reserved_mb": 16.0,
    }


def test_get_cache_info_returns_none_when_disabled():
    embedder = ImageEmbedder(settings=Settings(embed_cache_size=0))
    assert embedder.get_cache_info() is None


def test_load_model_routes_openvino_to_openvino_loader(monkeypatch):
    embedder = ImageEmbedder(settings=Settings(device="openvino:gpu"))
    spec = next(iter(MODEL_CATALOG.values()))
    expected = object()

    monkeypatch.setattr(embedder, "_resolve_device", lambda: "ov:GPU")
    monkeypatch.setattr(embedder, "_load_model_openvino", lambda s, device: expected if (s, device) == (spec, "ov:GPU") else None)

    assert embedder._load_model(spec) is expected


def test_load_model_openvino_exports_model_when_cache_missing(monkeypatch, tmp_path):
    state = _install_fake_openvino(monkeypatch)
    calls = _install_fake_openvino_transformers(monkeypatch)
    monkeypatch.setenv("OV_MODEL_CACHE", str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(zeros=lambda *shape: ("zeros", shape)),
    )

    embedder = ImageEmbedder(settings=Settings())
    spec = next(iter(MODEL_CATALOG.values()))

    loaded = embedder._load_model_openvino(spec, "ov:GPU")

    xml_path = tmp_path / spec.name.replace("/", "_") / "model.xml"
    assert xml_path.exists()
    assert calls == {"torch_model": 1, "processor": 1}
    assert len(state["convert_calls"]) == 1
    assert state["save_calls"] == [("converted-model", str(xml_path))]
    assert state["compile_calls"] == [("converted-model", "GPU")]
    assert loaded == ({"compiled_model": "converted-model", "device": "GPU"}, loaded[1], "ov:GPU")


def test_load_model_openvino_uses_cached_ir_when_present(monkeypatch, tmp_path):
    state = _install_fake_openvino(monkeypatch)
    calls = _install_fake_openvino_transformers(monkeypatch)
    monkeypatch.setenv("OV_MODEL_CACHE", str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(zeros=lambda *shape: ("zeros", shape)),
    )

    spec = next(iter(MODEL_CATALOG.values()))
    cache_dir = tmp_path / spec.name.replace("/", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    xml_path = cache_dir / "model.xml"
    xml_path.write_text("<xml/>", encoding="utf-8")

    embedder = ImageEmbedder(settings=Settings())
    loaded = embedder._load_model_openvino(spec, "ov:CPU")

    assert calls == {"torch_model": 0, "processor": 1}
    assert state["convert_calls"] == []
    assert state["read_calls"] == [str(xml_path)]
    assert state["compile_calls"] == [(f"read:{xml_path}", "CPU")]
    assert loaded == ({"compiled_model": f"read:{xml_path}", "device": "CPU"}, loaded[1], "ov:CPU")


def test_embed_openvino_path_normalizes_and_uses_cache(monkeypatch):
    processor = _FakeOvProcessor()
    model_calls = {"count": 0}
    _VEC = np.array([[3.0, 4.0] + [0.0] * 766], dtype=np.float32)  # shape (1, 768)

    def _model(inputs):
        model_calls["count"] += 1
        return [_VEC.copy()]

    embedder = ImageEmbedder(settings=Settings(embed_cache_size=8))
    monkeypatch.setattr(embedder, "_load_model", lambda _spec: (_model, processor, "ov:GPU"))
    monkeypatch.setattr(embedder, "_resolve_image_bytes", lambda *_a, **_k: b"payload-a")
    monkeypatch.setattr(embedder, "_image_from_bytes", lambda _data: object())

    first = embedder.embed(
        image_url=None,
        image_base64="payload-a",
        model="ViT-L-14",
        normalize=True,
        image_size=224,
    )
    second = embedder.embed(
        image_url=None,
        image_base64="payload-a",
        model="ViT-L-14",
        normalize=True,
        image_size=224,
    )

    assert first == second
    assert first[0][:2] == pytest.approx([0.6, 0.8])
    assert first[1] == 768
    assert first[2:] == ("local", "ViT-L-14", 224)
    assert model_calls["count"] == 1
    assert processor.calls[0]["return_tensors"] == "np"


def test_embedding_cache_returns_fresh_lists_for_cached_results():
    cache = EmbeddingLRUCache(8)
    key = EmbeddingLRUCache.make_key(b"payload-a", "ViT-L-14", 224, True)
    original = ([1.0, 2.0], 2, "local", "ViT-L-14", 224)

    cache.put(key, original)

    first = cache.get(key)
    second = cache.get(key)

    assert first == original
    assert second == original
    assert first is not second
    assert first[0] is not second[0]

    first[0][0] = 999.0

    third = cache.get(key)

    assert third == original


def test_embed_cached_mutation_does_not_poison_future_hits(monkeypatch):
    processor = _FakeOvProcessor()
    model_calls = {"count": 0}
    _VEC = np.array([[3.0, 4.0] + [0.0] * 766], dtype=np.float32)  # shape (1, 768)
    _EXPECTED = [3.0, 4.0] + [0.0] * 766

    def _model(inputs):
        model_calls["count"] += 1
        return [_VEC.copy()]

    embedder = ImageEmbedder(settings=Settings(embed_cache_size=8))
    monkeypatch.setattr(embedder, "_load_model", lambda _spec: (_model, processor, "ov:GPU"))
    monkeypatch.setattr(embedder, "_resolve_image_bytes", lambda *_a, **_k: b"payload-a")
    monkeypatch.setattr(embedder, "_image_from_bytes", lambda _data: object())

    first = embedder.embed(
        image_url=None,
        image_base64="payload-a",
        model="ViT-L-14",
        normalize=False,
        image_size=224,
    )
    first[0][0] = 999.0

    second = embedder.embed(
        image_url=None,
        image_base64="payload-a",
        model="ViT-L-14",
        normalize=False,
        image_size=224,
    )
    assert second == (_EXPECTED, 768, "local", "ViT-L-14", 224)
    second[0][1] = 888.0

    third = embedder.embed(
        image_url=None,
        image_base64="payload-a",
        model="ViT-L-14",
        normalize=False,
        image_size=224,
    )

    assert third == (_EXPECTED, 768, "local", "ViT-L-14", 224)
    assert second[0] is not third[0]
    assert model_calls["count"] == 1


def test_embed_batch_returns_empty_list_for_no_items():
    embedder = ImageEmbedder(settings=Settings())
    spec = next(iter(MODEL_CATALOG.values()))
    assert embedder.embed_batch(spec, spec.image_size, []) == []


def test_embed_batch_all_cached_skips_model_loading(monkeypatch):
    embedder = ImageEmbedder(settings=Settings(embed_cache_size=8))
    spec = next(iter(MODEL_CATALOG.values()))
    items = [
        BatchItem(None, "payload-a", True),
        BatchItem(None, "payload-b", False),
    ]
    expected = [
        ([1.0, 2.0], 2, "local", spec.name, spec.image_size),
        ([3.0, 4.0], 2, "local", spec.name, spec.image_size),
    ]

    monkeypatch.setattr(
        embedder,
        "_resolve_image_bytes",
        lambda _image_url, image_base64: image_base64.encode("ascii"),
    )

    for item, result in zip(items, expected):
        key = embedder._embedding_cache.make_key(
            item.image_base64.encode("ascii"),
            spec.name,
            spec.image_size,
            item.normalize,
        )
        embedder._embedding_cache.put(key, result)

    monkeypatch.setattr(embedder, "_load_model", lambda _spec: (_ for _ in ()).throw(AssertionError("should not load model")))

    assert embedder.embed_batch(spec, spec.image_size, items) == expected


def test_embed_batch_openvino_mixed_cache_errors_and_cleanup(monkeypatch):
    processor = _FakeOvProcessor()
    cleanup_calls = []
    # Two 768-dim output rows: [3,4,0...] and [5,12,0...]
    _ROW0 = [3.0, 4.0] + [0.0] * 766
    _ROW1 = [5.0, 12.0] + [0.0] * 766

    def _model(inputs):
        return [np.array([_ROW0, _ROW1], dtype=np.float32)]

    def _resolve_image_bytes(_image_url, image_base64):
        return image_base64.encode("ascii")

    def _image_from_bytes(data):
        if data == b"bad":
            raise ValueError("corrupt image")
        return object()

    embedder = ImageEmbedder(settings=Settings(embed_cache_size=8, embed_cleanup_every_n=2))
    spec = next(iter(MODEL_CATALOG.values()))

    cached_result = ([9.0, 9.0], 2, "local", spec.name, spec.image_size)
    cached_item = BatchItem(None, "cached", False)
    cached_key = embedder._embedding_cache.make_key(
        b"cached",
        spec.name,
        spec.image_size,
        cached_item.normalize,
    )
    embedder._embedding_cache.put(cached_key, cached_result)

    items = [
        cached_item,
        BatchItem(None, "bad", True),
        BatchItem(None, "good-normalized", True),
        BatchItem(None, "good-raw", False),
    ]

    monkeypatch.setattr(embedder, "_load_model", lambda _spec: (_model, processor, "ov:GPU"))
    monkeypatch.setattr(embedder, "_resolve_image_bytes", _resolve_image_bytes)
    monkeypatch.setattr(embedder, "_image_from_bytes", _image_from_bytes)
    monkeypatch.setattr("image_embedder.memory.cleanup_gpu_memory", lambda device: cleanup_calls.append(device))

    outcomes = embedder.embed_batch(spec, spec.image_size, items)

    assert outcomes[0] == cached_result
    assert isinstance(outcomes[1], ValueError)
    assert str(outcomes[1]) == "corrupt image"
    assert outcomes[2][0][:2] == pytest.approx([0.6, 0.8])
    assert len(outcomes[2][0]) == 768
    assert outcomes[2][1:] == (768, "local", spec.name, spec.image_size)
    assert outcomes[3][0][:2] == pytest.approx([5.0, 12.0])
    assert len(outcomes[3][0]) == 768
    assert outcomes[3][1:] == (768, "local", spec.name, spec.image_size)
    assert cleanup_calls == ["ov:GPU"]

    normalized_key = embedder._embedding_cache.make_key(b"good-normalized", spec.name, spec.image_size, True)
    raw_key = embedder._embedding_cache.make_key(b"good-raw", spec.name, spec.image_size, False)
    cached_good_normalized = embedder._embedding_cache.get(normalized_key)
    cached_good_raw = embedder._embedding_cache.get(raw_key)
    assert cached_good_normalized is not None
    assert cached_good_raw is not None
    assert cached_good_normalized == outcomes[2]
    assert cached_good_raw == outcomes[3]


def test_embed_remote_url_cache_keys_follow_fetched_bytes(monkeypatch):
    processor = _FakeOvProcessor()
    model_calls = {"count": 0}
    fetched_payloads = iter([b"first", b"second", b"second"])

    def _fetch_image_bytes(_image_url):
        return next(fetched_payloads)

    def _model(inputs):
        model_calls["count"] += 1
        value = float(model_calls["count"])
        # Return a 768-dim vector with the distinguishing value at index 0
        row = np.zeros((1, 768), dtype=np.float32)
        row[0, 0] = value
        return [row]

    embedder = ImageEmbedder(settings=Settings(embed_cache_size=8))
    monkeypatch.setattr(embedder, "_fetch_image_bytes", _fetch_image_bytes)
    monkeypatch.setattr(embedder, "_load_model", lambda _spec: (_model, processor, "ov:GPU"))
    monkeypatch.setattr(embedder, "_image_from_bytes", lambda _data: object())

    first = embedder.embed(
        image_url="https://example.com/poster.jpg",
        image_base64=None,
        model="ViT-L-14",
        normalize=False,
        image_size=224,
    )
    second = embedder.embed(
        image_url="https://example.com/poster.jpg",
        image_base64=None,
        model="ViT-L-14",
        normalize=False,
        image_size=224,
    )
    third = embedder.embed(
        image_url="https://example.com/poster.jpg",
        image_base64=None,
        model="ViT-L-14",
        normalize=False,
        image_size=224,
    )

    assert first[0][0] == pytest.approx(1.0)
    assert second[0][0] == pytest.approx(2.0)
    assert third == second
    assert model_calls["count"] == 2
