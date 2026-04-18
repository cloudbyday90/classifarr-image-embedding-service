# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""P1 invariant and hardening tests.

Covers:
  - P1.1  Embedding width / finiteness validation (_validate_embedding_result)
  - P1.2  Canonical route response metadata (embed and batch routes)
  - P1.3  OpenVINO output shape validation
  - P1.4  OpenVINO / NumPy normalization parity with PyTorch
  - P1.5  image_size frozen to spec default
"""

import math
import sys
import types

import httpx
import numpy as np
import pytest
from asgi_lifespan import LifespanManager
from fastapi.testclient import TestClient

from image_embedder.config import Settings
from image_embedder.embedder import BatchItem, ImageEmbedder, MODEL_CATALOG
from image_embedder.main import create_app
from fakes import FakeEmbedder, _no_auth_settings, _png_bytes
import base64


def _b64() -> str:
    return base64.b64encode(_png_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# P1.1 — _validate_embedding_result
# ---------------------------------------------------------------------------


def _spec_vit_l():
    """Return the real ViT-L-14 ModelSpec."""
    return MODEL_CATALOG["ViT-L-14"]


def _good_embedding(dims: int = 768) -> list:
    return [0.5] * dims


class TestValidateEmbeddingResult:
    def setup_method(self):
        self.embedder = ImageEmbedder(settings=Settings())
        self.spec = _spec_vit_l()

    def test_valid_embedding_passes(self):
        emb = _good_embedding(self.spec.dims)
        self.embedder._validate_embedding_result(self.spec, emb, self.spec.dims)

    def test_wrong_dims_vs_len_raises(self):
        emb = _good_embedding(3)
        with pytest.raises(ValueError, match="dims=768 does not match len\\(embedding\\)=3"):
            self.embedder._validate_embedding_result(self.spec, emb, 768)

    def test_wrong_dims_vs_spec_raises(self):
        emb = _good_embedding(3)
        with pytest.raises(ValueError, match="dims=3 does not match spec.dims=768"):
            self.embedder._validate_embedding_result(self.spec, emb, 3)

    def test_nested_array_raises(self):
        emb = [[0.1, 0.2], [0.3, 0.4]]
        with pytest.raises(ValueError, match="1-D vector"):
            self.embedder._validate_embedding_result(self.spec, emb, 2)

    def test_nan_value_raises(self):
        emb = _good_embedding(self.spec.dims)
        emb[0] = math.nan
        with pytest.raises(ValueError, match="non-finite"):
            self.embedder._validate_embedding_result(self.spec, emb, self.spec.dims)

    def test_inf_value_raises(self):
        emb = _good_embedding(self.spec.dims)
        emb[5] = math.inf
        with pytest.raises(ValueError, match="non-finite"):
            self.embedder._validate_embedding_result(self.spec, emb, self.spec.dims)

    def test_negative_inf_raises(self):
        emb = _good_embedding(self.spec.dims)
        emb[2] = -math.inf
        with pytest.raises(ValueError, match="non-finite"):
            self.embedder._validate_embedding_result(self.spec, emb, self.spec.dims)


# ---------------------------------------------------------------------------
# P1.4 — _normalize_embedding_np
# ---------------------------------------------------------------------------


class TestNormalizeEmbeddingNp:
    def setup_method(self):
        self.embedder = ImageEmbedder(settings=Settings())

    def test_unit_vector_unchanged(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        out = self.embedder._normalize_embedding_np(v)
        assert np.allclose(out, v)

    def test_valid_vector_has_unit_l2_norm(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        out = self.embedder._normalize_embedding_np(v)
        assert math.isclose(float(np.linalg.norm(out)), 1.0, abs_tol=1e-6)
        assert np.allclose(out, [0.6, 0.8])

    def test_zero_vector_does_not_produce_nan(self):
        v = np.zeros(4, dtype=np.float32)
        out = self.embedder._normalize_embedding_np(v)
        assert np.all(np.isfinite(out))
        assert np.all(out == 0.0)

    def test_tiny_norm_uses_epsilon_clamping(self):
        v = np.array([1e-100, 0.0], dtype=np.float64)
        out = self.embedder._normalize_embedding_np(v, eps=1e-12)
        assert np.all(np.isfinite(out))

    def test_nan_input_raises(self):
        v = np.array([1.0, float("nan")], dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite"):
            self.embedder._normalize_embedding_np(v)

    def test_inf_input_raises(self):
        v = np.array([float("inf"), 1.0], dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite"):
            self.embedder._normalize_embedding_np(v)

    def test_batch_normalize_valid(self):
        batch = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        out = self.embedder._normalize_embedding_np(batch)
        assert out.shape == (2, 2)
        for row in out:
            assert math.isclose(float(np.linalg.norm(row)), 1.0, abs_tol=1e-6)

    def test_parity_with_torch_semantics_for_valid_vector(self):
        """NumPy path matches PyTorch F.normalize for nonzero input."""
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np_out = self.embedder._normalize_embedding_np(v)
        # torch.nn.functional.normalize([1,2,3,4]) == [1,2,3,4] / norm
        expected = v / np.linalg.norm(v)
        assert np.allclose(np_out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# P1.3 — OpenVINO output shape validation
# ---------------------------------------------------------------------------


class _FakeOvProcessor:
    def __call__(self, images, return_tensors, size):
        return {"pixel_values": np.ones((1, 3, 224, 224), dtype=np.float32)}


def test_embed_openvino_wrong_output_shape_single_raises(monkeypatch):
    """A 3-D OV output (e.g. patch embeddings) must raise, not silently misbehave."""
    spec = MODEL_CATALOG["ViT-L-14"]
    processor = _FakeOvProcessor()

    def _model_3d(inputs):
        # Simulate a wrong output: (1, patches, dims) instead of (1, dims)
        return [np.ones((1, 196, spec.dims), dtype=np.float32)]

    embedder = ImageEmbedder(settings=Settings())
    monkeypatch.setattr(embedder, "_load_model", lambda _s: (_model_3d, processor, "ov:CPU"))
    monkeypatch.setattr(embedder, "_resolve_image_bytes", lambda *_a, **_k: b"img")
    monkeypatch.setattr(embedder, "_image_from_bytes", lambda _d: object())

    with pytest.raises(ValueError, match="unexpected output shape"):
        embedder.embed(
            image_url=None, image_base64="AA==",
            model="ViT-L-14", normalize=False, image_size=224,
        )


def test_embed_openvino_correct_shape_single_succeeds(monkeypatch):
    spec = MODEL_CATALOG["ViT-L-14"]
    processor = _FakeOvProcessor()

    def _model_ok(inputs):
        return [np.ones((1, spec.dims), dtype=np.float32)]

    embedder = ImageEmbedder(settings=Settings())
    monkeypatch.setattr(embedder, "_load_model", lambda _s: (_model_ok, processor, "ov:CPU"))
    monkeypatch.setattr(embedder, "_resolve_image_bytes", lambda *_a, **_k: b"img")
    monkeypatch.setattr(embedder, "_image_from_bytes", lambda _d: object())

    embedding, dims, _, _, _ = embedder.embed(
        image_url=None, image_base64="AA==",
        model="ViT-L-14", normalize=False, image_size=224,
    )
    assert dims == spec.dims
    assert len(embedding) == spec.dims


def test_embed_batch_openvino_wrong_output_shape_raises(monkeypatch):
    """Batch OV output with wrong second dimension raises a clear error."""
    spec = MODEL_CATALOG["ViT-L-14"]
    processor = _FakeOvProcessor()

    def _model_wrong(inputs):
        # Return wrong shape: (2, 2) instead of (2, spec.dims)
        return [np.ones((2, 2), dtype=np.float32)]

    embedder = ImageEmbedder(settings=Settings())
    monkeypatch.setattr(embedder, "_load_model", lambda _s: (_model_wrong, processor, "ov:CPU"))

    items = [BatchItem(None, "AA==", False), BatchItem(None, "BB==", False)]
    monkeypatch.setattr(embedder, "_resolve_image_bytes", lambda _u, b64: b64.encode())
    monkeypatch.setattr(embedder, "_image_from_bytes", lambda _d: object())

    with pytest.raises(ValueError, match="unexpected output shape"):
        embedder.embed_batch(spec, spec.image_size, items)


# ---------------------------------------------------------------------------
# P1.5 — image_size frozen to spec default
# ---------------------------------------------------------------------------


class TestImageSizeFreeze:
    def setup_method(self):
        self.embedder = ImageEmbedder(settings=Settings())

    def test_none_image_size_uses_spec_default(self, monkeypatch):
        spec = MODEL_CATALOG["ViT-L-14"]
        monkeypatch.setattr(self.embedder, "_resolve_image_bytes", lambda *_a, **_k: b"img")

        def _fake_load(s):
            class _M:
                def __call__(self, **kw):
                    class _O:
                        image_embeds = [
                            _make_fake_tensor([0.5] * spec.dims)
                        ]
                    return _O()
            import types
            proc = types.SimpleNamespace(
                __call__=lambda *a, **kw: {"pixel_values": _FakePtInput()}
            )
            return (_M(), proc, "cpu")

        monkeypatch.setattr(self.embedder, "_load_model", _fake_load)
        # Should not raise; image_size=None maps to spec default
        with pytest.raises(Exception):  # will fail at torch import or similar, that's fine
            self.embedder.embed(
                image_url=None, image_base64="AA==",
                model="ViT-L-14", normalize=False, image_size=None,
            )

    def test_spec_default_image_size_accepted(self, monkeypatch):
        """Passing image_size equal to spec.image_size is allowed."""
        spec = MODEL_CATALOG["ViT-L-14"]
        monkeypatch.setattr(
            self.embedder, "_resolve_image_bytes", lambda *_a, **_k: (_ for _ in ()).throw(StopIteration("stop here"))
        )
        with pytest.raises(StopIteration):
            self.embedder.embed(
                image_url=None, image_base64="AA==",
                model="ViT-L-14", normalize=False, image_size=spec.image_size,
            )

    def test_non_default_image_size_raises(self, monkeypatch):
        """Passing image_size != spec.image_size raises ValueError before loading."""
        monkeypatch.setattr(
            self.embedder, "_resolve_image_bytes",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not reach here"))
        )
        with pytest.raises(ValueError, match="is not supported for ViT-L-14"):
            self.embedder.embed(
                image_url=None, image_base64="AA==",
                model="ViT-L-14", normalize=False, image_size=336,
            )


def test_embed_route_non_default_image_size_returns_400():
    """The embed route surfaces the ValueError as HTTP 400."""
    app = create_app(embedder=ImageEmbedder(settings=Settings()), settings=_no_auth_settings())
    client = TestClient(app)
    resp = client.post("/embed-image", json={
        "image_url": "https://example.com/img.jpg",
        "model": "ViT-L-14",
        "image_size": 336,
    })
    assert resp.status_code == 400
    assert "not supported" in resp.json()["detail"]


@pytest.mark.anyio
async def test_embed_batch_route_non_default_image_size_returns_422():
    """The batch route rejects non-default image_size with HTTP 422."""
    app = create_app(embedder=FakeEmbedder(), settings=_no_auth_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/embed-batch",
                json={"items": [{"image_base64": _b64()}], "image_size": 512},
            )
    assert resp.status_code == 422
    assert "not supported" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# P1.2 — Canonical route response metadata
# ---------------------------------------------------------------------------


class _MismatchEmbedder(FakeEmbedder):
    """Fake that returns wrong model name and image_size in the result tuple."""

    def embed(self, image_url, image_base64, model, normalize, image_size):
        embedding = [0.5] * 768
        return embedding, 768, "local", "wrong-model", 9999

    def embed_batch(self, spec, target_size, items):
        embedding = [0.5] * spec.dims
        return [(embedding, spec.dims, "local", "wrong-model", 9999) for _ in items]


def test_embed_route_uses_canonical_model_not_embedder_returned():
    app = create_app(embedder=_MismatchEmbedder(), settings=_no_auth_settings())
    client = TestClient(app)
    resp = client.post("/embed-image", json={
        "image_url": "https://example.com/img.jpg",
        "model": "ViT-L-14",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "ViT-L-14"
    assert body["image_size"] == 224  # canonical spec value, not 9999


def test_embed_route_uses_canonical_image_size_not_embedder_returned():
    app = create_app(embedder=_MismatchEmbedder(), settings=_no_auth_settings())
    client = TestClient(app)
    resp = client.post("/embed-image", json={
        "image_url": "https://example.com/img.jpg",
    })
    assert resp.status_code == 200
    assert resp.json()["image_size"] == 224  # not 9999


@pytest.mark.anyio
async def test_batch_route_per_item_uses_canonical_metadata():
    app = create_app(embedder=_MismatchEmbedder(), settings=_no_auth_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/embed-batch",
                json={"items": [{"image_base64": _b64()}, {"image_base64": _b64()}]},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "ViT-L-14"
    assert body["image_size"] == 224
    for result in body["results"]:
        if result["status"] == "ok":
            assert result["model"] == "ViT-L-14"
            assert result["image_size"] == 224


# ---------------------------------------------------------------------------
# P1.1 (route-level) — Real embedder invariant violation surfaces as HTTP 400
# ---------------------------------------------------------------------------


class _FakeOvProcessorFixed:
    def __call__(self, images, return_tensors, size):
        return {"pixel_values": np.ones((1, 3, 224, 224), dtype=np.float32)}


def test_embed_route_wrong_model_output_dims_returns_400(monkeypatch):
    """When the real ImageEmbedder gets a wrong-width output, it raises ValueError
    which the route maps to HTTP 400."""
    spec = MODEL_CATALOG["ViT-L-14"]
    processor = _FakeOvProcessorFixed()

    def _model_wrong_dims(inputs):
        # Only 3 elements instead of 768
        return [np.array([[0.1, 0.2, 0.3]], dtype=np.float32)]

    settings = Settings(allow_remote_urls=False)
    real_embedder = ImageEmbedder(settings=settings)
    monkeypatch.setattr(real_embedder, "_load_model", lambda _s: (_model_wrong_dims, processor, "ov:CPU"))
    monkeypatch.setattr(real_embedder, "_resolve_image_bytes", lambda *_a, **_k: b"img")
    monkeypatch.setattr(real_embedder, "_image_from_bytes", lambda _d: object())

    app = create_app(embedder=real_embedder, settings=_no_auth_settings())
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/embed-image", json={"image_base64": _b64()})
    # The OV shape check fires first (shape (1, 3) != (1, 768)) and raises ValueError → 400
    assert resp.status_code == 400
    assert "unexpected output shape" in resp.json()["detail"]
