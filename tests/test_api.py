# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from fastapi.testclient import TestClient

from image_embedder.main import create_app
from fakes import _no_auth_settings, FakeEmbedder


def test_health():
    app = create_app(embedder=FakeEmbedder(), settings=_no_auth_settings())
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert data["device"]["type"] == "cpu"
    assert "models" in data
    assert len(data["models"]) == 2


def test_models():
    app = create_app(embedder=FakeEmbedder(), settings=_no_auth_settings())
    client = TestClient(app)
    response = client.get("/models")
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0]["id"] == "ViT-L-14"


def test_embed_image():
    app = create_app(embedder=FakeEmbedder(), settings=_no_auth_settings())
    client = TestClient(app)
    response = client.post("/embed-image", json={
        "image_url": "https://example.com/poster.jpg",
        "model": "ViT-L-14",
        "normalize": True,
        "image_size": 512
    })
    assert response.status_code == 200
    data = response.json()
    assert data["dims"] == 3
    assert data["model"] == "ViT-L-14"


def test_embed_image_requires_payload():
    app = create_app(embedder=FakeEmbedder(), settings=_no_auth_settings())
    client = TestClient(app)
    response = client.post("/embed-image", json={
        "model": "ViT-L-14"
    })
    assert response.status_code == 400


def test_ready():
    app = create_app(embedder=FakeEmbedder(), settings=_no_auth_settings())
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["default_model_loaded"] is True
    assert "device" in data
